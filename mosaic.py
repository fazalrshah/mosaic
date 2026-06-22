"""
Mosaic — a coverage-verified local RAG indexer.

Most RAG indexers fail silently: a crash mid-batch, a dropped insert, or an OCR miss loses chunks, and
you only discover it later as mysteriously bad retrieval. Mosaic guarantees every chunk lands — after
upserting it queries the vector store and asserts `stored_count == chunk_count`, and REFUSES to report
success if even one tile is missing.

Pipeline (block-structured; each stage names itself on failure so you know exactly what broke):
    extract (Docling, OCR)  ->  chunk (HybridChunker, token-aware)  ->  embed (sentence-transformers)
    ->  upsert (Milvus: dense + optional native BM25)  ->  verify_coverage  ->  done

100% local. No external APIs. Apple MPS / CUDA / CPU.

Usage:
    python mosaic.py index ./docs --collection mydocs
    python mosaic.py index report.pdf --collection mydocs --doc-id q3_report

Library:
    from mosaic import index_file, MosaicConfig
    result = index_file("report.pdf", collection="mydocs", doc_id="q3_report")
    assert result["coverage_ok"]
"""
from __future__ import annotations
import os, re, glob, time, argparse, dataclasses

from pymilvus import (connections, utility, Collection, CollectionSchema, FieldSchema,
                      DataType, Function, FunctionType)
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker


@dataclasses.dataclass
class MosaicConfig:
    milvus_host: str = os.environ.get("MILVUS_HOST", "localhost")
    milvus_port: str = os.environ.get("MILVUS_PORT", "19530")
    embed_model: str = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
    embed_device: str = os.environ.get("EMBED_DEVICE", "")          # "" = auto (mps/cuda/cpu)
    embed_dim: int = int(os.environ.get("EMBED_DIM", "1024"))       # bge-m3 = 1024
    chunk_tokens: int = int(os.environ.get("CHUNK_TOKENS", "800"))  # keep 500-1000
    embed_batch: int = int(os.environ.get("EMBED_BATCH", "16"))
    embed_retries: int = int(os.environ.get("EMBED_RETRIES", "3"))
    verify_retries: int = int(os.environ.get("VERIFY_RETRIES", "5"))
    enable_bm25: bool = os.environ.get("ENABLE_BM25", "1") not in ("0", "false", "no")


class StageError(Exception):
    """Carries which pipeline stage failed, so failures are attributable instead of mysterious."""
    def __init__(self, stage: str, msg):
        self.stage, self.msg = stage, str(msg)
        super().__init__(f"[{stage}] {self.msg}")


def _auto_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Mosaic:
    """Reusable indexer. Construct once (loads the model), then index many files."""

    def __init__(self, cfg: MosaicConfig | None = None):
        self.cfg = cfg or MosaicConfig()
        device = self.cfg.embed_device or _auto_device()
        print(f"[mosaic] loading {self.cfg.embed_model} on {device} ...", flush=True)
        self.model = SentenceTransformer(self.cfg.embed_model, device=device)
        try:
            self.tokenizer = self.model.tokenizer
        except Exception:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.embed_model)
        self.converter = DocumentConverter()
        self.chunker = self._make_chunker()
        connections.connect(alias="default", host=self.cfg.milvus_host, port=self.cfg.milvus_port)
        print("[mosaic] ready", flush=True)

    def _make_chunker(self):
        try:
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            return HybridChunker(
                tokenizer=HuggingFaceTokenizer(tokenizer=self.tokenizer, max_tokens=self.cfg.chunk_tokens),
                merge_peers=True)
        except Exception:
            pass
        try:
            return HybridChunker(tokenizer=self.tokenizer, max_tokens=self.cfg.chunk_tokens, merge_peers=True)
        except Exception:
            return HybridChunker(tokenizer=self.cfg.embed_model, max_tokens=self.cfg.chunk_tokens)

    # ---- stages ----------------------------------------------------------
    def _extract_chunk(self, path: str):
        try:
            dl_doc = self.converter.convert(path).document
        except Exception as e:
            raise StageError("extract", f"docling convert failed: {e}")
        out = []
        try:
            for ch in self.chunker.chunk(dl_doc):
                raw = (getattr(ch, "text", "") or "").strip()
                if not raw:
                    continue
                try:
                    ctx = self.chunker.contextualize(chunk=ch)
                except Exception:
                    ctx = raw
                section = ""
                try:
                    hs = getattr(ch.meta, "headings", None)
                    if hs:
                        section = " > ".join(h for h in hs if h)[:2048]
                except Exception:
                    pass
                out.append({"raw": raw[:65535], "ctx": (ctx or raw)[:65535], "section": section})
        except Exception as e:
            raise StageError("chunk", f"HybridChunker failed: {e}")
        if not out:
            raise StageError("chunk", "no chunks produced (empty parse / OCR yielded nothing)")
        return out

    def _embed(self, chunks):
        texts = [c["ctx"] for c in chunks]
        vecs = []
        for start in range(0, len(texts), self.cfg.embed_batch):
            batch = texts[start:start + self.cfg.embed_batch]
            last_err = None
            for attempt in range(self.cfg.embed_retries):
                try:
                    vecs.extend(self.model.encode(batch, normalize_embeddings=True,
                                                  batch_size=self.cfg.embed_batch).tolist())
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1 + attempt)
            if last_err is not None:
                raise StageError("embed", f"batch {start} failed after retries: {last_err}")
        if len(vecs) != len(chunks):
            raise StageError("embed", f"vector/chunk mismatch: {len(vecs)} vs {len(chunks)}")
        return vecs

    def _ensure_collection(self, name: str):
        if utility.has_collection(name):
            return Collection(name)
        fields = [
            FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("doc_id", DataType.VARCHAR, max_length=256),
            FieldSchema("version", DataType.INT64),
            FieldSchema("chunk", DataType.INT64),
            FieldSchema("text", DataType.VARCHAR, max_length=65535, enable_analyzer=self.cfg.enable_bm25),
            FieldSchema("section", DataType.VARCHAR, max_length=2048),
            FieldSchema("source", DataType.VARCHAR, max_length=512),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=self.cfg.embed_dim),
        ]
        functions = []
        if self.cfg.enable_bm25:
            fields.append(FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR))
            functions.append(Function(name="text_bm25", function_type=FunctionType.BM25,
                                      input_field_names=["text"], output_field_names=["sparse"]))
        schema = CollectionSchema(fields, description="Mosaic RAG", functions=functions)
        col = Collection(name, schema)
        col.create_index("vector", {"index_type": "HNSW", "metric_type": "COSINE",
                                    "params": {"M": 16, "efConstruction": 200}})
        if self.cfg.enable_bm25:
            col.create_index("sparse", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"})
        col.load()
        return col

    def _upsert(self, col, doc_id, version, source, chunks, vecs):
        try:
            col.delete(expr=f'doc_id == "{doc_id}"')
            rows = [{"doc_id": doc_id, "version": int(version), "chunk": i,
                     "text": c["raw"], "section": c["section"], "source": source, "vector": v}
                    for i, (c, v) in enumerate(zip(chunks, vecs))]
            col.insert(rows)
            col.flush()
        except Exception as e:
            raise StageError("upsert", f"milvus upsert failed: {e}")

    def _verify(self, col, doc_id, version, expected):
        stored = -1
        for _ in range(self.cfg.verify_retries):
            try:
                rows = col.query(expr=f'doc_id == "{doc_id}" and version == {int(version)}',
                                 output_fields=["chunk"], consistency_level="Strong")
                stored = len(rows)
            except Exception as e:
                raise StageError("verify", f"milvus query failed: {e}")
            if stored == expected:
                return stored
            time.sleep(1)
        raise StageError("verify", f"COVERAGE MISMATCH: stored {stored} of {expected} chunks")

    def index_file(self, path: str, collection: str, doc_id: str = "", version: int = 1) -> dict:
        """Index one file with a coverage guarantee. Raises StageError (with the failing stage) on any
        problem; on success, stored == chunk count is verified."""
        doc_id = doc_id or os.path.splitext(os.path.basename(path))[0]
        chunks = self._extract_chunk(path)
        vecs = self._embed(chunks)
        col = self._ensure_collection(collection)
        self._upsert(col, doc_id, version, os.path.basename(path), chunks, vecs)
        stored = self._verify(col, doc_id, version, len(chunks))
        return {"doc_id": doc_id, "version": version, "source": os.path.basename(path),
                "chunks": len(chunks), "stored": stored, "coverage_ok": stored == len(chunks),
                "collection": collection}


def index_file(path: str, collection: str, doc_id: str = "", version: int = 1,
               cfg: MosaicConfig | None = None) -> dict:
    """One-shot helper (constructs a Mosaic each call; for many files construct Mosaic once)."""
    return Mosaic(cfg).index_file(path, collection, doc_id, version)


def _cli():
    ap = argparse.ArgumentParser(description="Mosaic — coverage-verified local RAG indexer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    idx = sub.add_parser("index", help="index a file or a directory")
    idx.add_argument("path", help="file or directory of documents")
    idx.add_argument("--collection", required=True)
    idx.add_argument("--doc-id", default="", help="logical id (single file only; default = filename)")
    idx.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    m = Mosaic()
    if os.path.isdir(args.path):
        paths = [p for p in glob.glob(os.path.join(args.path, "**", "*"), recursive=True) if os.path.isfile(p)]
    else:
        paths = [args.path]
    ok = fail = 0
    for p in paths:
        try:
            r = m.index_file(p, args.collection, args.doc_id if not os.path.isdir(args.path) else "",
                             args.version)
            print(f"  OK   {r['source']} -> {r['chunks']} chunks, VERIFIED {r['stored']}/{r['chunks']}")
            ok += 1
        except StageError as e:
            print(f"  FAIL {os.path.basename(p)} at stage '{e.stage}': {e.msg}")
            fail += 1
    print(f"\nMosaic done: {ok} indexed, {fail} failed.")


if __name__ == "__main__":
    _cli()
