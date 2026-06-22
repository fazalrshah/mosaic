# 🧩 Mosaic — a coverage-verified local RAG indexer

**Most RAG indexers fail silently.** A crash mid-batch, a dropped insert, or an OCR miss loses chunks —
and you don't find out until weeks later, when retrieval is quietly, mysteriously bad. Mosaic fixes that
with one stubborn guarantee:

> **A document is only "indexed" when every chunk is provably in the vector store.**
> After upserting, Mosaic queries the store and asserts `stored_count == chunk_count`. If even one tile
> is missing, the job **fails loudly** instead of reporting a false success.

100% local. No external APIs. Runs on Apple MPS, CUDA, or CPU.

---

## Why Mosaic is different

| Typical RAG indexer | Mosaic |
|---|---|
| Fire-and-forget insert | **Coverage check**: `stored == chunks` or it fails |
| One opaque `try/except` | **Block-structured stages** — failures name the exact stage |
| `pypdf` (silently empty on scanned PDFs) | **Docling** extraction with **OCR** |
| Fixed character windows | **Token-aware, structure-aware** chunking (HybridChunker) |
| Dense vectors only | **Dense + native BM25** (hybrid-ready) out of the box |
| "Did it work?" 🤷 | `VERIFIED 84/84` ✅ |

## How it works

```
extract (Docling, OCR)  →  chunk (HybridChunker, token-aware, section metadata)
  →  embed (sentence-transformers)  →  upsert (Milvus: dense + BM25)
  →  verify_coverage (stored == chunks, else FAIL)  →  done
```

Each stage raises a `StageError(stage, message)`, so a failure tells you *exactly* where it broke
(`extract`, `chunk`, `embed`, `upsert`, or `verify`) — never a silent hang.

## Quickstart

```bash
# 1. A running Milvus 2.5+  (https://milvus.io/docs/install_standalone-docker.md)
# 2. Install
pip install -r requirements.txt

# 3. Index a folder (or a single file) — every chunk verified
python mosaic.py index ./docs --collection mydocs
python mosaic.py index report.pdf --collection mydocs --doc-id q3_report
```

Output:
```
  OK   q3_report.pdf -> 14 chunks, VERIFIED 14/14
  FAIL scanned_only.pdf at stage 'extract': empty text after OCR — needs a better OCR pass
Mosaic done: 1 indexed, 1 failed.
```

As a library:
```python
from mosaic import Mosaic
m = Mosaic()                                    # loads the model once
r = m.index_file("report.pdf", collection="mydocs", doc_id="q3_report")
assert r["coverage_ok"], r                      # stored == chunks, guaranteed
```

## Configuration (env)

| Var | Default | Notes |
|---|---|---|
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | your Milvus |
| `EMBED_MODEL` | `BAAI/bge-m3` | any sentence-transformers model |
| `EMBED_DEVICE` | auto | `mps` / `cuda` / `cpu` |
| `EMBED_DIM` | `1024` | must match the model |
| `CHUNK_TOKENS` | `800` | keep 500–1000 |
| `ENABLE_BM25` | `1` | Milvus 2.5+ native full-text (set `0` for dense-only) |

See [`.env.example`](.env.example).

## Hard-won lessons baked in

These cost real debugging time; Mosaic encodes them so you don't repeat them:

- **Coverage is the whole point.** "It ran without error" ≠ "every chunk is in the store." Always verify
  `stored == chunks` *after* the insert/flush, with a short retry for consistency lag.
- **Scanned PDFs need OCR.** `pypdf` returns empty text on image-only PDFs → blank chunks → missing
  context. Docling does OCR, so those documents actually make it in (or fail loudly at `extract`).
- **Chunk in tokens, not characters.** `1200 chars ≈ 300 tokens` — far below a sane 500–1000 target.
  Measure with the embedding model's *own* tokenizer.
- **Embeddings are local.** `bge-m3` runs great on Apple MPS via `sentence-transformers` — no Ollama, no
  hosted API.
- **Attribute failures to a stage.** A single `try/except` around a long pipeline tells you nothing;
  per-stage errors turn "it hangs" into "it failed at `extract`."

## Roadmap

- [ ] Hybrid retrieval helper (dense + BM25 → RRF) + cross-encoder reranking
- [ ] Pluggable job sources (directory watch, Postgres queue, S3)
- [ ] Per-chunk ACL metadata + filtered retrieval
- [ ] Parent-document (hierarchical) retrieval using the stored section paths

## License

MIT — see [LICENSE](LICENSE).
