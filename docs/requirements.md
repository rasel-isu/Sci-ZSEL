# Requirements

> Part of the [Sci-ZSEL](../README.md) documentation.

**Hardware.** The reference runs used a single **NVIDIA A100-SXM4-80GB**. `bert-large-uncased`
with `--train_batch_size 64` and `--max_seq_length 192` needs ~40 GB; every training script passes
`--data_parallel`, so multiple smaller GPUs also work (ReS reads its GPU list from `res.gpus`,
which ships as `0,1,2,3` — see [Troubleshooting](troubleshooting.md)).
Alias generation needs a GPU for the local Ollama server, or will fall back to CPU (much slower).

**Disk.** ~10 GB for the pretrained checkpoints, plus ~2.7 GB per fine-tuned bi-encoder,
~1.3 GB per cross-encoder epoch, and ~1.5 GB per ReS epoch.

**Software.** Two conda environments, because BLINK and ReS need incompatible `transformers`
versions:

| env | Python | torch | transformers | used by |
|---|---|---|---|---|
| `sci-zsel` | 3.11 | 2.x (cu12/cu13) | 4.31.0 | data preparation, BLINK retriever + cross-encoder |
| `sci-zsel-res` | 3.9 | 2.0.1 (cu117) | 4.30.2 | ReS reranker |

Reference versions actually used: `sci-zsel` = Python 3.11.15 / torch 2.13.0+cu130 / transformers
4.31.0; `sci-zsel-res` = Python 3.9.25 / torch 2.0.1+cu117 / transformers 4.30.2. Note that
`modeling/BLINK/requirements.txt` does not pin `torch` — it arrives as a dependency of
`sentence-transformers`, so pin it yourself if you need bit-level reproducibility.

**[Ollama](https://ollama.com) is required** — it serves the alias-generation LLM in step 5.2, and
`data_preparation/alias_generation.sh` starts it, waits for it, and shuts it down. Install it before
running the pipeline (see [Installation](../README.md#2-installation)). Steps 5.1, 5.3 and 5.4 do not need it.

**External services / models downloaded at runtime:**

- `bert-large-uncased` (Hugging Face) — retriever and cross-encoder backbone
- `roberta-base` (Hugging Face) — ReS backbone
- `FremyCompany/BioLORD-2023-M` (Hugging Face) — similarity model for the ontology-aware filter
- `llama3.2:3b-instruct-fp16` (Ollama) — alias generation LLM

On an offline cluster, pre-cache these and set `HF_HOME` / `OLLAMA_MODELS` accordingly.
