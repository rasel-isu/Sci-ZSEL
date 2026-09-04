# Machine-specific paths you MUST edit

> Part of the [Sci-ZSEL](../README.md) documentation.

Every shell script activates conda by absolute path, and two files hard-code cache directories from
the original machine. **The pipeline will not run until these are pointed at your own system.**

| File | Line(s) | What to change |
|---|---|---|
| `install.sh` | 1–2, 4, 10 | conda hook path, `profile.d/conda.sh`, and the two `--prefix` env locations |
| `data_preparation/grag_to_blink.sh` | 1–2 | conda hook + `profile.d/conda.sh` |
| `data_preparation/entity_selection_and_pseudo_pair_construction.sh` | 1–2 | same |
| `data_preparation/alias_generation.sh` | 22–23 | same |
| `data_preparation/run_ollama_to_serve_llm` | 1 | `OLLAMA_MODELS` — where Ollama keeps its blobs |
| `data_preparation/pseudo_pair_construction.py` | 192 | `cache_dir` for the BioLORD sentence-transformer |
| `modeling/BLINK/scripts/get_biencoder_top_k.sh` | 1–2 | conda hook + `profile.d/conda.sh` |
| `modeling/BLINK/scripts/get_biencoder_cands_for_reranker_training.sh` | 1–2 | same |
| `modeling/BLINK/scripts/retriever_fine_tuning.sh` | 1–2 | same |
| `modeling/BLINK/scripts/blink_reranker_fine_tuning.sh` | 1–2 | same |
| `modeling/ReS/res_reranker_fine_tuning.sh` | 1–2 | same |

A quick way to find them all again:

```bash
grep -rn "/lustre/\|OLLAMA_MODELS\|cache_dir" --include='*.sh' --include='*.py' \
     install.sh data_preparation modeling | grep -v __pycache__
```

`data_preparation/run_ollama_to_serve_llm` also fixes the Ollama port at **11435**
(`OLLAMA_HOST=127.0.0.1:11435`). `alias_generation.py` hard-codes the same port in
`LLMSelector.get_ollama` (`base_url = "http://127.0.0.1:11435"`), so change both together if the
port is taken.
