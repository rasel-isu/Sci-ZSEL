# Troubleshooting and known quirks

> Part of the [Sci-ZSEL](../README.md) documentation.

**Run every script from its own directory.** `config.json` is opened as `../config.json` from
`data_preparation/` and `../../config.json` from `modeling/BLINK/` and `modeling/ReS/`. Running
`bash data_preparation/grag_to_blink.sh` from the repo root fails with `FileNotFoundError`.

**The two conda envs are not interchangeable.** BLINK code needs `transformers==4.31.0`, ReS needs
`4.30.2`. Each `.sh` activates the right one; if you invoke a `.py` by hand, activate it yourself.

**`--has_gt` cannot be switched off by passing `false`.** `params.py:383` declares it as
`type=bool`, and `bool("false")` is `True` — so `--has_gt false`, `--has_gt False` and
`--has_gt true` all mean *true*; only omitting the flag gives false. `scripts/load_config.sh`
therefore emits the entire flag or nothing (`CG_HAS_GT_FLAG`), driven by
`blink.candidate_generation.has_gt` (which defaults to the top-level `has_ground_truth`). Watch for
this if you add other `type=bool` flags.

**`has_gt` is per-corpus, but "does this split have labels" is per-split.** Setting
`has_ground_truth: false` for a QTL world is right for `get_biencoder_top_k.sh`, which reads the
unlabeled raw train corpus. It is wrong for reranker candidate generation, whose inputs are the
pseudo-labeled training set and the annotated test set. With the flag off there,
`nn_prediction.py` skips the connected-candidate (ontology parent/child) block, leaving
`connected_candidates_graph` empty while `context_vecs` is full, and `train_cross.py` dies with
`IndexError: list index out of range` in `modify_list`. That is why
`get_biencoder_cands_for_reranker_training.sh` passes `--has_gt true` unconditionally.

**Ollama port 11435 is hard-coded in two places** — `run_ollama_to_serve_llm` (`OLLAMA_HOST`) and
`alias_generation.py::LLMSelector.get_ollama` (`base_url`). `alias_generation.sh` polls
`http://127.0.0.1:11435/api/tags` until the server answers and kills the whole process group on
exit; if a previous run left a server on that port, the trap in the new run will kill it.

**`get_retriever_candidates.py` is commented out** in `res_reranker_fine_tuning.sh`. ReS training
will fail on a missing `res_format/.../train.json` unless you uncomment it for the first run (see [Running the pipeline](../README.md#4-running-the-pipeline)).

**ReS sets `CUDA_VISIBLE_DEVICES` from `res.gpus`**, which ships as `"0,1,2,3"`. Change it in
`config.json` to match your machine (e.g. `"0"` for a single GPU).

**Weights & Biases** is used by the cross-encoder trainer (`train_cross.py`) and by ReS
(`run_disambiguation_attention.py`); both reranker scripts already set
`export WANDB_MODE=disabled`. Drop that line and run `wandb login` if you want the runs logged. The
retriever trainer does not use wandb at all.

**`log.txt` collisions.** `utils.Logger` appends `+` to the filename when a log already exists,
which is why you may see `train.log`, `train.log+`, `train.log++` in a ReS output directory. The
latest run is the one with the most `+`.

**Duplicate detection is strict.** `utils.check_duplicate` raises `ValueError('Some samples are
duplicated!')` if a generated `train.jsonl` contains two identical samples (ignoring `sample_id`).
If you hit this after modifying the construction code, that is the assertion firing.
