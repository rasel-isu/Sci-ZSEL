# LLM cost accounting

> Part of the [Sci-ZSEL](../README.md) documentation.

Cost control is the point of the entity-selection stage, and the pipeline measures it for you.
`alias_generation.py` writes a `*_token_counts.txt` next to each entity set with per-entity and
aggregate counts. For NCBI-Disease / MEDIC:

| Entity set | Entities prompted | Input tokens | Output tokens | Wall clock |
|---|---|---|---|---|
| E_EM (`m1_e1_unq_ents`) | 169 | 64,672 | 7,273 | 00:19:12 |
| E_BT (`top_1_from_biencoder`) | 633 | 236,546 | 26,504 | 01:13:36 |
| **Total** | **802** | **301,218** | **33,777** | **~1.5 h** |

802 prompted entities out of 13,316 in MEDIC — about **6%** of the ontology. Wall clock is for
`llama3.2:3b-instruct-fp16` served locally by Ollama and will vary with your hardware.
