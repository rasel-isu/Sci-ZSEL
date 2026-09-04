# Decoding the experiment names

> Part of the [Sci-ZSEL](../README.md) documentation.

The directory names encode how a training set was built. `U` means set union.

| Component | Paper name | Meaning |
|---|---|---|
| `original_data` | — | Gold-labeled conversion of the raw corpus. `test.jsonl` is the evaluation set; `train.jsonl` is used only as the retrieval mention pool. **Never a training set.** |
| `(m1_e1)` | E_EM pairs | Mention string exactly equals an ontology entity name. No LLM involved. |
| `(m3_e1)` | E_EM + aliases | Mention matches an LLM-generated alias of an **exact-match-selected** entity. |
| `(m4_e2)` | E_BT + aliases | Mention matches an LLM-generated alias of a **bi-encoder top-1** entity. |
| `_multi_prime` | — | The LLM returns a comma-separated list; every element counts as an alias. |
| `_rm_sm_e` | ontology-aware filter | "remove smaller entity": drop aliases whose similarity to the entity name is below any ontology neighbor's, or below 0.9. |
| `synonym` | Synonym baseline | Pairs built from the ontology's own curated synonym list. No LLM involved. |

So the two headline configurations are:

- **Sci-ZSEL** = `(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e`
- **Sci-ZSEL + Synonym** = `synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e`

and the filter ablation is `(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime` (same sets, no
`_rm_sm_e`).

Step 5.2 builds all of them in one pass. Sanity-check your run against the NCBI-Disease pair counts
(`wc -l` on each `blink_format/train/<exp>/train.jsonl`):

| Experiment | Pairs |
|---|---|
| `(m1_e1)` | 854 |
| `(m3_e1)_multi_prime` → `_rm_sm_e` | 556 → 360 |
| `(m4_e2)_multi_prime` → `_rm_sm_e` | 1,102 → 740 |
| `(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime` (w/o filter) | 1,736 |
| `(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e` (**Sci-ZSEL**) | 1,299 |
| `synonym` | 3,283 |
| `synonymU(...)` (**Sci-ZSEL + Synonym**) | 4,582 |

Each directory also gets `cat_wise_acc.json` (pseudo-label precision per overlap category, when
ground truth exists) and `train_category_count.json`.

### Lexical-overlap categories

Every evaluation file breaks results down by how much the mention and the gold entity name overlap
(`data_preparation/utils.py::get_category`, mirrored in `modeling/BLINK/utils.py:451` and
`modeling/ReS/utils.py:140`):

| Category | Condition |
|---|---|
| **HO** | Mention equals the entity name, or differs only by a trailing plural `s` |
| **MINT** | Mention is a proper substring of the entity name |
| **LO** | Mention and entity name share at least one token, but neither of the above |
| **NO** | No shared token — the case the paper targets |

The NCBI-Disease test set splits 277 / 29–31 / 432–435 / 219–220 across HO / MINT / LO / NO.
