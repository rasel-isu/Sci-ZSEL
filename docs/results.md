# Settings and where their results land

> Part of the [Sci-ZSEL](../README.md) documentation.

Four training settings ship, and each is just a directory name that flows through every stage of the
pipeline. Pick one by putting its name in `exp_list`; every result path below is that same name
again. [Experiment names](experiment-names.md) explains what the name components mean.

### The four settings

| Paper setting | `exp_list` entry |
|---|---|
| **Sci-ZSEL** | `(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e` |
| **Sci-ZSEL w/o filter** | `(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime` |
| **Sci-ZSEL + Synonym** | `synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e` |
| **Synonym** (baseline) | `synonym` |

Watch the leading `(` — the names begin `(m1_e1)`, not `m1_e1)`.

Step 5.2 builds all four under `datasets/<world>/blink_format/train/` in one pass, so you choose
between them at training time, not at data-prep time. Three keys select it, and they are
independent — set all three to the same value unless you deliberately want to mix:

```jsonc
"blink": {
  "retriever": { "exp_list": ["<setting>"] },   // step 5.3
  "reranker":  { "exp_list": ["<setting>"] }    // step 5.4
},
"res":        { "exp_list": ["<setting>"] }     // step 5.5
```

`exp_list` is an array: list several and each is trained in turn, one output tree per entry.

### Where the results land

Writing `<exp>` for the chosen setting, `<world>` for the corpus and `<s>` for the seed:

| Stage | File |
|---|---|
| Retriever | `saved_models/<world>/biencoder/train/<exp>/epoch_<i>/top64_candidates/test_eval.txt` |
| BLINK reranker | `saved_models/<world>/crossencoder/train/fine-tune/seed-<s>/<exp>/epoch_<i>/crossencoder_predictions_eval.txt` |
| ReS reranker | `saved_models/<world>/res/<world>/train/seed-<s>/<exp>/<world>_<exp>/epoch_<i>/pred_eval.txt` |

Note the ReS path repeats `<world>` twice and then again as the `<world>_<exp>` prefix; that is the
real layout, not a typo.

**Which `epoch_<i>` is the final one.** The retriever and the BLINK reranker count from `epoch_0`,
ReS counts from `epoch_1`. With the shipped defaults (retriever 1 epoch, reranker 3, ReS 3) the last
directory of each is:

| Stage | Epochs | Directories | Final |
|---|---|---|---|
| Retriever | 1 | `epoch_0` | `epoch_0` |
| BLINK reranker | 3 | `epoch_0` … `epoch_2` | `epoch_2` |
| ReS reranker | 3 | `epoch_1` … `epoch_3` | `epoch_3` |

Every epoch is scored, so pick the epoch you mean to report rather than assuming the highest number
is best.

### Worked example — LPT, Sci-ZSEL + Synonym, seed 0

```
saved_models/lpt/biencoder/train/synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e/epoch_0/top64_candidates/test_eval.txt

saved_models/lpt/crossencoder/train/fine-tune/seed-0/synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e/epoch_2/crossencoder_predictions_eval.txt

saved_models/lpt/res/lpt/train/seed-0/synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e/lpt_synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e/epoch_3/pred_eval.txt
```

Swap the `<exp>` segments for another row of the first table to get the other three settings, and
`lpt` for another corpus.

### Two files that are not a setting's result

- `saved_models/<world>/biencoder/train/original_data/top64_candidates/test_eval.txt` has no `<exp>`
  and no `epoch_` directory. It is the **off-the-shelf** bi-encoder retrieving over the ontology with
  no fine-tuning at all — the zero-shot floor every setting is measured against.
- Inside `crossencoder_predictions_eval.txt`, the `Bi-Encoder` block is *also* that non-fine-tuned
  retriever, reported because it produced the candidates the reranker consumed. The setting's actual
  reranker result is the `Cross-Encoder` block below it.

For an unlabeled split there is nothing to score, so the matching `*_eval.txt` holds a
`... evaluation skipped.` note instead of metrics; see
[`has_ground_truth`](../README.md#3-configjson-reference).
