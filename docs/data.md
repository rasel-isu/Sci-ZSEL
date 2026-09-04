# Data

> Part of the [Sci-ZSEL](../README.md) documentation.

### What ships

| Directory | Corpus | Ontology | Train mentions | Test mentions | Ontology entities | Train GT? |
|---|---|---|---|---|---|---|
| `datasets/ncbi_disease/` | NCBI-Disease | MEDIC (`medic.json`) | 5,145 | 960 | 13,316 | yes |
| `datasets/bc5cdr/` | BC5CDR | MeSH (`mesh.json`) | 9,285 | 9,654 | 355,213 | yes |
| `datasets/qtl_cmo/` | Animal science (QTL) | CMO (`cmo_kb.json`) | 16,385 | 2,032 | 4,133 | **no** |
| `datasets/qtl_vt/` | Animal science (QTL) | VT (`vt_kb.json`) | 16,385 | 1,688 | 4,044 | **no** |
| `datasets/qtl_lpt/` | Animal science (QTL) | LPT (`lpt_kb.json`) | 16,385 | 722 | 520 | **no** |

The three `qtl_*` directories are the **new animal science benchmark** released with the paper. They
share one unlabeled 16,385-mention training corpus (`ground_truth: []` — this is the
zero-human-annotation setting the method targets) and have one annotated test set per livestock
trait ontology. Set `has_ground_truth: false` in `config.json` for these (see [`config.json` reference](../README.md#3-configjson-reference)).

Prompts also ship for `cometa`, but the COMETA corpus itself is not redistributed here.

### Raw corpus format (`*_grag.json`)

A JSON list. The mention is delimited in-place by `[MENTION_START]` / `[MENTION_END]`:

```json
{
  "sample_id": 1760124371,
  "mention": "skin tumour",
  "mention_context": "A common human [MENTION_START] skin tumour [MENTION_END] is caused by ...",
  "ground_truth": [{"id": "D012878", "title": "Skin Neoplasms"}]
}
```

`ground_truth` is a list (a mention may have several acceptable ontology ids) and is `[]` for
unlabeled training corpora.

### Ontology format

A JSON dict keyed by entity id:

```json
{
  "C538288": {
    "id": "C538288",
    "name": "10p Deletion Syndrome (Partial)",
    "def": "",
    "synonyms": ["Chromosome 10, monosomy 10p", "..."],
    "altdiseaseid": [],
    "ParentIDs": ["MESH:D002872", "MESH:D025063"]
  }
}
```

- `name` / `def` become the entity title and description given to the encoders.
- `synonyms` is used only for the curated-synonym baseline (key name is configurable).
- `altdiseaseid` holds alternate ids that count as correct at evaluation time.
- `ParentIDs` drives both the ontology-aware alias filter and the parent/child-aware negative
  sampling. **An entity with no `def` is dropped from the LLM alias-generation pool.**

`datasets/bc5cdr/mesh.json` is still in raw MeSH form (`synonym`, `parent_of`, `is_a` as a string),
so it needs `synonym_key_on_ontology: "synonym"` and `has_ent_alt_id: false`.
`data_preparation/utils.py::convert_kb` will **not** convert it: that helper is the OBO→KB converter
used to build the animal-science ontologies (its output shape matches `cmo_kb.json` exactly), and it
expects a *list* of OBO-exported entities with OBO-quoted synonyms (`"foo" EXACT []`). On the
dict-keyed `mesh.json` it raises `TypeError`, then `IndexError` in its synonym regex.

### Generated intermediates

| Path | Written by | Contents |
|---|---|---|
| `datasets/<world>/blink_format/train/<exp>/train.jsonl` | data preparation | BLINK training pairs |
| `datasets/<world>/blink_format/train/original_data/test.jsonl` | data preparation | the gold test set |
| `.../<exp>/{kb.jsonl,entity.jsonl,id_map.json}` | data preparation | ontology as integer-indexed candidates + int↔ontology-id map |
| `datasets/<world>/res_format/train/<exp>/train.json` | `get_retriever_candidates.py` | ReS training pairs with top-64 candidates |
| `datasets/<world>/res_format/train/original_data/test.json` | `get_retriever_candidates.py` | ReS test set |

`id_map.json` maps the contiguous integer ids used inside BLINK back to ontology ids; nearly every
evaluation path goes through it, so never mix an `id_map.json` from one experiment with candidates
from another.
