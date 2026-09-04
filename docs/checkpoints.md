# Pretrained checkpoints to download

> Part of the [Sci-ZSEL](../README.md) documentation.

Three checkpoints go into `saved_models/` at the repo root.

**1–2. BLINK retriever + cross-encoder** — from
[facebookresearch/BLINK](https://github.com/facebookresearch/BLINK). Run from the repo root:

```bash
mkdir -p saved_models
wget -c -P saved_models http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.bin    # 2.7 GB
wget -c -P saved_models http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.bin # 1.3 GB
```

**3. ReS zero-shot checkpoint** — `zeshel_disambiguation_attention.pt` (1.5 GB), the ZESHEL-trained
model released by [HITsz-TMG/Read-and-Select](https://github.com/HITsz-TMG/Read-and-Select)
(*A Read-and-Select Framework for Zero-shot Entity Linking*, Findings of EMNLP 2023).

It is hosted on OneDrive, which refuses anonymous non-browser requests (`403`), so `wget`/`curl`
will not work — **download it in a browser** from
<https://1drv.ms/u/s!AoTJ9uWa69GGf3Rr-zCtO14Nvyo?e=CFWExB> (the link in that repo's
`model_disambiguation/README.md`), then move it into place:

```bash
mv ~/Downloads/zeshel_disambiguation_attention.pt saved_models/
```

If you are on a headless cluster, download it on your laptop and copy it over:

```bash
scp zeshel_disambiguation_attention.pt <user>@<host>:<path-to-repo>/saved_models/
```

Only ReS needs this file; steps 5.1–5.4 (BLINK) run without it. The path is configurable via
`config.json → res.pretrained_model`.

**Verify.** All three should be present with these sizes:

```bash
ls -l saved_models/*.bin saved_models/*.pt
# 2681357077  biencoder_wiki_large.bin
# 1340677176  crossencoder_wiki_large.bin
# 1522192979  zeshel_disambiguation_attention.pt
```

Everything else under `saved_models/` is produced by the pipeline, including the cached ontology
encodings at `<saved_model_dir>/<kb_name>_entity_pool.t7` and `<kb_name>_entity_encodings.t7` — for
the shipped config, `saved_models/ncbi_disease/medic_entity_*.t7`. They are built on the first
bi-encoder run and reused blindly afterwards, so **delete them whenever the ontology changes**.
