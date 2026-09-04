# Loads ../../config.json into shell variables.
#
# Usage, from modeling/BLINK/ :
#     ROOT="../.."
#     source scripts/load_config.sh
#
# Every path it exports is already prefixed with $ROOT, so the callers never
# spell out an ontology name or a dataset directory.
#
# Paths inside config.json are written relative to data_preparation/ (that is
# how the Python code reads them), i.e. "../datasets/foo/" means "<repo>/datasets/foo/".
# The leading "../" is stripped here and replaced by $ROOT.
#
# Override the config file with:  CONFIG_FILE=/path/to/other.json source scripts/load_config.sh

ROOT="${ROOT:-../..}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/config.json}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "load_config.sh: cannot find $CONFIG_FILE — run this script from modeling/BLINK/" >&2
    return 1 2>/dev/null || exit 1
fi

# Emit `NAME=value` / `NAME=(a b c)` assignments and eval them. Values are
# shell-quoted by python, so spaces and parentheses in experiment names are safe.
__cfg_assignments=$(ROOT="$ROOT" python3 - "$CONFIG_FILE" <<'PY'
import json, os, shlex, sys

with open(sys.argv[1]) as f:
    cfg = json.load(f)

root = os.environ.get("ROOT", "../..").rstrip("/")
blink = cfg.get("blink", {})
missing = []


def q(v):
    return shlex.quote(str(v))


def emit(name, value):
    print(f"{name}={q(value)}")


def emit_arr(name, values):
    print(f"{name}=({' '.join(q(v) for v in values)})")


def rooted(rel):
    """'../datasets/foo/' (relative to data_preparation/) -> '<ROOT>/datasets/foo'."""
    rel = str(rel).strip().rstrip("/")
    while rel.startswith("../"):
        rel = rel[3:]
    return f"{root}/{rel}"


def get(section, key, default=None, required=False):
    if key in section:
        return section[key]
    if required:
        missing.append(key)
        return ""
    return default


# ---- corpus / ontology -------------------------------------------------
world = get(cfg, "world", required=True)
kb_name = get(cfg, "kb_name", required=True)
kb_file = get(cfg, "kb_file", required=True)
data_dir = rooted(get(cfg, "data_dir", required=True))
saved_model_dir = rooted(get(cfg, "saved_model_dir", required=True))

emit("WORLD", world)
emit("KB_NAME", kb_name)
emit("DATA_DIR", data_dir)
emit("SAVED_MODEL_DIR", saved_model_dir)
emit("KB_FILE_PATH", f"{data_dir}/{kb_file}")
emit("GRAG_DATA_PATH", data_dir)
emit("CANDIDATE_ENCODINGS", f"{saved_model_dir}/{kb_name}_entity_encodings.t7")
emit("CANDIDATE_POOL_PATH", f"{saved_model_dir}/{kb_name}_entity_pool.t7")
emit("SPLITNAME", get(blink, "split_name", "train"))

# ---- pretrained BLINK checkpoints --------------------------------------
base = blink.get("base_models", {})
emit("BIENCODER_BASE_MODEL",
     rooted(get(base, "biencoder", "../saved_models/biencoder_wiki_large.bin")))
emit("CROSSENCODER_BASE_MODEL",
     rooted(get(base, "crossencoder", "../saved_models/crossencoder_wiki_large.bin")))

# ---- candidate generation (eval_biencoder.py) --------------------------
cg = blink.get("candidate_generation", {})
emit("CG_TOP_K", get(cg, "top_k", 64))
emit("CG_MAX_CONTEXT_LENGTH", get(cg, "max_context_length", 64))
emit("CG_ENCODE_BATCH_SIZE", get(cg, "encode_batch_size", 8))
emit("CG_EVAL_BATCH_SIZE", get(cg, "eval_batch_size", 32))
emit("CG_BERT_MODEL", get(cg, "bert_model", "bert-large-uncased"))
emit("CG_HAS_GT", str(get(cg, "has_gt", cfg.get("has_ground_truth", True))).lower())

# ---- retriever fine-tuning (train_biencoder.py) ------------------------
rt = blink.get("retriever", {})
emit_arr("RT_EXP_LIST", get(rt, "exp_list", [], required=True) or [])
emit_arr("RT_SEEDS", get(rt, "seeds", [0]))
emit_arr("RT_VARIANT", get(rt, "negative_selection", ["add_prch_in_pos_list"]))
emit("RT_EPOCHS", get(rt, "epochs", 1))
emit("RT_TRAIN_BATCH_SIZE", get(rt, "train_batch_size", 64))
emit("RT_ENCODE_BATCH_SIZE", get(rt, "encode_batch_size", 8))
emit("RT_LEARNING_RATE", get(rt, "learning_rate", "2e-05"))
emit("RT_DROPOUT_RATE", get(rt, "dropout_rate", 0.2))
emit("RT_MAX_CONTEXT_LENGTH", get(rt, "max_context_length", 128))
emit("RT_MAX_CAND_LENGTH", get(rt, "max_cand_length", 128))
emit("RT_MAX_SEQ_LENGTH", get(rt, "max_seq_length", 192))
emit("RT_TOP_K", get(rt, "top_k", 64))
emit("RT_BERT_MODEL", get(rt, "bert_model", "bert-large-uncased"))
emit("RT_TYPE_OPTIMIZATION", get(rt, "type_optimization", "all_encoder_layers"))

# ---- reranker fine-tuning (train_cross.py) -----------------------------
rr = blink.get("reranker", {})
emit_arr("RR_EXP_LIST", get(rr, "exp_list", [], required=True) or [])
emit_arr("RR_SEEDS", get(rr, "seeds", [0]))
emit("RR_EPOCHS", get(rr, "epochs", 3))
emit("RR_TRAIN_BATCH_SIZE", get(rr, "train_batch_size", 16))
emit("RR_GRAD_ACC_STEPS", get(rr, "gradient_accumulation_steps", 2))
emit("RR_EVAL_BATCH_SIZE", get(rr, "eval_batch_size", 32))
emit("RR_LEARNING_RATE", get(rr, "learning_rate", "2e-05"))
emit("RR_DROPOUT_RATE", get(rr, "dropout_rate", 0.2))
emit("RR_MAX_CONTEXT_LENGTH", get(rr, "max_context_length", 64))
emit("RR_MAX_CAND_LENGTH", get(rr, "max_cand_length", 128))
emit("RR_MAX_SEQ_LENGTH", get(rr, "max_seq_length", 192))
emit("RR_NEGATIVE_SELECTION", get(rr, "negative_selection", "only_bienc_20_neg"))
emit("RR_BERT_MODEL", get(rr, "bert_model", "bert-large-uncased"))
emit("RR_TYPE_OPTIMIZATION", get(rr, "type_optimization", "all_encoder_layers"))

if missing:
    sys.stderr.write(
        "load_config.sh: missing required key(s) in %s: %s\n" % (sys.argv[1], ", ".join(missing))
    )
    sys.exit(1)
PY
) || { echo "load_config.sh: failed to parse $CONFIG_FILE" >&2; return 1 2>/dev/null || exit 1; }

eval "$__cfg_assignments"
unset __cfg_assignments

# Derived: the directory name train_biencoder.py/eval_biencoder.py write candidates into.
TOPK_DIR="top${CG_TOP_K}_candidates"
