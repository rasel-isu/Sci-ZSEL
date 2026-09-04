eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel

# Usage: bash scripts/get_biencoder_cands_for_reranker_training.sh <experiment> <mode>
#   <experiment>  a directory under datasets/<world>/blink_format/<split>/
#   <mode>        train | test  (which *.jsonl to retrieve candidates for)
# The corpus, ontology and all paths come from config.json.

ROOT="../.."
source scripts/load_config.sh

EXP="${1:?Usage: $0 <experiment_name> <mode>}"
MODE="${2:?Usage: $0 <experiment_name> <mode>}"

DATA_PATH="$DATA_DIR/blink_format/$SPLITNAME/$EXP"
ENTITY_DICT="$DATA_PATH/kb.jsonl"
OUTPUT_DIR="$SAVED_MODEL_DIR/biencoder/$SPLITNAME/$EXP"

export PYTHONPATH=.
python blink/biencoder/eval_biencoder.py \
--path_to_model $BIENCODER_BASE_MODEL \
--entity_dict_path $ENTITY_DICT \
--cand_encode_path $CANDIDATE_ENCODINGS \
--cand_pool_path $CANDIDATE_POOL_PATH \
--data_path $DATA_PATH \
--output_path $OUTPUT_DIR \
--grag_data_path $GRAG_DATA_PATH \
--kb_file_path $KB_FILE_PATH \
--onto $WORLD \
--experiment $EXP \
--max_context_length $CG_MAX_CONTEXT_LENGTH \
--encode_batch_size $CG_ENCODE_BATCH_SIZE --eval_batch_size $CG_EVAL_BATCH_SIZE \
--top_k $CG_TOP_K \
--save_topk_result \
--bert_model $CG_BERT_MODEL --mode $MODE \
--data_parallel \
$CG_HAS_GT_FLAG
