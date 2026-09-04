eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel

ROOT="../.."
source scripts/load_config.sh   # -> WORLD, DATA_DIR, KB_FILE_PATH, CG_*, ... from config.json

EXP="original_data"
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
--bert_model $CG_BERT_MODEL --mode $SPLITNAME \
--data_parallel \
--has_gt $CG_HAS_GT \
