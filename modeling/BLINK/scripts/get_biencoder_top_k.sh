eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel

ROOT="../.."
MODEL_PATH="$ROOT/saved_models/biencoder_wiki_large.bin"
EXP="original_data"
SPLITNAME="train"
ONTO="ncbi_disease"
ENTITY_DICT="$ROOT/datasets/$ONTO/blink_format/$SPLITNAME/$EXP/kb.jsonl"
CANDIDATE_ENCODINGS="$ROOT/saved_models/$ONTO/medic_entity_encodings.t7"
KB_FILE_PATH="$ROOT/datasets/$ONTO/medic.json"
CANDIDATE_POOL_PATH="$ROOT/saved_models/$ONTO/medic_entity_pool.t7"
GRAG_DATA_PATH="$ROOT/datasets/$ONTO/"
DATA_PATH="$ROOT/datasets/$ONTO/blink_format/$SPLITNAME/$EXP"
OUTPUT_DIR="$ROOT/saved_models/$ONTO/biencoder/$SPLITNAME/$EXP"

export PYTHONPATH=.
python blink/biencoder/eval_biencoder.py \
--path_to_model $MODEL_PATH \
--entity_dict_path $ENTITY_DICT \
--cand_encode_path $CANDIDATE_ENCODINGS \
--cand_pool_path $CANDIDATE_POOL_PATH \
--data_path $DATA_PATH \
--output_path $OUTPUT_DIR \
--grag_data_path $GRAG_DATA_PATH \
--kb_file_path $KB_FILE_PATH \
--onto $ONTO \
--experiment $EXP \
--max_context_length 64 \
--encode_batch_size 8 --eval_batch_size 32 \
--top_k 64 \
--save_topk_result \
--bert_model bert-large-uncased --mode train \
--data_parallel \
--has_gt true \

