eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel


ROOT="../.."
BASE_MODEL_PATH="$ROOT/saved_models/biencoder_wiki_large.bin"
SPLITNAME="train"

EXP_LIST=(
    # "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime"
    # "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"
    # "synonym"
    "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"
)
ONTO_LIST=("ncbi_disease")
VARIANT=(
  # "BASE-NEG-ENTIRE-KB"
  # "remove_prchsbl_from_neg_list"
  "add_prch_in_pos_list"
)
SEEDS=(0)
TRAIN_BATCH_SIZE=512
EPOCH=1
TIME_START=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="logs/bienc/$(date '+%Y-%m-%d-%H-%M-%S').log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "============================================"
echo "Job started at: $START_TIMESTAMP"
echo "SEEDS:     ${SEEDS[*]}"
echo "VARIANTS:  ${VARIANT[*]}"
echo "ONTO_LIST: ${ONTO_LIST[*]}"
echo "EXP_LIST:  ${EXP_LIST[*]}"
echo "============================================"

export PYTHONPATH=.

for SEED in "${SEEDS[@]}"; do
for VARI in "${VARIANT[@]}"; do
  for ONTO in "${ONTO_LIST[@]}"; do
    for EXP in "${EXP_LIST[@]}"; do
      ENTITY_DICT="$ROOT/datasets/$ONTO/blink_format/$SPLITNAME/$EXP/kb.jsonl"
      if [ "$ONTO" = "ncbi_disease" ]; then
        CANDIDATE_ENCODINGS="$ROOT/saved_models/$ONTO/medic_entity_encodings.t7"
        KB_FILE_PATH="$ROOT/datasets/$ONTO/medic.json"
        CANDIDATE_POOL_PATH="$ROOT/saved_models/$ONTO/medic_entity_pool.t7"
        GRAG_DATA_PATH="$ROOT/datasets/$ONTO/"
      else
        echo "Unknown ontology: $ONTO"
        exit 1
      fi
        echo "Using encoding file: $CANDIDATE_ENCODINGS"

      DATA_PATH="$ROOT/datasets/$ONTO/blink_format/$SPLITNAME/$EXP"
      TEST_DATA_PATH="$ROOT/datasets/$ONTO/blink_format/$SPLITNAME/original_data"
      OUTPUT_DIR="$ROOT/saved_models/$ONTO/biencoder/$SPLITNAME/$EXP"

      python blink/biencoder/train_biencoder.py \
        --blink_base_model_path $BASE_MODEL_PATH \
        --entity_dict_path $ENTITY_DICT \
        --cand_encode_path $CANDIDATE_ENCODINGS \
        --cand_pool_path $CANDIDATE_POOL_PATH \
        --data_path $DATA_PATH \
        --test_data_path $TEST_DATA_PATH \
        --output_path $OUTPUT_DIR \
        --grag_data_path $GRAG_DATA_PATH \
        --kb_file_path $KB_FILE_PATH \
        --onto $ONTO \
        --experiment $EXP \
        --seed  $SEED \
        --encode_batch_size 8 --eval_batch_size $TRAIN_BATCH_SIZE \
        --top_k 64 \
        --save_topk_result \
        --bert_model bert-large-uncased \
        --type_optimization all_encoder_layers \
        --data_parallel \
        --learning_rate  2e-05 \
        --dropout_rate 0.2 \
        --num_train_epochs $EPOCH \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --max_context_length 128 \
        --max_cand_length 128 \
        --max_seq_length 192 \
        --bi_enc_negative_selection $VARI 
    done
  done
done
done


TIME_END=$(date +%s)
END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED_SEC=$(( TIME_END - TIME_START ))
ELAPSED_HRS=$(printf "%02dh %02dm %02ds" \
  $(( ELAPSED_SEC / 3600 )) \
  $(( (ELAPSED_SEC % 3600) / 60 )) \
  $(( ELAPSED_SEC % 60 )))

echo "============================================"
echo "Job finished at:  $END_TIMESTAMP"
echo "Started at:       $START_TIMESTAMP"
echo "Elapsed time:     ${ELAPSED_HRS} hrs  (${ELAPSED_SEC}s)"
echo "--------------------------------------------"
echo "Run configuration:"
echo "  SEEDS:     ${SEEDS[*]}"
echo "  VARIANTS:  ${VARIANT[*]}"
echo "  ONTO_LIST: ${ONTO_LIST[*]}"
echo "  EXP_LIST:  ${EXP_LIST[*]}"
echo "============================================"
