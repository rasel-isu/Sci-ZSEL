eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel

# Corpus, ontology, experiment list and every hyperparameter come from config.json
# -> "blink": { "retriever": { ... } }.
#
# Valid "exp_list" entries:
#   "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime"                          Sci-ZSEL w/o filter
#   "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"          Sci-ZSEL
#   "synonym"                                                                  synonym baseline
#   "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"  Sci-ZSEL + Synonym
#
# Valid "negative_selection" entries:
#   "add_prch_in_pos_list"  "remove_prchsbl_from_neg_list"  "BASE-NEG-ENTIRE-KB"

ROOT="../.."
source scripts/load_config.sh

TIME_START=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="logs/bienc/$(date '+%Y-%m-%d-%H-%M-%S').log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "============================================"
echo "Job started at: $START_TIMESTAMP"
echo "CONFIG:    $CONFIG_FILE"
echo "WORLD:     $WORLD  (ontology: $KB_NAME)"
echo "SEEDS:     ${RT_SEEDS[*]}"
echo "VARIANTS:  ${RT_VARIANT[*]}"
echo "EXP_LIST:  ${RT_EXP_LIST[*]}"
echo "============================================"

export PYTHONPATH=.

for SEED in "${RT_SEEDS[@]}"; do
for VARI in "${RT_VARIANT[@]}"; do
  for EXP in "${RT_EXP_LIST[@]}"; do
      DATA_PATH="$DATA_DIR/blink_format/$SPLITNAME/$EXP"
      TEST_DATA_PATH="$DATA_DIR/blink_format/$SPLITNAME/original_data"
      ENTITY_DICT="$DATA_PATH/kb.jsonl"
      OUTPUT_DIR="$SAVED_MODEL_DIR/biencoder/$SPLITNAME/$EXP"
      echo "Using encoding file: $CANDIDATE_ENCODINGS"

      python blink/biencoder/train_biencoder.py \
        --blink_base_model_path $BIENCODER_BASE_MODEL \
        --entity_dict_path $ENTITY_DICT \
        --cand_encode_path $CANDIDATE_ENCODINGS \
        --cand_pool_path $CANDIDATE_POOL_PATH \
        --data_path $DATA_PATH \
        --test_data_path $TEST_DATA_PATH \
        --output_path $OUTPUT_DIR \
        --grag_data_path $GRAG_DATA_PATH \
        --kb_file_path $KB_FILE_PATH \
        --onto $WORLD \
        --experiment $EXP \
        --seed  $SEED \
        --encode_batch_size $RT_ENCODE_BATCH_SIZE --eval_batch_size $RT_TRAIN_BATCH_SIZE \
        --top_k $RT_TOP_K \
        --save_topk_result \
        --bert_model $RT_BERT_MODEL \
        --type_optimization $RT_TYPE_OPTIMIZATION \
        --data_parallel \
        --learning_rate  $RT_LEARNING_RATE \
        --dropout_rate $RT_DROPOUT_RATE \
        --num_train_epochs $RT_EPOCHS \
        --train_batch_size $RT_TRAIN_BATCH_SIZE \
        --max_context_length $RT_MAX_CONTEXT_LENGTH \
        --max_cand_length $RT_MAX_CAND_LENGTH \
        --max_seq_length $RT_MAX_SEQ_LENGTH \
        --bi_enc_negative_selection $VARI
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
echo "  WORLD:     $WORLD"
echo "  SEEDS:     ${RT_SEEDS[*]}"
echo "  VARIANTS:  ${RT_VARIANT[*]}"
echo "  EXP_LIST:  ${RT_EXP_LIST[*]}"
echo "============================================"
