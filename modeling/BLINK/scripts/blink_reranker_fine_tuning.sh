eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel

export WANDB_MODE=disabled

# Corpus, ontology, experiment list and every hyperparameter come from config.json
# -> "blink": { "reranker": { ... } }.  See retriever_fine_tuning.sh for the valid
# "exp_list" values; set both stages to the same experiment unless you mean otherwise.

ROOT="../.."
source scripts/load_config.sh

TIME_START=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="logs/cross/$(date '+%Y-%m-%d-%H-%M-%S').log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "============================================"
echo "Job started at: $START_TIMESTAMP"
echo "CONFIG:    $CONFIG_FILE"
echo "WORLD:     $WORLD  (ontology: $KB_NAME)"
echo "SEEDS:     ${RR_SEEDS[*]}"
echo "EXP_LIST:  ${RR_EXP_LIST[*]}"
echo "============================================"

echo "Getting candidates from retriever for testset ...."
bash scripts/get_biencoder_cands_for_reranker_training.sh original_data test
echo "Done! got candidates from retriever for testset!"

for EXP in "${RR_EXP_LIST[@]}"; do
  echo "Getting candidates from retriever for $EXP ...."
  bash scripts/get_biencoder_cands_for_reranker_training.sh "$EXP" train
  echo "Done! got candidates from retriever for $EXP!"

for SEED in "${RR_SEEDS[@]}"; do
  echo "KB: $KB_FILE_PATH"

  RAW_DATA_PATH="$DATA_DIR/blink_format/$SPLITNAME/$EXP/"
  DATA_PATH="$SAVED_MODEL_DIR/biencoder/$SPLITNAME/$EXP/$TOPK_DIR"
  TEST_DATA_PATH="$SAVED_MODEL_DIR/biencoder/$SPLITNAME/original_data/$TOPK_DIR"
  OUTPUT_DIR="$SAVED_MODEL_DIR/crossencoder/$SPLITNAME/fine-tune/seed-$SEED/$EXP"

  export PYTHONPATH=.
  python blink/crossencoder/train_cross.py \
    --data_path  $DATA_PATH \
    --raw_data_path $RAW_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --grag_data_path $GRAG_DATA_PATH \
    --kb_file_path $KB_FILE_PATH \
    --cand_pool_path $CANDIDATE_POOL_PATH \
    --output_path $OUTPUT_DIR \
    --onto $WORLD \
    --experiment $EXP \
    --seed  $SEED \
    --blink_model_path $CROSSENCODER_BASE_MODEL \
    --learning_rate $RR_LEARNING_RATE \
    --dropout_rate $RR_DROPOUT_RATE \
    --num_train_epochs $RR_EPOCHS \
    --max_context_length $RR_MAX_CONTEXT_LENGTH \
    --max_cand_length $RR_MAX_CAND_LENGTH \
    --max_seq_length $RR_MAX_SEQ_LENGTH \
    --train_batch_size $RR_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $RR_GRAD_ACC_STEPS \
    --eval_batch_size $RR_EVAL_BATCH_SIZE \
    --cross_enc_negative_selection $RR_NEGATIVE_SELECTION \
    --bert_model $RR_BERT_MODEL \
    --type_optimization $RR_TYPE_OPTIMIZATION \
    --add_linear \
    --data_parallel

  unset PYTHONHOME
  unset PYTHONPATH

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
echo "  SEEDS:     ${RR_SEEDS[*]}"
echo "  EXP_LIST:  ${RR_EXP_LIST[*]}"
echo "============================================"
