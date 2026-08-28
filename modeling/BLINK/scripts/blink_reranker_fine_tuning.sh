eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel

export WANDB_MODE=disabled

ROOT="../.."
BLINK_MODEL_PATH="$ROOT/saved_models/crossencoder_wiki_large.bin"
SPLITNAME="train"

EXP_LIST=(
    # "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime"
    # "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"
    # "synonym"
    "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"
)
ONTO_LIST=("ncbi_disease")
SEEDS=(0)
TRAIN_BATCH_SIZE=16
EPOCH=3
TIME_START=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="logs/cross/$(date '+%Y-%m-%d-%H-%M-%S').log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "============================================"
echo "Job started at: $START_TIMESTAMP"
echo "SEEDS:     ${SEEDS[*]}"
echo "ONTO_LIST: ${ONTO_LIST[*]}"
echo "EXP_LIST:  ${EXP_LIST[*]}"
echo "============================================"




for ONTO in "${ONTO_LIST[@]}"; do
  echo "Getting candidates from retriever for testset ...."
  bash scripts/get_biencoder_cands_for_reranker_training.sh original_data test $ONTO
  echo "Done! got candidates from retriever for testset!"

for EXP in "${EXP_LIST[@]}"; do
  echo "Getting candidates from retriever for $EXP ...."
  bash scripts/get_biencoder_cands_for_reranker_training.sh $EXP train $ONTO
  echo "Done! got candidates from retriever for $EXP!"

for SEED in "${SEEDS[@]}"; do
  if [ "$ONTO" = "ncbi_disease" ]; then
    KB_FILE_PATH="$ROOT/datasets/$ONTO/medic.json"
    CANDIDATE_POOL_PATH="$ROOT/saved_models/$ONTO/medic_entity_pool.t7"
    GRAG_DATA_PATH="$ROOT/datasets/$ONTO/"
  else
    echo "Unknown ontology: $KB_FILE_PATH"
    exit 1
  fi
  echo "KB: $KB_FILE_PATH"
  
  RAW_DATA_PATH="$ROOT/datasets/$ONTO/blink_format/$SPLITNAME/$EXP/"
  DATA_PATH="$ROOT/saved_models/$ONTO/biencoder/$SPLITNAME/$EXP/top64_candidates"
  TEST_DATA_PATH="$ROOT/saved_models/$ONTO/biencoder/$SPLITNAME/original_data/top64_candidates"
  OUTPUT_DIR="$ROOT/saved_models/$ONTO/crossencoder/$SPLITNAME/fine-tune/seed-$SEED/$EXP"

  export PYTHONPATH=.
  # python -m debugpy --listen 0.0.0.0:5677 --wait-for-client blink/crossencoder/train_cross.py \
  python blink/crossencoder/train_cross.py \
    --data_path  $DATA_PATH \
    --raw_data_path $RAW_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --grag_data_path $GRAG_DATA_PATH \
    --kb_file_path $KB_FILE_PATH \
    --cand_pool_path $CANDIDATE_POOL_PATH \
    --output_path $OUTPUT_DIR \
    --onto $ONTO \
    --experiment $EXP \
    --seed  $SEED \
    --blink_model_path $BLINK_MODEL_PATH \
    --learning_rate 2e-05 \
    --dropout_rate 0.2 \
    --num_train_epochs $EPOCH \
    --max_context_length 64 \
    --max_cand_length 128 \
    --max_seq_length 192 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --eval_batch_size 32 \
    --cross_enc_negative_selection only_bienc_20_neg \
    --bert_model bert-large-uncased \
    --type_optimization all_encoder_layers \
    --add_linear \
    --data_parallel 

  unset PYTHONHOME
  unset PYTHONPATH

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
echo "  ONTO_LIST: ${ONTO_LIST[*]}"
echo "  EXP_LIST:  ${EXP_LIST[*]}"
echo "============================================"

