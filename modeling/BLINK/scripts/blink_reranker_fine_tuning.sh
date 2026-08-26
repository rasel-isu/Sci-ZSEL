

# if you include "prime" keyword in EXP then it will use prime as label title, otherwise not.
# So, if keep "prime" it will use prime if not then will use original title


export WANDB_MODE=disabled

EXP_LIST=(

    # "original_def_prmbl"

    "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"
    "synonym"
    "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime"
    "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"

    # "synonymU(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"
    # "synonymU(m1_e1)"
    
)

SPLITNAME="train"
# SPLITNAME="test"

ONTO_LIST=(
  # "lpt"
  # "vt"
  # "cmo"
  # "ncbi"
  # "bc5cdr"
  'cometa'
)

SEEDS=(
  0
  42
  52313
)

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

for SEED in "${SEEDS[@]}"; do
for ONTO in "${ONTO_LIST[@]}"; do
  for EXP in "${EXP_LIST[@]}"; do
    # cd /lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK
    eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
    source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
    conda activate /lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/conda_env
    export PYTHONPATH=.
    

    RAW_DATA_PATH="data/$ONTO/blink_format/$SPLITNAME/$EXP/"
    BLINK_MODEL_PATH="models/crossencoder_wiki_large.bin"
    DATA_PATH="models/$ONTO/biencoder/$SPLITNAME/$EXP/top64_candidates"
    
    TEST_DATA_PATH="models/$ONTO/biencoder/$SPLITNAME/original_def_prmbl/top64_candidates"
    # TEST_DATA_PATH="models/$ONTO/biencoder/$SPLITNAME/original_def/top64_candidates"
    # TEST_DATA_PATH="models/$ONTO/biencoder/$SPLITNAME/$EXP/top64_candidates"
    # DATA_PATH="/lustre/hdd/LAS/qli-lab/nhat/KANG_BLINK/BLINK_rasel/models/ncbi/top64_candidates_pretrained_original_def_64"
    
    OUTPUT_DIR="models/$ONTO/crossencoder/$SPLITNAME/fine-tune/seed-$SEED/$EXP"

    if [ "$ONTO" = "ncbi" ]; then
      KB_FILE_PATH="data/$ONTO/onto/only_medic_def.json"
      CANDIDATE_POOL_PATH="models/$ONTO/only_medic_def_entity_pool.t7"
      GRAG_DATA_PATH="data/$ONTO/Prompt-BioEL"
    elif [ "$ONTO" = "bc5cdr" ]; then
        KB_FILE_PATH="data/$ONTO/onto/merged_onto.json"
        CANDIDATE_POOL_PATH="models/$ONTO/mesh_entity_pool.t7"
        GRAG_DATA_PATH="data/$ONTO/Prompt-BioEL"
    elif [ "$ONTO" = "cmo" ]; then
        KB_FILE_PATH="data/$ONTO/onto/cmo_kb.json"
        CANDIDATE_POOL_PATH="models/$ONTO/cmo_entity_pool.t7"
        GRAG_DATA_PATH="data/$ONTO/"
    elif [ "$ONTO" = "vt" ]; then
        KB_FILE_PATH="data/$ONTO/onto/vt_kb.json"
        CANDIDATE_POOL_PATH="models/$ONTO/vt_entity_pool.t7"
        GRAG_DATA_PATH="data/$ONTO/"
    elif [ "$ONTO" = "lpt" ]; then
        KB_FILE_PATH="data/$ONTO/onto/lpt_kb.json"
        CANDIDATE_POOL_PATH="models/$ONTO/lpt_entity_pool.t7"
        GRAG_DATA_PATH="data/$ONTO/"
    elif [ "$ONTO" = "cometa" ]; then
        KB_FILE_PATH="data/$ONTO/onto/SNOMEDCT.json"
        CANDIDATE_POOL_PATH="models/$ONTO/snomedct_entity_pool.t7"
        GRAG_DATA_PATH="data/$ONTO/"
    else
      echo "Unknown ontology: $KB_FILE_PATH"
      exit 1
    fi
    echo "KB: $KB_FILE_PATH"
    
    # --cross_enc_negative_selection only_bienc_63_neg 
    #  bienc_20_neg_and_prnt_chld_as_neg 
    #  only_bienc_20_neg
    #  only_prnt_chld_as_neg 
    #  prnt_chld_as_pos
    #  crsenc_ranked_neg_after_label
    # comb_crsenc_ranked_neg_and_prnt_chld_as_neg 

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
      --num_train_epochs 3 \
      --max_context_length 64 \
      --max_cand_length 128 \
      --max_seq_length 192 \
      --train_batch_size 16 \
      --gradient_accumulation_steps 2 \
      --eval_batch_size 32 \
      --cross_enc_negative_selection only_bienc_20_neg \
      --bert_model bert-large-uncased \
      --type_optimization all_encoder_layers \
      --add_linear \
      --data_parallel \
      # --only_infer_test_set \
      # --only_test_each_epoch \
      # --exclude_gt \

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

