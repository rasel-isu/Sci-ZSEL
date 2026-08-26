eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/qli-lab/rasel/projects/Sci-ZSEL/Read-and-Select/conda_env
# conda activate /lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/conda_data_env

export WANDB_MODE=disabled
python train_and_eval.py 
