eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel-res
export WANDB_MODE=disabled
# python get_retriever_candidates.py
python train_and_eval.py 

