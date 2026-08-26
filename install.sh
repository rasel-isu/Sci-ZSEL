eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh

conda create --prefix /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/envs/sci-zsel python=3.11 -y

conda activate sci-zsel

pip install -r requirements.txt

