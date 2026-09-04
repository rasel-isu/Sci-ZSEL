eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh

conda create --prefix /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/envs/sci-zsel python=3.11 -y
conda activate sci-zsel
pip install -r modeling/BLINK/requirements.txt
conda deactivate


conda create --prefix /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/envs/sci-zsel-res python=3.9.25 -y
conda activate sci-zsel-res
pip install -r modeling/ReS/requirements.txt
conda deactivate




