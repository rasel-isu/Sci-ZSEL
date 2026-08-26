eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel

# Entity Selection
python entity_selection.py

# Alias Generation
bash alias_generation.sh 

# Pseudo Pair Construction
python pseudo_pair_construction.py


