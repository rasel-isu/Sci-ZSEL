# Sci-ZSEL

# install
bash install.sh

# grag_to_blink
bash data_preparation/grag_to_blink.sh

# Ent selection 

<!-- from root dir --> 
cd models/reranker/BLINK/
bash scripts/get_biencoder_top_k.sh

<!-- cd ../../../ (from root dir) --> 
cd data_preparation
bash entity_selection_and_pseudo_pair_construction.sh 




