# Sci-ZSEL

# install
bash install.sh

# Convert into expected data format
cd data_preparation
bash grag_to_blink.sh

# 5.1 Entity Selection from the Ontology
<!-- from root dir --> 
cd modeling/BLINK/
bash scripts/get_biencoder_top_k.sh

# 5.2 Pseudo-Pair Construction
<!-- cd ../../../ (from root dir) --> 
cd data_preparation
bash entity_selection_and_pseudo_pair_construction.sh 

# 5.3 Ontology-Aware Retriever Fine-tuning
cd modeling/BLINK/
bash scripts/retriever_fine_tuning.sh

# 5.4 Reranker Fine-tuning
cd modeling/reranker
bash scripts/blink_reranker_fine_tuning.sh
bash scripts/res_reranker_fine_tuning.sh


