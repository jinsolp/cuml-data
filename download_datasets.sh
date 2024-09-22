#!/bin/bash
mkdir data

# download gist (1M, 960) and deep (10M, 96) datasets
python ann-benchmarks/ann_benchmarks/datasets.py
mv ann-benchmarks/ann_benchmarks/data/gist-960-euclidean.hdf5 ./data
mv ann-benchmarks/ann_benchmarks/data/deep-image-96-angular.hdf5 ./data

# download and process Amazon food dataset (5M, 384)
wget -P ./data/ https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Grocery_and_Gourmet_Food.json.gz
gzip -d ./data/Grocery_and_Gourmet_Food.json.gz
python3 python/json_to_pkl.py --json-path data/Grocery_and_Gourmet_Food.json --pkl-path data/Grocery_and_Gourmet_Food.pkl

# download raft's wiki-all dataset (88M, 768)
mkdir data/wiki_all_88M/
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.{00..9} | tar -xf - -C data/wiki_all_88M/