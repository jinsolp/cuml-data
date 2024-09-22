This repo is for testing HDBSCAN in [cuML](https://github.com/rapidsai/cuml)

### Building cuML
To run cuML's HDBSCAN with NN Descent as the build algorithm for knn, build cuML on [this PR](https://github.com/rapidsai/cuml/pull/5939)
To have additional features of running cuML's HDBSCAN with data on host (enables batched NN Descent), build cuML on [this PR](https://github.com/rapidsai/cuml/pull/6044)

### Cloning this repo and Downloading Data
```
git clone --recursive https://github.com/jinsolp/cuml-data.git
cd cuml-data

pip install -r requirements.txt
sh download_datasets.sh
```

### Running Tests
Detailed options for running test inside `tests.py` file.
```
cd cuml-data
python3 tests.py
```
