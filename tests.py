#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
import h5py
import argparse
import numpy as np
import rmm
import os
import pickle as pkl
from cuml.cluster import HDBSCAN
import sklearn
import time

def memmap_bin_file(
    bin_file, dtype, shape=None, mode="r", size_dtype=np.uint32
):
    extent_itemsize = np.dtype(size_dtype).itemsize
    offset = int(extent_itemsize) * 2
    if bin_file is None:
        return None
    if dtype is None:
        dtype = np.float32
    if mode[0] == "r":
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        if shape is None:
            shape = (a[0], a[1])
        else:
            shape = tuple(
                [
                    aval if sval is None else sval
                    for aval, sval in zip(a, shape)
                ]
            )
        return np.memmap(
            bin_file, mode=mode, dtype=dtype, offset=offset, shape=shape
        )
    elif mode[0] == "w":
        if shape is None:
            raise ValueError("Need to specify shape to map file in write mode")
        print("creating file", bin_file)
        dirname = os.path.dirname(bin_file)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        a[0] = shape[0]
        a[1] = shape[1]
        a.flush()
        del a
        fp = np.memmap(
            bin_file, mode="r+", dtype=dtype, offset=offset, shape=shape
        )
        return fp
    
parser = argparse.ArgumentParser()


# Using nn descent build algo requires that cuML is built on this PR https://github.com/rapidsai/cuml/pull/5939

parser.add_argument("--data", default="gist", help="[gist, food, deep, wiki]")

# Putting data on host only works when using nn descent, and requires that cuML is built on this PR https://github.com/rapidsai/cuml/pull/6044
parser.add_argument("--host", action="store_true", help="putting data on host")
parser.add_argument("--cluster", default=1, type=int)

parser.add_argument("--rows", default=0, type=int, help="number of rows to take from the original dataset, defaults to using all rows")
parser.add_argument("--cols", default=0, type=int, help="number of cols to take from the original dataset, defaults to using all cols")

parser.add_argument("--gd", default=64, type=int, help="graph degree of using nn descent")
parser.add_argument("--igd", default=128, type=int, help="intermediate graph degree of using nn descent")
parser.add_argument("--iters", default=20, type=int, help="number of iters for running nn descent")

args = parser.parse_args()
name_to_path = {"gist": "data/gist-960-euclidean.hdf5",
                "food": "data/Grocery_and_Gourmet_Food.pkl",
                "deep": "data/deep-image-96-angular.hdf5",
                "wiki": "data/wiki_all_88M/base.88M.fbin"}

if name_to_path[args.data][-4:] == "hdf5":
    hf = h5py.File(name_to_path[args.data], 'r')
    data = np.array(hf['train'])
elif name_to_path[args.data][-3:] == "pkl":
    with open(name_to_path[args.data], 'rb') as file:
        data = pkl.load(file)
elif name_to_path[args.data][-3:] == "bin":
    data = memmap_bin_file(name_to_path[args.data], None)
    data = np.asarray(data)
        
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),
    initial_pool_size=2**30,
    maximum_pool_size=2**50
)
rmm.mr.set_current_device_resource(pool)

if args.rows != 0:
    data = data[:args.rows, :]
if args.cols != 0:
    data = data[:, :args.cols]
print(args)
print(data.shape, type(data))

min_samples = 12

hdbscan_nnd = HDBSCAN(min_samples=min_samples, build_algo="nn_descent", build_kwds={'nnd_graph_degree': args.gd, 'nnd_intermediate_graph_degree': args.igd,
    'nnd_max_iterations': args.iters, 'nnd_return_distances': True, "nnd_n_clusters": args.cluster})


start = time.time()
# use this when running with argument host
# labels_nnd = hdbscan_nnd.fit(data, data_on_host=args.host).labels_
labels_nnd = hdbscan_nnd.fit(data).labels_
end = time.time()

print(f"Done running hdbscan with nnd - time taken: {(end - start) * 1000} ms\n")
    
if os.path.exists(f"hdbscan_bfk_{args.data}_{args.rows}_{args.cols}_{min_samples}_label.pkl"):
    print("using previously run result of bfk HDBSCAN")
    with open(f"hdbscan_bfk_{args.data}_{args.rows}_{args.cols}_{min_samples}_label.pkl", 'rb') as file:
        labels_bfk = pkl.load(file)
else:
    hdbscan_bfk = HDBSCAN(min_samples=min_samples, build_algo="brute_force_knn")
    start = time.time()
    labels_bfk = hdbscan_bfk.fit(data).labels_
    end = time.time()
    
    print(f"Done running hdbscan with bfk - time taken: {(end - start) * 1000} ms\n")

    with open(f"hdbscan_bfk_{args.data}_{args.rows}_{args.cols}_{min_samples}_label.pkl", 'wb') as f:
        pkl.dump(labels_bfk, f)

score = sklearn.metrics.adjusted_rand_score(labels_nnd, labels_bfk)
print(score)
