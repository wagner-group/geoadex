# Adversarial Examples for k-Nearest Neighbor Classifiers Based on Higher-Order Voronoi Diagrams

Published in NeurIPS 2021 [[arXiv]](https://arxiv.org/abs/2011.09719)

## Abstract

Adversarial examples are a widely studied phenomenon in machine learning models. While most of the attention has been focused on neural networks, other practical models also suffer from this issue. In this work, we propose an algorithm for evaluating the adversarial robustness of $k$-nearest neighbor classification, i.e., finding a minimum-norm adversarial example. Diverging from previous proposals, we propose the first geometric approach by performing a search that expands outwards from a given input point. On a high level, the search radius expands to the nearby higher-order Voronoi cells until we find a cell that classifies differently from the input point. To scale the algorithm to a large $k$, we introduce approximation steps that find perturbation with smaller norm, compared to the baselines, in a variety of datasets. Furthermore, we analyze the structural properties of a dataset where our approach outperforms the competition.

## Files

- `test_script.py`: Main test script with hyperparameters.
- `lib/geoadex.py`: Main implementation of GeoAdEx.
- `test_sw.py`: For testing Sitawarin & Wagner [2020].
- See `yang_et_al` and `wang_et_al` for running experiments with Yang et al. [2019] and Wang et al. [2020], respectively. We modified the source code in these repositories only to add new datasets and to compute confidence intervals with random train/test splits. 
- Dataset can be downloaded from [link](https://github.com/yangarbiter/adversarial-nonparametrics/tree/master/nnattack/datasets/files) and [link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html).

## Setup Guide

### GeoAdEx

Packages and requirements:

- Python3
- numpy
- torch
- scikit-learn
- scipy
- faiss-cpu
- joblib
- pyskiplist
- keras (for loading FMNIST datasets)

```[bash]
conda env create -f environment.yml
mkdir data save
# Download datasets and put in ./data/ 
cp yang_et_al/nnattack/datasets/files/* data/
cd data
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t
cd ..
# Run the test script
python test_script.py
```

### Yang et al. [2020]

- Reference: [https://github.com/yangarbiter/adversarial-nonparametrics](https://github.com/yangarbiter/adversarial-nonparametrics)
- Requires Gurobi license (academic license available).
- Other requirements: `gcc, blas, lapack`.

```[bash]
# Need conda with python 3.7 for gurobi
conda create -n yang python=3.7
conda activate yang
conda install -c gurobi gurobi=8.1.1    # using gurobi 9 may run into a bug with "BarIterCount"

cd yang_et_al
pip install --upgrade pip
pip install --upgrade -r requirements.txt
pip install autovar keras==2.3  # only keras <= 2.3 is compatible with tensorflow 1.15

cp ../data/letter.scale* nnattack/datasets/files/
./setup.py build_ext -i
sh run.sh
```

If you run into a `numpy` error, try `conda install numpy`.

### Wang et al. [2019]

Reference: [https://github.com/wangwllu/knn_robustness](https://github.com/wangwllu/knn_robustness)  

### Sitawarin & Wagner [2020]

Reference: [https://github.com/chawins/knn-defense](https://github.com/chawins/knn-defense)  

- We also include the attack implementation of Sitawarin & Wagner [2020] in `lib/dknn_attack_v2.py`.
- To run the stand-alone Sitawarin & Wagner attack, run `python test_sw.py`.
- `test_sw.py` shares similar structure with `test_script.py` and uses the same data loader.
