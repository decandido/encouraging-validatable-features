# Encouraging Validatable Features in DNNs
This repository contains the code to reproduce the results found in the 
paper "Encouraging Validatable Features in Machine Learning-based Highly 
Automated Driving Functionss" by O. De Candido, M. Koller, and W. Utschick, 
published in the [IEEE Transactions on Intelligent Vehicles](https://ieeexplore.ieee.org/document/9765664) in 2022.

## How to use the code
1. Store the training data in `data/` folder, e.g., the extracted lane 
   changes from the highD 
   dataset from this [project](https://github.com/decandido/highD-extract-lane-changes).
2. Run `main.py` to train the DNNs with the network architectures stored in 
   `parameters/params.py`.
3. Run `extract_embeddings.py` to extract the feature embeddings from the 
   trained networks, and calculate k-means on those embeddings.
4. Run `umap_embeddings.py` to calculate the [UMAP](https://umap-learn.readthedocs.io/en/latest/) representations of the feature embeddings.
5. (Optional) Update the network architectures in `parameters/params.py` and 
   rerun the scripts.

## Requirements
The required packages can be installed via the `requirements.txt` file, e.g.,
`conda install -r requirements.txt`.
This code was tested using `Python v3.7.0`.

## Paper Reference
```angular2html
@article{decandido2022encouraging,
    author={De Candido, Oliver and Koller, Michael and Utschick, Wolfgang},
    title={Encouraging Validatable Features in Machine Learning-based Highly Automated Driving Functions},
    journal={IEEE Trans. on Intell. Vehicles},
    year={2022},
    doi={10.1109/TIV.2022.3171215},}
}
```
### Note
This paper is an extention of our [2020 IEEE ITSC paper](https://ieeexplore.ieee.org/document/9294555).
The code relating to that project can be found [here](https://github.com/decandido/feature-validation).
