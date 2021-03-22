# Final project : Bayesian Machine Learning

This code illustrates our report on the paper "Inconsistency of Pitman-Yor Process Mixtures for the Number of Components", by Jeffrey W. Miller, Matthew T. Harrison.
## Install
### Requirements
```
pip install -r requirements.txt
```
### Thyroid data
Data taken from https://www.di.ens.fr/~cappe/fr/Enseignement/data/thyroid/index.html
```
./get_data.sh
```
## Run
### Synthetic experiments

Plot the posterior distribution of cluster size:

```python synthetic_experiments.py --distribution {poisson, gaussian}```
### Thyroid experiments
```python thyroid.py```

