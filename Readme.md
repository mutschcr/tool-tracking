# Measurement Data

The measurement data repository is a collection of:
- _labeled_ data
- information about measurements
- plots, visualization of the data

Setup an environment and start Jupyter Notebook
```
conda env create -f environment.yml
conda activate mdenv
pip install seglearn
cd data_tools
python deploy.py
cd ..
python -m jupyter notebook
```

The mandatory structure of this repository is as follows:

![Repository structure](info/structure_scheme.jpg)  

You have one dedicated folder for each __(hand) tool__. On the next level are folders for each __measurement campaign__. The undermost level contains all __measurements__ of the corresponding measurement campaign.

The preferred way to access the data for analysis and learning is using the `datatools` package which is part of this repository. You can find more information about the usage of that package in the corresponding folder.
