# Tool Tracking Dataset - Supplement Code

In order to get things running:

1. Clone the repository

```
git clone https://github.com/mutschcr/tool-tracking.git
cd tool-tracking
```

2. Then you need to download the measurement data from an external host:
```
wget https://owncloud.fraunhofer.de/index.php/s/MQUpf2vhIghAtke/download -O tool-tracking-data.zip
unzip tool-tracking-data.zip && rm tool-tracking-data.zip
```

3. Setup a virtual python environment (e.g. with [conda](https://www.anaconda.com/))
```
conda create --name tool-tracking_env python=3.7
conda activate tool-tracking_env
pip install -r requirements.txt
```

4. Get introduced on how to load and work with the data

Start [Jupyter](https://jupyter.org/) and run the both notebook `How-to-load-the-data.ipynb` and `plot_window_sizes.ipynb` with:
```
jupyter notebook
```

Changelog:
- 2020-08-31: Add <a href="https://htmlpreview.github.io/?https://github.com/mutschcr/tool-tracking/blob/master/html/index.html">HTML API-docs</a> and user guide
- 2020-08-14: Update dataset with enhanced rivetter labels
- 2020-08-12: Update dataset with twice the amount of labeled data; enhanced labels.
- 2020-07-29: Update data loader and notebooks

Known issues:
- 2020-08-12: Enforcing window lenghts when segmenting rivetter data lets two very short windows (label -1) slip through. These cause problems and need to be filtered out: 'filter_labels(..)'

License:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
