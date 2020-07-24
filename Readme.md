# Tool Tracking Dataset - Supplement Code

In order to get things running:

1. Clone the repository

```
git clone https://github.com/mutschcr/tool-tracking.git
cd tool-tracking
```

2. Then you need to download the measurement data from an external host:
```
wget https://owncloud.fraunhofer.de/index.php/s/UOzJU8ypG3ZKKte/download -O electric_screwdriver.tar.gz
tar -xzvf electric_screwdriver.tar.gz && rm electric_screwdriver.tar.gz

wget https://owncloud.fraunhofer.de/index.php/s/ZisA9yrSb0cwAmV/download -O pneumatic_riveting_gun.tar.gz
tar -xzvf pneumatic_riveting_gun.tar.gz && rm pneumatic_riveting_gun.tar.gz

wget https://owncloud.fraunhofer.de/index.php/s/WOiEDXTOz2JgWCD/download -O pneumatic_screwdriver.tar.gz
tar -xzvf pneumatic_screwdriver.tar.gz && rm pneumatic_screwdriver.tar.gz
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

License:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
