from setuptools import setup, find_packages

setup(
    name="fhg-utils",
    packages=find_packages(),
    version="0.1.0",
    description="Util classes and functions for working with the tool tracking data",
    maintainer="wllr",
    install_requires=[
        "scikit-learn",
        "seglearn",
        "numpy>=1.18",
        "tqdm>=4.44",
    ]
)
