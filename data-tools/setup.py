from setuptools import setup, find_packages
from setup_utils import get_meta_info_from_version_file

meta = get_meta_info_from_version_file(__file__, "datatools")

setup(
    name="data-tools",
    packages=find_packages(),
    package_data={"": ["*.ini"]},
    entry_points={
        'console_scripts': [
            'datatools=datatools.cli:cli'
        ]
    },
    version=meta.version,
    description=meta.description,
    maintainer=meta.maintainer,
    install_requires=[
        "numpy>=1.18",
        "pandas>1.0",
        "click>=6.7",
        "tqdm>=4.44",
        "python-dateutil>=2.8"
    ]
)
