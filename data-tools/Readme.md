# `datatools`
[![pipeline status](https://git01.iis.fhg.de/abt-la/mlv/data-tools/badges/develop/pipeline.svg)](https://git01.iis.fhg.de/abt-la/mlv/data-tools/-/commits/develop)  [![coverage report](https://git01.iis.fhg.de/abt-la/mlv/data-tools/badges/develop/coverage.svg)](https://git01.iis.fhg.de/abt-la/mlv/data-tools/-/commits/develop)

## Installation
```
conda install data-tools -c http://ux1702:8083/fraunhofer/ -c conda-forge
```

## Documentation
see [sphinx docu](http://abt-la.git01.iis.fhg.de/mlv/data-tools/datatools.html) for more information

## Development

### Installation
```
git clone git@git01.iis.fhg.de:abt-la/mlv/data-tools.git
cd data-tools
pip install -e .
```

Verify setup with `pytest tests`.

### Testing
After installation, you can launch the test suite from outside the source directory:
```
(pip install pytest)
pytest tests
```

### Code Style
Check your code style compliance _before_ commiting from outside the source directory with `flake8`
The project-specific configurations of flake8 (e.g. max line length) can be found in `setup.cfg` and should be appended 
as command line options when running `flake8` locally:
```
(pip install flake8 flake8-builtins flake8-mutable)
flake8 datatools (--max-line-lenght=120 ...)
```

### Git-Commit
> Attention: After running the tests locally `config.ini` is changed. Do not commit those changes! 
> Use `git checkout -- datatools/config.ini` to discard the changes.
