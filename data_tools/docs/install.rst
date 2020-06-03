#######
Install
#######

Dependencies
============

- clone the repository
- enter the repository folder
- now choose one of the following cases depending on your precondition:

No pre-exisiting conda environment
----------------------------------

- install dependencies using a conda environment with :code:`conda env create --file environment.yml`

.. note::

   Using :code:`conda env create`, you create an virtual environment with it's isolated python interpreter and packages. The dependencies for this repository are listed in the :code:`environment.yml`.

- activate the new environment :code:`mdenv` using :code:`conda activate mdenv`

.. note::

   This is nesessary every time you start the terminal to run the app. The command line should start with :code:`(mdenv)` if the needed environment is active.

- finally run the app with :code:`python khapp.py`

Update existing conda environment
-------------------------------------

- activate the environment you want to update: :code:`conda activate <your_environment>`
- install dependencies using: :code:`conda env update --file environment.yml`

.. note::

   Updating the dependencies using the :code:`environment.yml` can lead to wrong package versions for your code.


Installation
============
- open a console in :code:`<path>/measurement-data/data_tools` and enter :code:`python deploy.py [--version VERSION]`

.. note::

   You can pass a desired version of :code:`datatools` using the :code:`--version` flag. The available versions can be shown with the :code:`git tag` command. If version is left out the master will be checked out.
