# -*- coding: utf-8 -*-
"""
Deployment script to add package modules to pythonpath via usersites.
Execute the first time before using datatools.
"""

import os
import site
import logging
import pathlib
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [Deploy] %(message)s')


# ---------------
# --- parsing ---
# ---------------

parser = argparse.ArgumentParser(description="Deployment script")
parser.add_argument("--version", dest="version", type=str)
args = parser.parse_args()


# --------------------
# --- checking out ---
# --------------------

version = args.version
try:
    from git import Repo, exc
    try:
        repo_root = pathlib.Path(__file__).absolute().parents[1]
        logging.info(f"Try to find repo at '{repo_root}'")
        local_repo = Repo(repo_root)
        git = local_repo.git

        # checkout desired version
        if version is not None:
            try:
                git.checkout(version)
                logging.info(f"Version {version} is selected")
            except exc.GitCommandError as e:
                logging.warning(f"Version {version} is not available or valid."
                                f" Maybe you have to stash or commit your current changes.")
                logging.debug(e)
                tags = str(git.tag()).replace("\n", ", ")
                logging.info(f"The following versions are available: {tags}")

        # checkout master
        else:
            git.checkout('master')
            logging.info("Check out master")

    except exc.InvalidGitRepositoryError:
        logging.error("Could not find the measurement-data repo")


except (ImportError, ModuleNotFoundError):
    logging.warning("Package GitPython is missing")


# ------------------
# --- deployment ---
# ------------------

filename = os.path.join(site.USER_SITE, "data_tools.pth")

if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

# Add packages to user path
root_dir = os.path.abspath('')

logging.info(f"Start deployment --> '{filename}'")

with open(filename, "w") as file:
    file.write(str(root_dir))

try:
    import datatools
    logging.info(f"Complete deployment of {f'version {datatools.__version__}' if version is not None else 'master'}"
                 f" successfully")
except (ImportError, ModuleNotFoundError):
    logging.error(f"Deployment failed, please contact {datatools.__maintainer__}")
