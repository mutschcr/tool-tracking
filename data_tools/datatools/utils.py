# -*- coding: utf-8 -*-

# Built-in/Generic Imports
import sys
import uuid
import pickle
import pathlib
import tempfile
import itertools
import collections
import functools

# Libs
import pandas as pd
import numpy as np

# Own modules
from .constants import *
from datatools import __version__


# ---------------
# --- classes ---
# ---------------
class MSGPrinter:
    LEVEL_NAMES = {'error': 0,
                   'warn': 1,
                   'info': 2,
                   'indent': 2,
                   'debug': 3,
                   }

    def __init__(self, verbose=2):
        self.verbose = verbose

    def __call__(self, string, cat=None):
        levels = MSGPrinter.LEVEL_NAMES.keys()

        if MSGPrinter.LEVEL_NAMES.get(cat, 0) <= self.verbose:
            if cat == 'indent':
                cat = '  '
            else:
                cat = f"[{cat.upper()}] "

            print(cat + string)

    def set_verbosity(self, verbose):
        if 3 >= self.verbose >= 0:
            self.verbose = verbose
        else:
            raise ValueError("No valid verbosity level (0-3).")


print_msg = MSGPrinter()  # preferred alias


class ProgressBar:
    def __init__(self, total, num_steps=10):
        self.total = total
        self.n = 0
        self.num_steps = num_steps
        self.finished = False
        self.description = ''
        self.postfix = ''
        self.max_string_length = 0

        self.loading_cycle = itertools.cycle(["▜", "▚", "▙", "▛", "▞", "▟"])

        sys.stdout.write(f"[{' ' * self.num_steps}] 0%")
        sys.stdout.flush()

    def set_description(self, description, **kwargs):
        self.description = description + " "

    def set_postfix(self, file, **kwargs):
        self.postfix = ", file=" + file

    def update(self, n=1):
        if not self.finished:
            if n < 0:
                raise ValueError(f"n ({n}) cannot be negative")

            self.n += n

            progress = (self.n / self.total)
            current_num_steps = round(progress * self.num_steps)

            if progress > 1.0:
                raise ValueError(f"progress cannot be bigger than 1.0 but is {progress}!")

            sys.stdout.write('\r')
            output_str = f"{self.description}" \
                         f"{next(self.loading_cycle)}[" \
                         f"{'='*current_num_steps + ' '*(self.num_steps-current_num_steps)}] " \
                         f"{round(progress * 100)}%" \
                         f"{self.postfix}"
            self.max_string_length = len(output_str) if len(output_str) > self.max_string_length else self.max_string_length

            sys.stdout.write(output_str + " "*(self.max_string_length - len(output_str)))
            sys.stdout.flush()

    def close(self):
        if not self.finished:
            self.finished = True
            if self.total == self.n:
                sys.stdout.write(' ✔\n')
                sys.stdout.flush()
            else:
                sys.stdout.write(' ✘\n')
                sys.stdout.flush()


class SamplingRateError(Exception):
    def __init__(self, message=''):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class TimeColumnError(Exception):
    def __init__(self, message=''):
        super().__init__(message)


class ConcatError(Exception):
    def __init__(self, message=''):
        super().__init__(message)


FILE_FORMAT = ["time [s]", "acceleration x-axis [m/s²]", "acceleration y-axis [m/s²]", "acceleration z-axis [m/s²]",
               "angular rate x-axis [°/s]", "angular rate y-axis [°/s]", "angular rate z-axis [°/s]",
               "magnetic field x-axis [Gs]", "magnetic field y-axis [Gs]", "magnetic field z-axis [Gs]", "label"]


# -----------------
# --- functions ---
# -----------------
def convert_old_file_format(path, delimiter=",", no_labels=False):
    path = pathlib.Path(path)
    for file in path.iterdir():
        if file.is_file() and file.suffix == '.csv':
            print_msg(f"Processing {file.name}", "info")
            df = pd.read_csv(file, delimiter=delimiter)

            if no_labels:
                if ",".join(FILE_FORMAT[0:-1]) != ",".join(df.columns):
                    raise ValueError("Wrong file format.")

                df_IMU = df.drop(columns=[df.columns[7], df.columns[8], df.columns[9]])
                df_MF = df.drop(columns=[df.columns[1], df.columns[2], df.columns[3],
                                         df.columns[4], df.columns[5], df.columns[6]])

                for df, prefix in zip([df_IMU, df_MF], ["IMU", "MF"]):
                    id_, sr = file.name.split('.')[0].split('-')[1:3]
                    df.to_csv(file.parent / ('-'.join([prefix, id_, sr]) + DATA_FILE_EXTENSION),
                              index=False, sep=CSV_DELIMITER)

            else:
                if ",".join(FILE_FORMAT) != ",".join(df.columns):
                    raise ValueError("Wrong file format.")

                df_IMU = df.drop(columns=[df.columns[7], df.columns[8], df.columns[9], df.columns[10]])
                df_MF = df.drop(columns=[df.columns[1], df.columns[2], df.columns[3],
                                         df.columns[4], df.columns[5], df.columns[6], df.columns[10]])
                df_labels = df[df.columns[10]]

                for df, prefix in zip([df_IMU, df_MF, df_labels], ["IMU", "MF", "labels"]):
                    id_, sr = file.name.split('.')[0].split('-')[1:3]
                    df.to_csv(file.parent / ('-'.join([prefix, id_, sr]) + DATA_FILE_EXTENSION),
                              index=False, sep=CSV_DELIMITER)
    print_msg("Finished conversion.", 'info')


def convert_log_files(path):
    path = pathlib.Path(path)
    for file in path.iterdir():
        if file.is_file() and file.suffix == '.log' and 'sensor' in file.name:
            print_msg(f"Processing {file.name}", "info")
            df = pd.read_csv(file, delimiter=', ', header=None, names=FILE_FORMAT[0:-1])
            id_ = file.name.split('.')[0].split('-')[-1]

            # convert timestamps
            df[df.columns[0]] -= df[df.columns[0]][0]
            df[df.columns[0]] /= 1000

            sr = int(np.round(1 / np.mean(np.diff(df[df.columns[0]]))))
            print_msg(f"Detected a sampling rate of {sr} Hz", 'info')

            df_IMU = df.drop(columns=[df.columns[7], df.columns[8], df.columns[9]])
            df_MF = df.drop(columns=[df.columns[1], df.columns[2], df.columns[3],
                                     df.columns[4], df.columns[5], df.columns[6]])

            for df, prefix in zip([df_IMU, df_MF], ["IMU", "MF"]):
                df.to_csv(file.parent / ('-'.join([prefix, id_, str(sr)]) + DATA_FILE_EXTENSION),
                          index=False, sep=CSV_DELIMITER)
    print_msg("Finished conversion.", 'info')


def check_master_up_to_date():
    """
    Checks if it's master of the current local repository is up to date with the remote repository
     and allows to pull changes if desired.
    """
    print_msg("Checking if master is up to date ...", 'info')

    try:
        from git import Repo, Diff, exc
    except (ImportError, ModuleNotFoundError):
        print_msg("Skipping check because GitPython could not be found.\n"
                  "Install GitPython (pip install gitpython) for auto-checking.",
                  'warn')
        return

    try:
        local_repo = Repo(pathlib.Path(__file__).parents[2])
        local_master = local_repo.heads.master
        correct_flag = False

        # get access to the local git instance
        git = local_repo.git

        try:
            # get active branch
            active_branch = local_repo.active_branch
            # check if the active branch is the master branch
            if not active_branch == local_master:
                print_msg("Checkout the master branch!", 'warn')
            else:
                # get message which is printed by the `git status` command
                status_msg = git.status()
                correct_flag = False

                # we want to find out using the status msg if
                # the local branch is behind or ahead of the remote master branch
                for item in status_msg.split("\n"):
                    if "ahead" in item:
                        # print the part of the status msg which releated to the 'aheadness'
                        print_msg(item.strip(), 'warn')
                        break

                    elif "behind" in item:
                        # print the part of the status msg which releated to the 'behindness'
                        print_msg(item.strip(), 'error')

                        user_decision = input(
                            "Do you want to pull changes? (master will be checked out) [y]/n: ") or 'y'

                        if user_decision == 'y':
                            git.checkout('master')
                            pull_msg = git.pull()
                            print_msg(pull_msg, 'info')

                        break
                    else:
                        correct_flag = True
        except TypeError:
            # bug in gitpython
            print_msg(f"You do not use the newest version of datatools but (v{__version__})", 'warn')

        if correct_flag:
            print_msg(f"The datatools (v{__version__}) package is set up correctly", 'info')
    except exc.InvalidGitRepositoryError:
        print_msg("Could not find the measurement-data repo", 'error')


def get_sampling_rate(t, t_unit='s', kind='mean', ndigits=None):
    if t_unit == 's':
        factor = 1
    elif t_unit == 'ms':
        factor = 10**-3
    elif t_unit == 'us':
        factor = 10**-6
    else:
        raise ValueError("Unknown unit " + str(t_unit) + " for time!")

    if kind == 'mean':
        sr = 1 / np.mean(np.diff(t * factor))
    elif kind == 'median':
        sr = 1 / np.median(np.diff(t * factor))
    else:
        raise ValueError(f"Unknown kind '{kind}'")

    if ndigits is None:
        return sr
    elif ndigits == 0:
        return int(round(sr, ndigits=ndigits))
    else:
        return round(sr, ndigits=ndigits)


def flatten(d, parent_key="", sep="_", stop_cls=None):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping) and\
                (not isinstance(v, stop_cls) if (stop_cls is not None) else True):
            items.extend(flatten(v, new_key, sep=sep, stop_cls=stop_cls).items())
        else:
            items.append((new_key, v))
    return dict(items)


def pickle_cache(path=tempfile.gettempdir(), size=10):
    path = pathlib.Path(path)
    cache_size = size
    maxlen = cache_size * 10

    def decorator(func):
        mapping = {}
        key_history = collections.deque(maxlen=maxlen)

        def _sanitize():
            if len(mapping) >= cache_size:
                key_least_common = collections.Counter(key_history).most_common(maxlen)[-1][0]
                mapping[key_least_common].unlink()
                del mapping[key_least_common]

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if kwargs.get('caching', False):
                try:
                    key = str(args) + str(tuple(sorted(kwargs.items())))

                    if key in mapping:
                        filename = mapping[key]
                        with open(filename, 'rb') as f:
                            res = pickle.load(f)
                        print_msg("Loading from cache", 'info')
                    else:
                        _sanitize()
                        res = func(self, *args, **kwargs)
                        filename = path / f"{uuid.uuid4()}_mdr_cache.pkl"

                        with open(filename, 'wb') as f:
                            pickle.dump(res, f)

                        mapping.update({key: filename})
                        print_msg("Cached result", 'info')

                    key_history.append(key)

                except Exception as e:
                    res = func(self, *args, **kwargs)
                    print_msg(f"Caching failed: {e}", 'warn')
            else:
                res = func(self, *args, **kwargs)

            return res
        return wrapper
    return decorator
