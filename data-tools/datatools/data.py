# --- built-in ---
from __future__ import annotations
from typing import Optional, List, Dict
from collections import defaultdict
from functools import wraps, reduce
from itertools import combinations
from operator import eq, add
from enum import Enum
from pathlib import Path
import abc
import difflib
import warnings
import math
import re

# --- third-party ---
import pandas as pd
import numpy as np

# --- own ---
from .utils import print_msg, SamplingRateError, TimeColumnError, ConcatError, get_sampling_rate
from .constants import CSV_DELIMITER
from datatools import __maintainer__

warnings.simplefilter(action='ignore', category=FutureWarning)


class DataType(Enum):
    """Enum representing all valid data types"""
    ACC = "acc"   # acceleration
    GYR = "gyr"   # gyroscope
    MAG = "mag"   # magnetometer
    POS = "pos"   # position
    VEL = "vel"   # velocity
    MIC = "mic"   # audio data


# bring enums into global space for ease of use
ACC = DataType.ACC
GYR = DataType.GYR
MAG = DataType.MAG
POS = DataType.POS
VEL = DataType.VEL
MIC = DataType.MIC

DATA_TYPES = tuple(type_.value for type_ in DataType)


def consistent_sampling_rate(func):
    @wraps(func)
    def wrapper(self, other, *args, **kwargs):
        if self.sampling_rate != other.sampling_rate:
            raise SamplingRateError(f"Sampling rate does not match!: {self.sampling_rate}Hz ≠ {other.sampling_rate}Hz")
        return func(self, other, *args, **kwargs)
    return wrapper


def consistent_time_column(func):
    @wraps(func)
    def wrapper(self, other, *args, **kwargs):
        time_column = difflib.get_close_matches("time", self.columns)
        try:
            # both df have time columns
            if not np.array_equal(self[time_column[0]], other[time_column[0]]):
                raise TimeColumnError("Time columns do not match!")

            self = self.drop(columns=[time_column[0]])

        except (KeyError, IndexError):
            # at least one df has no time column
            try:
                self = self.drop(columns=[time_column[0]])
            except Exception:
                pass
            try:
                other = self.other(columns=[time_column[0]])
            except Exception:
                pass

            print_msg("At least one df has no time column. Pay attention to the result.", 'warn')

        return func(self, other, *args, **kwargs)
    return wrapper


class ToolDataBase(pd.DataFrame, metaclass=abc.ABCMeta):

    _metadata = ['sampling_rate', '_has_y_column', '_id']

    def __init__(self, *args, **kwargs):
        self.sampling_rate: float = kwargs.pop("sampling_rate", None)
        self._has_y_column: bool = kwargs.pop("has_y_column", False)
        self._id: str = kwargs.pop("id_", '00')
        super().__init__(*args, **kwargs)

    @property
    def sr(self) -> Optional[float]:
        if self.sampling_rate is not None:
            return self.sampling_rate
        else:
            print_msg("No sampling rate info available.", 'warn')

    @property
    def target(self) -> Optional[np.ndarray]:
        return self.y

    @property
    def label(self) -> Optional[np.ndarray]:
        return self.y

    @property
    @abc.abstractmethod
    def _constructor(self):
        return ToolDataBase

    @classmethod
    @abc.abstractmethod
    def from_file(cls, file):
        pass

    @property
    def X(self) -> np.ndarray:
        new = self
        if self._has_y_column:
            new = self.drop(columns=['label'])
        try:
            time_column = difflib.get_close_matches("time", new.columns)
            new = new.drop(columns=[time_column[0]])
        except IndexError:
            pass
        return new.values

    @property
    def y(self) -> Optional[np.ndarray]:
        if self._has_y_column:
            return self['label'].values
        else:
            print_msg("No label vector are available.", 'warn')
            return None

    @property
    def time(self) -> Optional[np.ndarray]:
        time_column = difflib.get_close_matches("time", self)
        try:
            return self[time_column[0]].values
        except IndexError:
            print_msg("No time column available.", "warn")

    @property
    def t(self) -> Optional[np.ndarray]:
        return self.time

    @property
    def ts(self) -> np.ndarray:
        new = self
        if self._has_y_column:
            new = self.drop(columns=['label'])

        # move time column to the beginning if available
        try:
            time_column = difflib.get_close_matches("time", self)[0]
            cols = list(new.columns)
            cols.remove(time_column)
            cols = [time_column] + cols
            new = self[cols]
        except IndexError:
            print_msg("No time column available.", "warn")

        return new.values

    @property
    def features(self) -> List[str]:
        new = self
        if self._has_y_column:
            new = self.drop(columns=['label'])
        try:
            time_column = difflib.get_close_matches("time", new.columns)
            new = new.drop(columns=[time_column[0]])
        except IndexError:
            print_msg("No time column", 'debug')
            pass

        return list(new.columns)

    def set_y(self, y, broadcast=False):
        if self._has_y_column:
            print_msg("Labels for data already exist ... overwriting", 'debug')
        else:
            self._has_y_column = True

        if not broadcast:
            self["label"] = np.array(y)
        else:
            self["label"] = np.full(shape=(len(self), 1), fill_value=y)

    @consistent_sampling_rate
    @consistent_time_column
    def __add__(self, other: ToolDataBase) -> ToolDataBase:
        if self._has_y_column and other._has_y_column:
            other.drop(columns='label', inplace=True)
        df = self.add(other, fill_value=0)
        df.sampling_rate = self.sampling_rate
        df._has_y_column = self._has_y_column or other._has_y_column
        df._id = self._id
        return df

    def __iadd__(self, other):
        raise NotImplementedError("Not implemented yet. Contact maintainer.")

    def __finalize__(self, other: ToolDataBase, method: Optional[str] = None, **kwargs):
        """propagate metadata from other to self """
        # merge operation: using metadata of the left object
        new = self
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            try:
                sr_ = [df.sampling_rate for df in other.objs]
                if len(set(sr_)) != 1:
                    raise SamplingRateError("Sampling rate does not match!: "
                                            f"{self.sampling_rate}Hz ≠ {other.sampling_rate}Hz")

                for name in self._metadata:
                    object.__setattr__(self, name, getattr(other.objs[0], name, None))

                new = self
                if other.objs[0]._id != other.objs[1]._id:
                    try:
                        time_column = difflib.get_close_matches("time", self.columns)
                        new = self.drop(columns=[time_column[0]])
                        print_msg("Dropping time column.", "warn")
                    except (KeyError, IndexError):
                        pass

                new.reset_index(drop=True, inplace=True)
            except IndexError:
                print_msg("Could not concatenate. Maybe your data dict contains only one measurement.", 'warn')
                return None

        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))

        return new


class DataCSV(ToolDataBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return DataCSV

    @classmethod
    def from_file(cls, file: Path) -> DataCSV:
        df_csv = pd.read_csv(file, sep=CSV_DELIMITER, index_col=None, parse_dates=False)
        df_csv.__class__ = cls

        declared_sr = round(float(file.stem.split("-")[-1]), ndigits=3)

        try:
            kind = 'mean'
            true_sr = get_sampling_rate(df_csv.time, kind=kind, ndigits=3)
            if not math.isclose(true_sr, declared_sr):
                print_msg(f"The {kind} sampling rate from the filename ({file}) "
                          f"and from the timestamps differ by {abs(true_sr - declared_sr):.3f}Hz\n", 'warn')
        except TypeError:
            # no time column available
            pass

        df_csv.sampling_rate = declared_sr
        df_csv._has_y_column = False
        df_csv._id = file.stem.split("-")[1]

        return df_csv

    @classmethod
    def from_dict(cls: DataCSV, *args, **kwargs) -> DataCSV:
        t_unit: str = kwargs.pop("t_unit", "s")
        kind: str = kwargs.pop("kind", "mean")

        df_csv = super().from_dict(*args, **kwargs)
        df_csv.__class__ = cls

        df_csv._has_y_column = False

        if "t" in df_csv:
            df_csv["time"] = df_csv["t"]
            del df_csv["t"]

            if t_unit == 's':
                factor = 1
            elif t_unit == 'ms':
                factor = 10 ** -3
            elif t_unit == 'us':
                factor = 10 ** -6
            else:
                raise ValueError("Unknown unit " + str(t_unit) + " for time!")

            df_csv["time"] *= factor

        df_csv.sampling_rate = get_sampling_rate(
            df_csv.time,
            t_unit="s",  # unit is now 's' because we converted the time beforehand
            kind=kind,
            ndigits=3
        )

        return df_csv

    def split(self) -> Optional[Dict[str, List[DataCSV]]]:
        if self._has_y_column:
            section_indices = np.where(np.diff(self.y) != 0)[0] + 1
            splits: List[DataCSV] = np.split(self, section_indices, axis=0)

            res = defaultdict(list)

            for split in splits:
                res[str(int(split['label'].values[0]))].append(split)

            return res
        else:
            return None


# -----------------------
# --- DataBunch class ---
# -----------------------

class Bunch(dict):
    """Container object for datasets

    Dictionary-like object that exposes its keys as attributes.

    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass


class DataBunch(Bunch):
    """
    Class for storing data of one single measurement.
    """

    def __init__(self,
                 acc: Optional[DataCSV] = None,
                 gyr: Optional[DataCSV] = None,
                 mag: Optional[DataCSV] = None,
                 pos: Optional[DataCSV] = None,
                 vel: Optional[DataCSV] = None,
                 mic: Optional[DataCSV] = None,
                 classes: Dict[int, str] = None
                 ):
        super(DataBunch, self).__init__(acc=acc, gyr=gyr, mag=mag, pos=pos, vel=vel, mic=mic, classes=classes)

    def __add__(self, other: DataBunch) -> DataBunch:
        for key in self.data_keys():
            self[key] = pd.concat([self[key], other[key]], sort=False, ignore_index=True)
        return self

    # -------------------------
    # --- private functions ---
    # -------------------------

    def _remove_rows_by_class(self, cls: str, data_type: str):
        data = self[data_type]
        try:
            inv_map = {v: k for k, v in self.classes.items()}
            idx = inv_map[cls]
            self[data_type] = data[data.y != idx]
        except ValueError:
            # no matching label found in self.labels
            print_msg(f"Class '{cls}' is unknown", 'warn')

    def _remove_rows_by_label(self, label: float, data_type: str):
        data = self[data_type]
        self[data_type] = data[data.y != label]

    def _remove_zero_rows(self, data_type: str):
        data = self[data_type]
        if np.any((data.X == 0).all(axis=1)):
            print_msg("Dropping zero rows.", 'debug')
            df_without_zero_rows = data.loc[~(data.X == 0).all(axis=1)]
            df_without_zero_rows.reset_index(drop=True, inplace=True)
            self[data_type] = df_without_zero_rows

    def _remove_rows_with_nan(self, data_type: str):
        data = self[data_type]
        if data._has_y_column and np.isnan(data.y).any():
            df_without_nan = data.dropna(subset=['label'], inplace=False)
            print_msg(f"Dropping {len(data) - len(df_without_nan)} samples of {len(data)} "
                      f"with nan target of data type {data_type}", 'info')
            self[data_type] = df_without_nan

    def _sanitize_target(self):
        """
        Replace all nan entries with -1 and change the data type to integer.
        """
        for data in self.values():
            if isinstance(data, DataCSV) and data._has_y_column:
                data.fillna(inplace=True, value={'label': -1})
                data.astype(copy=False, dtype={'label': int})

    def _check(self):
        """
        General consistency checks when reading data is finished.
        """

        # check if all sensor data have the same number of classes
        unique_ys = {key: np.unique(data.y) for key, data in self.items()
                     if isinstance(data, DataCSV) and data._has_y_column}
        for a, b in combinations(unique_ys, 2):
            assert unique_ys[a].tolist() == unique_ys[b].tolist(),\
                f"Miss match of number of classes between '{a}' and '{b}'. Contact maintainer ({__maintainer__})"

        # remove all empty (no data was available) members
        members_to_be_removed = [k for k, v in self.items() if v is None]

        for member in members_to_be_removed:
            del self[member]

    # ------------------------
    # --- public functions ---
    # ------------------------

    @property
    def info(self) -> List[str]:
        """
        List all available keys.
        """
        return list(self.keys())

    @classmethod
    def concat(cls, data) -> DataBunch:
        """
        Concatenate multiple data bunches to a single bunch.
        """
        keys = [set(data_bunch.keys()) for data_bunch in data.values()]
        if all(eq(*comb) for comb in combinations(keys, 2)):
            return reduce(add, data.values())
        else:
            raise ConcatError("Could not concatenate.")

    def data_keys(self) -> List[str]:
        """
        Return all keys associated with stored data.
        """
        keys = [key for key in self.keys() if isinstance(self[key], DataCSV)]
        return keys

    def remove_units_from_header(self):
        """
        Remove the units indicated by '[<unit>]' from the column names.
        """
        for data_type in self.data_keys():
            data = self[data_type]
            column_name_map = {column: re.sub(r' \[.*\]', '', column) for column in data.columns}
            self[data_type] = data.rename(column_name_map, axis='columns')

    def remove(self, classes=(), labels=(), zero=False, nan=False):
        """
        Remove samples/rows from the data. The condition for removal can be any of the following:

        * All samples which are labelled with a class from :code:`classes`
        * All samples with a label from :code:`labels`
        * All rows with only zero entries
        * All samples with a nan label

        Parameters
        ----------
        classes : list or tuple of str, optional
        labels : list or tuple of int, optional
        zero : bool, default=False
        nan : bool, default=False

        """
        for data_type in self.data_keys():

            if len(classes) > 0:
                for class_ in classes:
                    self._remove_rows_by_class(class_, data_type)

            if len(labels) > 0:
                for label in labels:
                    self._remove_rows_by_label(label, data_type)

            if zero:
                self._remove_zero_rows(data_type)

            if nan:
                self._remove_rows_with_nan(data_type)

    def finalize(self):
        """
        Carries out clean up tasks and checks.
        """
        self._sanitize_target()
        self._check()

    def split_to_actions(self) -> Dict[str, List[DataBunch]]:

        # discard not annotated data
        for data_type in self.data_keys():
            self._remove_rows_by_label(-1, data_type)

        splits = {key: data.split() for key, data in self.items() if isinstance(data, DataCSV) and data._has_y_column}

        first_key = list(splits.keys())[0]

        sorted_bunches = defaultdict(list)
        label: str  # class label of an action (e.g. id of "tightening" action)
        for label in splits[first_key].keys():

            cls = self.classes[int(label)]

            values: Dict[str, List[DataCSV]]
            for i, (data_type, values) in enumerate(splits.items()):
                for j, data_csv in enumerate(values[label]):
                    if i == 0:
                        bunch = DataBunch(classes={k: v for k, v in self.classes.items() if int(label) == k})
                        sorted_bunches[cls].append(bunch)
                    else:
                        bunch = sorted_bunches[cls][j]

                    bunch[data_type] = data_csv

        for bunch_list in sorted_bunches.values():
            for bunch in bunch_list:
                bunch.finalize()

        return sorted_bunches

    @classmethod
    def from_dict(cls, data_dict: Dict, label: float = None, **kwargs) -> DataBunch:
        df_dict = dict()

        for key, value in data_dict.items():
            df = DataCSV.from_dict(value, **kwargs)
            if label:
                df.set_y(label, broadcast=True)
            df_dict[key] = df

        bunch = cls(**df_dict)
        bunch._sanitize_target()
        bunch._check()

        return bunch
