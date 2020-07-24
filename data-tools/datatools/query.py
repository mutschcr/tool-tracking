# --- build-in ---
from __future__ import annotations
from typing import Callable, Any, Optional, Union, List, Dict, Set
from collections import defaultdict
from operator import and_, or_, methodcaller, lt, le, eq, ne, ge, gt
from functools import reduce
from itertools import chain
from copy import copy

# --- third-party ---
from dateutil.parser import parse
import pandas as pd
import numpy as np

# --- own ---
from .data import DataType, DataBunch
from .utils import print_msg, count_objects


class _AtomicFilter:
    """
    A helper class which creates filter mask functions for dataframe columns,
    which contains only atomic (indivisible) values. All basic comparison operators are available for filtering.

    Parameters
    ----------
    mapping
        Mapping from short form (keys) to dataframe columns (values). E.g.: {"id": "tool_id"}.

    Attributes
    ----------
    _default_key
        Key to be used when applying a comparision operator before setting a corresponding key.
    _corresponding_key
        Key to be used when applying a comparision operator.
    _mapping
        See Parameters/mapping.

    Examples
    --------
    >>> df = pd.DataFrame({"name": ["foo", "bar", "bar"], "tree_size": [2, 4, 8]})
    >>> mapping = {"name": "name", "size": "tree_size", "DEFAULT": "name"}
    >>> filt = _AtomicFilter(mapping=mapping)

    No we get a filter mask function which filters a dataframe based on the column "name"

    >>> filt_func = filt == "bar"
    >>> mask = filt_func(df)
    >>> filtered_df = df[mask]
    >>> filtered_df
      name  tree_size
    1  bar          4
    2  bar          8

    We can also explicitly name what we want to filter

    >>> filt_func = filt.size < 5
    >>> df[filt_func(df)]
      name  tree_size
    0  foo          2
    1  bar          4
    """
    def __init__(self, mapping: Optional[dict] = None):
        default = mapping.pop("DEFAULT", None)
        self._default_key: str = default
        self._corresponding_key: str = default
        self._mapping = mapping

    def __getattr__(self, item):
        self._corresponding_key = self._mapping.get(item)
        return self

    def __contains__(self, item):
        return item in self._mapping

    def _preprocess(self, other):
        key = copy(self._default_key) if self._corresponding_key is None else copy(self._corresponding_key)
        self._corresponding_key = self._default_key

        if self._corresponding_key == "date":
            if isinstance(other, (list, tuple)):
                other = [pd.Timestamp(parse(o)) for o in other]
            else:
                other = pd.Timestamp(parse(other))

        return other, key

    def _create_mask_func(self, operator: Callable, other) -> Callable:
        other, key = self._preprocess(other)
        if isinstance(other, (list, tuple)):
            return lambda df: reduce(or_, [operator(df[key], o) for o in other])
        else:
            return lambda df: operator(df[key], other)

    def __lt__(self, other: Any) -> Callable:
        return self._create_mask_func(lt, other)

    def __le__(self, other: Any) -> Callable:
        return self._create_mask_func(le, other)

    def __eq__(self, other: Any) -> Callable:
        return self._create_mask_func(eq, other)

    def __ne__(self, other: Any) -> Callable:
        return self._create_mask_func(ne, other)

    def __ge__(self, other: Any) -> Callable:
        return self._create_mask_func(ge, other)

    def __gt__(self, other: Any) -> Callable:
        return self._create_mask_func(gt, other)


class _NonAtomicFilter:
    def __lt__(self, other):
        raise NotImplementedError(f"'lt' operator not implemented for {self.__name__}")

    def __le__(self, other):
        raise NotImplementedError(f"'le' operator not implemented for {self.__name__}")

    def __ge__(self, other):
        raise NotImplementedError(f"'ge' operator not implemented for {self.__name__}")

    def __gt__(self, other):
        raise NotImplementedError(f"'gt' operator not implemented for {self.__name__}")

    def __ne__(self, other):
        raise NotImplementedError(f"'ne' operator not implemented for {self.__name__}")


class _DataTypes(_NonAtomicFilter):
    def __eq__(self, other: Union[List[DataType], DataType]) -> Callable:
        if isinstance(other, DataType):
            other = {other, }

        def evaluate(df: pd.DataFrame):
            data_types = [set(DataType(type_) for type_ in types)
                          for types in df["data_type"].values.tolist()]
            mask = [data_type >= set(other) for data_type in data_types]
            df.loc[mask, "data_type"] = [{data_type.value for data_type in other} for _ in range(sum(mask))]
            return mask

        return evaluate


class _Action(_NonAtomicFilter):
    @staticmethod
    def _convert_to_set(other) -> Set:
        return set(other) if isinstance(other, list) else {other, }

    def __eq__(self, other: Union[List[str], str]) -> Callable:
        other = self._convert_to_set(other)

        def evaluate(df: pd.DataFrame):
            mask = df["classes"] >= other
            df.loc[mask, "classes"] = [other for _ in range(sum(mask))]
            return mask

        return evaluate

    def __ne__(self, other: Union[List[str], str]) -> Callable:
        other = self._convert_to_set(other)

        def evaluate(df: pd.DataFrame):
            selection = np.empty(shape=(len(df), ), dtype=object)
            mask = np.full_like(selection, False, dtype=bool)
            for index, row in df.iterrows():
                selection[index] = row["classes"] - other
                mask[index] = len(selection[index]) > 0
            df.loc[mask, "classes"] = selection[mask]
            return mask

        return evaluate


DataTypes = _DataTypes()
DataTypes.__name__ = "DataTypes"

Action = _Action()
Action.__name__ = "Action"

Tool = _AtomicFilter({
    "name": "tool",
    "id": "tool_id",
    "model": "model",
    "model_id": "model_id",
    "DEFAULT": "tool"
})
Tool.__name__ = "Tool"

Config = _AtomicFilter({
    "rpm_cw": "rpm_cw",
    "rpm_ccw": "rpm_ccw",
    "torque": "torque",
    "air_pressure": "air_pressure"
})
Config.__name__ = "Config"

MeasurementSeries = _AtomicFilter({
    "name": "measurement_series_name",
    "date": "date",
    "module": "module",
    "campaign_id": "measurement_campaign_id",
    "DEFAULT": "measurement_series_name"
})
MeasurementSeries.__name__ = "MeasurementSeries"

Measurement = _AtomicFilter({
    "id": "measurement_id",
    "person": "test_person",
    "work_piece": "work_piece",
    "DEFAULT": "measurement_id"
})
Measurement.__name__ = "Measurement"


class Query:
    """
    Class which represents a query for measurement data. The available data is indexed and described by a dataframe.
    The query results represents the actual data and can be read by a reader object,
    which knows how and where to read the data

    Parameters
    ----------
    df
        Dataframe to be queried
    reader
        Reader object implementing a read_measurement function. Knows how and where to read the data from.
    query_type: {Action, Measurement}
        Defines the type of query which will be performed.

    Examples
    --------
    >>> df = pd.DataFrame({"tool": ["foo", "bar", "bar"], "tool_id": ["01", "01", "02"]})
    >>> q = Query(df, None, Measurement)
    >>> _ = q.filter_by(Tool.name == "bar", Tool.id == "01")

    Now we can have a look on the query results with

    >>> q.df
      tool tool_id
    0  bar      01
    """
    def __init__(self, df: pd.DataFrame, reader, query_type: _AtomicFilter):
        self._df = df
        self._reader = reader
        self._type = query_type

        self._filtered_df: pd.DataFrame = pd.DataFrame()

    # --- private ---

    @staticmethod
    def _split_into_action(res) -> Dict[str, List[DataBunch]]:
        actions = defaultdict(list)

        bunches = []
        if isinstance(res, dict):
            for id_, bunch in res.items():
                bunches.append(bunch.split_to_actions())
        else:
            bunches.append(res.split_to_actions())

        dict_items = map(methodcaller('items'), bunches)
        for k, v in chain.from_iterable(dict_items):
            actions[k].extend(v)

        return actions

    # --- public ---

    @property
    def df(self):
        """Return the current query result as a dataframe"""
        return self._filtered_df.reset_index(drop=True)

    def _read_data(self) -> Optional[DataBunch, Dict[str, DataBunch], Dict[str, List[DataBunch]]]:
        """Read data based on this query"""
        df = self._filtered_df.reset_index()

        if df.empty:
            return None

        mapping = {k: v.tolist() for k, v in df.groupby("measurement_series_path")["measurement_id"]}

        if not isinstance(df["data_type"].values.tolist()[0], set):
            data_types = pd.unique(df["data_type"])
        else:
            data_types = reduce(or_, df["data_type"].values.tolist())

        print_msg("Preparing data from:", 'info')
        for ms in df["measurement_series_path"]:
            print_msg(str(ms), 'indent')

        res = self._reader.read_measurements(
            measurement_series=mapping,
            data_types=data_types
        )

        if self._type is Action:  # because Action is not a class, we need "is" instead of "isinstance()"

            if not isinstance(df["classes"].values.tolist()[0], set):
                selected_actions = set(pd.unique(df["classes"]).tolist())
            else:
                selected_actions = reduce(or_, df["classes"].values.tolist())

            res = {key: value for key, value in self._split_into_action(res).items() if key in selected_actions}

            num_actions = count_objects(res, DataBunch)
            print_msg(f"Finished with {num_actions} action(s).", 'info')
        else:
            print_msg(f"Finished with {len(res)} measurement(s).", 'info')

        return res

    def get(self,
            remove_garbage: bool = True,
            remove_not_annotated: bool = False,
            columns_without_units: bool = False,
            drop_zero_rows: bool = False) -> Optional[DataBunch, Dict[str, DataBunch], Dict[str, List[DataBunch]]]:
        """
        Return the results represented by this Query using the Reader object.

        Parameters
        ----------
        remove_garbage
            If True, all samples labelled as garbage are removed (i.e. -1 in :code:`.annotation~` file).
        remove_not_annotated
            If True, all samples which are not labelled are removed.
            (i.e. no corresponding label in the :code:`.annotation~` file)
        columns_without_units
            If True, all units will be removed from the column names, e.g. "time [s]" :math:`\\rightarrow` "time"
        drop_zero_rows
            All samples where all feature values are zero will be removed.
        """

        # -----------------
        # --- read data ---
        # -----------------
        data = self._read_data()

        # ----------------------
        # --- postprocessing ---
        # ----------------------
        def postprocess(bunch: DataBunch):
            bunch.remove(labels=[-1] if remove_not_annotated else [], zero=drop_zero_rows, nan=remove_garbage)

            if columns_without_units:
                bunch.remove_units_from_header()

            # final consistency check, etc.
            bunch.finalize()

        for d in data.values():
            if isinstance(d, list):
                for bunch in d:
                    postprocess(bunch)
            else:
                postprocess(d)

        return data

    def count(self) -> int:
        """Return a count of DataBunches this Query would return."""
        return self._filtered_df.shape[0]

    def filter_by(self, *args: Callable, join: str = "and"):
        """
        Add the given filtering criterion to this Query. Multiple criteria may be specified as comma separated;
        the effect is that they will be joined together using conjunction.
        Another logical operator maybe specified using the "join" argument.
        """
        join_func: Callable
        if join == "and":
            join_func = and_
        elif join == "or":
            join_func = or_
        else:
            raise ValueError(f"Unknown join func '{join}'")

        try:
            df = self._df.copy()
            self._filtered_df = df[reduce(join_func, [arg(df) for arg in args])]
        except KeyError as err:
            print_msg(f"Could not query for {err}", 'warn')
            raise err

        return self
