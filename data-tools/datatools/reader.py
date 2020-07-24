# --- built-in ---
import os
import json
import tempfile
import collections
import pathlib
import difflib
import configparser
from typing import Union, List, Dict, Optional, Set
from pathlib import Path

# --- third-party ---
from dateutil.parser import parse
from dateutil.parser import ParserError
from tqdm import tqdm
import numpy as np
import pandas as pd

# --- own ---
import datatools
from .data import DataCSV, DataBunch, DATA_TYPES
from .utils import print_msg, TimeColumnError, SamplingRateError, flatten, pickle_cache
from .constants import DATA_FILE_EXTENSION, CONFIG_INI, GLOBAL_MAPPING
from .query import Query, Measurement, Action, _AtomicFilter
from .nova import DiscreteAnnotation

ANNOTATION_HEADER_FILE_EXTENSION = ".annotation"  # NOVA header file (=xml)
ANNOTATION_DATA_FILE_EXTENSION = ".annotation~"  # NOVA data file

DATA_STRUCTURE = ("start", "end", "label_id", "confidence")  # header of the NOVA annotation data file
CLASS_UNKNOWN_ID = -1  # label id which will be assigned by default if no information are available


class Reader:
    """Helper class which can read csv data from a registered local database, which is a well structured folder tree."""
    EXTENSIONS = (DATA_FILE_EXTENSION, ANNOTATION_HEADER_FILE_EXTENSION, ANNOTATION_DATA_FILE_EXTENSION)
    LABEL_MAPPING: Optional[Dict[int, str]] = None
    # ------------------------
    # --- public functions ---
    # ------------------------

    @classmethod
    def read_measurements(cls,
                          measurement_series: Dict[pathlib.Path, List[str]],
                          data_types: List[str] = DATA_TYPES,
                          ) -> Union[DataBunch, Dict[str, DataBunch]]:
        """
        All measurements of the selected measurement series are loaded. This includes the data and
        the corresponding labels (if available). If no specific desired data type is passed, all sensor data is loaded.
        """

        ms = measurement_series.keys()
        data_ms = dict()  # data of all selected measurement series

        for ms_id, ms_path in enumerate(ms):

            # data of all measurements from one measurement series
            data_mea = collections.defaultdict(DataBunch)

            # --------------------------------
            # --- read data from csv files ---
            # --------------------------------

            possible_data_files = list(ms_path.glob('*.csv'))

            pbar = tqdm(total=len(possible_data_files))
            pbar.set_description("[INFO] Read data")

            for mea_path in possible_data_files:
                pbar.set_postfix(file=mea_path.name, refresh=False)

                if mea_path.is_file() and (mea_path.suffix in cls.EXTENSIONS):
                    dt, id_, sr = cls._get_meta(mea_path)

                    if id_ not in measurement_series[ms_path]:
                        pbar.update()
                        continue

                else:
                    print_msg(f"Skipping surplus csv file {mea_path}.", 'info')
                    continue

                try:
                    if dt in data_types:
                        df = DataCSV.from_file(mea_path)

                        # create new dataframe if none present for this data type
                        if id_ not in data_mea or data_mea[id_][dt] is None:
                            data_mea[id_][dt] = df

                        # add sensor data to existing dataframe
                        else:
                            data_mea[id_][dt] = data_mea[id_][dt] + df
                    else:
                        print_msg(f"Skipping {dt} data (not selected).", 'debug')
                        continue

                except (TimeColumnError, SamplingRateError) as e:
                    print_msg(f"\n{e}", 'error')
                    print_msg(f"Unable to load {mea_path.name}. Skipping ...", 'warn')
                    continue
                except ValueError as e:
                    print_msg(f"\n{e}", 'debug')
                    print_msg(f"Unable to load {mea_path.name}. Skipping ...", 'error')
                    continue
                finally:
                    pbar.update()

            # ----------------------------------------
            # --- add labels from annotation files ---
            # ----------------------------------------

            pbar.set_description("[INFO] Read annotation")
            for id_ in data_mea.keys():

                if (ms_path / f'data-{id_}.annotation').exists():

                    annotation_file_path = ms_path / f'data-{id_}.annotation'
                    pbar.set_postfix(file=annotation_file_path.name, refresh=True)

                    anno = DiscreteAnnotation.from_files(annotation_file_path).update_label_mapping(cls.LABEL_MAPPING)
                    annotation_df = anno.data.rename(columns=dict(zip(anno._columns, DATA_STRUCTURE)))

                    # set labels for all data
                    for dt in data_mea[id_].data_keys():
                        data_mea[id_][dt].set_y(cls._get_y_from_annotation(
                            num_samples=len(data_mea[id_][dt]),
                            annotation_df=annotation_df.copy(),
                            sampling_rate=data_mea[id_][dt].sampling_rate))

                    data_mea[id_].classes = {item.label: item.class_name for item in anno.items}

                else:
                    print_msg(f"No annoation file available for {id_}", 'warn')

            for bunch in data_mea.values():
                bunch.finalize()

            data_ms[ms_path.name] = data_mea
            pbar.close()

        # --- sanitize and finalize data ---

        if len(data_ms) > 1:
            flattened_data = flatten(data_ms, stop_cls=DataBunch)
            return flattened_data
        else:
            return data_ms[list(data_ms.keys())[0]]

    # -------------------------
    # --- private functions ---
    # -------------------------

    @staticmethod
    def _get_meta(mea_path):
        dt, id_, sr = mea_path.stem.split('-')
        # sanitize id (leading zero)
        if len(id_) == 1:
            try:
                id_ = f"{int(id_):02d}"
            except Exception as e:
                raise ValueError(f"Measurement id {id_} is incorrect. ({e})")
        elif len(id_) != 2:
            raise ValueError("Measurement id {id_} is incorrect.")
        else:
            pass

        return dt.lower(), id_, sr

    @staticmethod
    def _get_y_from_annotation(num_samples, annotation_df, sampling_rate):
        """
        Evaluates the given annotation data to get the corresponding label vector. Each segment from the annotation file
        is described by a start and end time and a (class) id. Now we need a sample based label vector instead.

        Parameters
        ----------
        num_samples : int
            Duration of the annotated data in number of samples.
        annotation_df : pd.DataFrame
            Annotation data.
        sampling_rate : float
            Sampling rate of the annotated data in Hz.

        Returns
        -------
        y : np.ndarray
            Label vector with multi-class labels.
        """

        # conversion from time to sample number
        for column in ['start', 'end']:
            annotation_df[column] = annotation_df[column].apply(lambda z: round(z * sampling_rate))

        assert annotation_df.dtypes["start"] == np.int64()
        assert num_samples >= annotation_df["end"].values[-1], (f"Target vector can not be evaluated from the NOVA file"
                                                                f" because the number of samples {num_samples}"
                                                                f" is smaller than the max annotation index"
                                                                f" {annotation_df['end'].values[-1]}. "
                                                                f"This might be because of wrong sampling rates"
                                                                f" (sr={sampling_rate})")

        y = np.ones(num_samples, dtype=float) * CLASS_UNKNOWN_ID
        label_ids = annotation_df['label_id'].tolist()

        for i, (start, end) in enumerate(zip(annotation_df['start'].tolist(), annotation_df['end'].tolist())):
            y[start:end] = np.nan if (label_ids[i] == -1) else label_ids[i]

        return y


class MeasurementDataReader:
    """
    Class for loading and preparing data files and corresponding labels from a fixed directory structure
    with predefined naming conventions. Therefore it can parse annotation files created with NOVA.

    Parameters
    ----------
    source : str or pathlib.Path, (default=None)
        Registered source OR path to the root directory of a data source.
        In not set the registered default source will be used!
    use_global_mapping
        Enables usage of global label mapping.
    verbose : integer, optional
        Controls the verbosity: the higher, the more messages.

    Attributes
    ---------
    _path_to_root_dir : pathlib.Path
        Path to the root directory of the measurement data.
    _tool_dict : dict
        Measurement series accessible by the corresponding tools' names.
    _info_df : pandas.DataFrame
        DataFrame containing information about all measurements within the root directory.
    _cache_folder : pathlib.Path
        Temporal location for caching files.
    _caching : bool, default=True
        If True, caching for the read function is enabled.
    """

    def __init__(self, source: Union[str, Path, None] = None, use_global_mapping: bool = True, verbose: int = 2):

        self._use_global_mapping = use_global_mapping
        self._caching = False

        try:
            # Load config file
            config = configparser.ConfigParser()
            config.read(CONFIG_INI)
        except Exception as e:
            raise RuntimeError("Could not load config. "
                               f"Contact the maintainer {datatools.__maintainer__} and send: {e}")

        if source is None:
            # try to load default source
            try:
                default_key = config["Sources"]["default"]
                self._path_to_root_dir = Path(config["Sources"][default_key])
            except KeyError:
                raise ValueError("Register a source and make it default!")

        elif source is not None and not Path(source).exists():
            self._path_to_root_dir = Path(config["Sources"][source])
            try:
                self._path_to_root_dir = Path(config["Sources"][source])
            except Exception as e:
                print_msg(f"Could not load config. No source with key {source}: {e}", 'error')

        elif Path(source).exists():
            self._path_to_root_dir = Path(source)
        else:
            raise ValueError(f"Invalid source '{source}'. Pass an existing path or an preregistered source.")

        print_msg.set_verbosity(verbose)

        if not self._path_to_root_dir.exists():
            raise ValueError(f"Provided path ({self._path_to_root_dir}) does not exist!")

        # read all available_tools
        tools = [p for p in self._path_to_root_dir.iterdir() if self._is_tool_dir(p) and
                 all(s not in p.name for s in [".", "tests", "info", "data_tools", "report"])]

        self._tool_dict = {p.name: {m.name: m for m in p.iterdir() if m.is_dir() and self._is_ms_name(m.name)}
                           for p in tools}

        self._info_df = self._fit_global_info_df()

        self._cache_folder = pathlib.Path(tempfile.gettempdir())

        self._reader = Reader()

        if self._use_global_mapping:
            Reader.LABEL_MAPPING = GLOBAL_MAPPING

    def __del__(self):
        if self._caching:
            for file in self._cache_folder.glob('*_mdr_cache.pkl'):
                try:
                    os.remove(str(file))
                except OSError:
                    pass

            print_msg(f"Removed cached files from {self._cache_folder}", 'info')

    @property
    def label_mapping(self) -> Dict[int, str]:
        return GLOBAL_MAPPING

    # -------------------------
    # --- private functions ---
    # -------------------------
    def _is_tool_dir(self, path: pathlib.Path) -> bool:
        """
        Check whether a given path is a tool directory.
        """
        if path.is_dir() and any(self._is_ms_name(ms_path.name) for ms_path in path.iterdir()):
            return True
        else:
            return False

    @staticmethod
    def _is_ms_name(s: str) -> bool:
        """
        Check whether a given string satisfies the naming convention for a measurement series, i.e.
        <lab/location>-<mc_id>-<yyyymmdd>, with mc_id being a 2-digit number.
        """
        parts = s.split('-')
        if len(parts) < 3:
            return False
        else:
            ms_date = parts[-1]
            ms_id = parts[-2]
            if ms_date.isdigit() and len(ms_date) == 8 and ms_id.isdigit() and len(ms_id) == 2:
                return True
            else:
                return False

    def _fit_global_info_df(self) -> pd.DataFrame:
        """
        Fit global DataFrame containing information about all measurements within the source directory.
        """
        return pd.concat([self._fit_tool_info_df(tool) for tool in self._tool_dict.keys()])

    def _fit_tool_info_df(self, tool: str) -> pd.DataFrame:
        """
        Fit DataFrame containing information about all measurements of the specified tool.
        """
        info_dict = collections.defaultdict(list)

        for ms_path in self._tool_dict[tool].values():

            # get information from "info" file
            try:
                ms_info_dict = self._load_info_file(ms_path)
            except FileNotFoundError:
                print_msg(f"Could not read info file for {tool}", 'warn')
                continue

            # figure out which data types are available
            available_data_types = self._get_avaliable_data_types(ms_path)

            # read available labels/classes for each measurement from annotation file
            available_labels_and_classes = self._get_available_labels_and_classes(ms_path)

            _, mc_id, _ = ms_path.name.split("-")

            for i, measurement in enumerate(ms_info_dict['measurements']):
                # add general information of measurement series
                info_dict['measurement_series_path'].append(ms_path)
                info_dict['measurement_series_name'].append(ms_path.name)
                info_dict['measurement_campaign_id'].append(mc_id)
                info_dict['date'].append(ms_info_dict['date'])
                if ms_info_dict['tool'] == tool:
                    info_dict['tool'].append(tool)
                else:
                    raise ValueError(f'Tool name in info.json file ({ms_info_dict["tool"]}) does not match tool'
                                     f'name of directory ({tool})')
                info_dict['model'].append(ms_info_dict['model'])
                info_dict['work_piece'].append(ms_info_dict['work_piece'])
                # add measurement specific information
                info_dict['measurement_id'].append(measurement['id'])
                info_dict['test_person'].append(measurement['test_person'])
                info_dict['time'].append(measurement['time'])
                info_dict['module'].append(measurement['module'])
                info_dict['model_id'].append(measurement['model_id'])
                info_dict['data_type'].append(available_data_types)
                info_dict['labels'].append(available_labels_and_classes[measurement['id']]["labels"])
                info_dict['classes'].append(available_labels_and_classes[measurement['id']]["classes"])
                # add tool and measurement specific information
                for key in measurement['tool_settings']:
                    info_dict[key].append(measurement['tool_settings'][key])

        return pd.DataFrame(info_dict)

    @staticmethod
    def _load_info_file(path: pathlib.Path) -> dict:
        """
        Load info.json of corresponding measurement series.
        """
        info_file_path = path / 'info.json'
        with open(str(info_file_path)) as json_file:
            info_dict = json.load(json_file)

        for idx, measurement_dict in enumerate(info_dict['measurements']):
            if int(measurement_dict['id']) != idx + 1:
                raise ValueError(f'Corrupt info.json file found for measurement series {path.name}')

        # try to convert the date info from string to pandas' timestamp representation
        try:
            info_dict["date"] = pd.Timestamp(parse(info_dict["date"], ignoretz=True))
        except ParserError:
            print_msg(f"Could not parse date of '{path.name}'. Skip conversion ...", 'warn')

        return info_dict

    @staticmethod
    def _get_avaliable_data_types(measurement_path: pathlib.Path) -> Set[str]:
        return {csv_file.stem.split("-")[0].lower()
                for csv_file in measurement_path.glob("*.csv") if csv_file.name.split("-")[0].lower() in DATA_TYPES}

    def _get_available_labels_and_classes(self, measurement_path: pathlib.Path) -> Dict[str, Dict[str, Set]]:
        res = collections.defaultdict(lambda: collections.defaultdict(set))
        for annotation_file in measurement_path.glob(f"*{ANNOTATION_HEADER_FILE_EXTENSION}"):
            global_mapping = GLOBAL_MAPPING if self._use_global_mapping else None
            for item in DiscreteAnnotation.from_files(annotation_file).update_label_mapping(global_mapping).items:
                mea_id: str = annotation_file.stem.split("-")[1]
                res[mea_id]["classes"].add(item.class_name)
                res[mea_id]["labels"].add(item.label)

        return res

    @staticmethod
    def _parse_type_selection(data_types: List[str]) -> List[str]:
        """Checks and parses the desired data types passed by the user."""
        if len(data_types) == 0:
            selected_types = DATA_TYPES
        else:
            selected_types = [data_type.lower() for data_type in data_types
                              if data_type.lower() in DATA_TYPES]

        return selected_types

    def _parse_measurement_series_selection(
            self, measurement_series: List[str], tools: List[str]) -> List[pathlib.Path]:
        """Tries to find matching measurement series based on the users input (measurement_series, tools)."""

        measurement_series_dict = collections.ChainMap(
            *[ms for tool_name, ms in self._tool_dict.items()])  # `ms` means measurement_series

        if not measurement_series and not tools:
            selected_ms = measurement_series_dict.values()
        else:
            try:
                tools = [difflib.get_close_matches(
                    tool, self._tool_dict.keys(), cutoff=0.1, n=1)[0] for tool in tools]

                measurement_series = [difflib.get_close_matches(
                    meas, measurement_series_dict.keys(), cutoff=0.1, n=1)[0] for meas in measurement_series]

            except IndexError:
                print_msg(f"No matching tool or measurement series found in {self._path_to_root_dir}"
                          f" based on {measurement_series}", 'warn')
                return list()

            selected_ms_by_tools = set()
            selected_ms_by_name = set()

            if len(tools) > 0:
                for tool in tools:
                    selected_ms_by_tools.update({*list(self._tool_dict[tool].values())})

            if len(measurement_series) > 0:
                for tool, measurement in self._tool_dict.items():
                    selected_ms_by_name.update(
                        {path for name, path in measurement.items() if name in measurement_series})

            if len(tools) > 0 and len(measurement_series) > 0:
                selected_ms = selected_ms_by_tools & selected_ms_by_name
            elif len(tools) > 0 and len(measurement_series) == 0:
                selected_ms = selected_ms_by_tools
            else:
                selected_ms = selected_ms_by_name

        return list(selected_ms)

    # ------------------------
    # --- public functions ---
    # ------------------------

    @pickle_cache()
    def read(self,
             measurement_series: List[str] = (),
             tools: List[str] = (),
             data_types: List[str] = (),
             measurement_campaigns: List[int] = (),
             remove_garbage: bool = True,
             remove_not_annotated: bool = False,
             concatenated: bool = False,
             columns_without_units: bool = False,
             exclude_classes: List[str] = (),
             exclude_labels: List[str] = (),
             drop_zero_rows: bool = False,
             caching: bool = False,
             **kwargs
             ):
        """
        Reads and prepares measurement data from the measurement-data repository
        with convenient generation of datasets for machine learning in mind.

        Parameters
        ----------
        measurement_series
            Names (not a paths) of measurement series which should be processed.
            Passed names can be incomplete but should be unambiguous,
            e.g. "bmw" :math:`\\rightarrow` "factory_bmw_regensburg-00-20170830"
            If `tools` is also passed, the cut set is loaded.
        tools
            Names of the hand tools for which all measurements should be processed.
            Passed names can be incomplete but should be unambiguous,
            e.g. "elec" :math:`\\rightarrow` "electric_screwdriver"
            If `measurement_series` is also passed, the cut set is loaded.
        data_types
            Abbreviation for data types which should be incorporated. Can be 'IMU', 'MF' or 'AUDIO'.
        measurement_campaigns
            Measurement campaign id for which all measurements should be processed.
        remove_garbage
            If True, all samples labelled as garbage are removed (i.e. -1 in :code:`.annotation~` file).
        remove_not_annotated
            If True, all samples which are not labelled are removed.
            (i.e. no corresponding label in the :code:`.annotation~` file)
        concatenated
            If True, all measurements will be concatenated (if matching sampling rates)
            to a single pd.DataFrame-like object.
        columns_without_units
            If True, all units will be removed from the column names, e.g. "time [s]" :math:`\\rightarrow` "time"
        exclude_classes
            All samples which are labelled with these classes (str) will be discarded.
        exclude_labels
            All samples which are labelled with these label (int) will be discarded.
        drop_zero_rows
            All samples where all feature values are zero will be removed.
        caching
            Enables caching of result (as pickle files).

        Returns
        -------
        data : dict or DataBunch
            If there is only one measurement (data set), a DataBunch is returned instead of a dict. Each measurement
            can be accessed by its id as the key.
        """

        self._caching = caching

        # ------------------------------------
        # --- some user-friendly shortcuts ---
        # ------------------------------------

        if "ms" in kwargs and len(measurement_series) == 0:
            measurement_series = kwargs.pop("ms", ())

        if "mc" in kwargs and len(measurement_campaigns) == 0:
            measurement_campaigns = kwargs.pop("mc", ())

        if "dt" in kwargs and len(data_types) == 0:
            data_types = kwargs.pop("dt", ())

        if len(measurement_campaigns) != 0:
            raise NotImplementedError(f"This feature is under development."
                                      f" Contact the maintainer {datatools.__maintainer__}")

        if len(kwargs) > 0:
            raise ValueError(f"Wrong keywords where used {kwargs}. Check for typos.")

        # -----------------
        # --- read data ---
        # -----------------

        measurement_series = self._parse_measurement_series_selection(measurement_series, tools)
        if len(measurement_series) == 0:
            raise ValueError("No measurement series found")

        dts = self._parse_type_selection(data_types)

        print_msg("Preparing measurements from:", 'info')
        for ms in measurement_series:
            print_msg(str(ms), 'indent')

        df = self._info_df[self._info_df["measurement_series_path"].isin(measurement_series)]
        mapping = {k: v.tolist() for k, v in df.groupby("measurement_series_path")["measurement_id"]}

        data = self._reader.read_measurements(measurement_series=mapping, data_types=dts)

        # ----------------------
        # --- postprocessing ---
        # ----------------------
        for data_bunch in data.values():

            if remove_not_annotated is True:
                exclude_labels = list(exclude_labels) + [-1]

            data_bunch.remove(classes=exclude_classes, labels=exclude_labels, zero=drop_zero_rows, nan=remove_garbage)

            if columns_without_units:
                data_bunch.remove_units_from_header()

            # final consistency check, etc.
            data_bunch.finalize()

        print_msg(f"Finished with {len(data)} measurement(s).", 'info')

        if concatenated:
            try:
                print_msg("Concatenating data.", 'info')
                data = DataBunch.concat(data)
            except SamplingRateError:
                print_msg("Concatenating failed due to incompatible sampling rates.", 'warn')

        if len(data) <= 1:
            return list(data.values())[0]
        else:
            return data

    def query(self, query_type: Union[_AtomicFilter] = Measurement) -> Optional[Query]:

        if query_type is not Measurement and query_type is not Action:
            print_msg(f"{query_type.__name__} is not supported", 'error')
            return None

        return Query(self._info_df, self._reader, query_type)
