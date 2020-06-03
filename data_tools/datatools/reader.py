# -*- coding: utf-8 -*-
# Built-in/Generic Imports
import os
import tempfile
import collections
import pathlib
import difflib
import xml.etree.ElementTree

# Libs
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm as ProgressBar
except ImportError:
    from .utils import ProgressBar

# Own
import datatools
from .data import DataCSV, DataBunch
from .utils import print_msg, TimeColumnError, SamplingRateError, flatten, pickle_cache
from .constants import *


ANNOTATION_HEADER_FILE_EXTENSION = ".annotation"  # NOVA header file (=xml)
ANNOTATION_DATA_FILE_EXTENSION = ".annotation~"   # NOVA data file

DATA_FOLDER_STR = "measurements"

DATA_TYPES = ['imu', 'acc', 'mf', 'gyr', 'pos', 'vel', 'audio']

DATA_STRUCTURE = ("start", "end", "label_id", "confidence")  # header of the NOVA annotation data file
CLASS_UNKNOWN_ID = -1  # label id which will be assigned by default if no information are available


class MeasurementDataReader:
    """
    Class for loading and preparing data files and corresponding labels from a fixed directory structure
    with predefined naming conventions. Therefore it can parse annotation files created with NOVA.

    Parameters
    ----------
    path_to_root_dir : str, (default=None)
        Path to the root directory of the measurement data (relative or absolute).
        In case of the default value `../` will be used.

    verbose : integer, optional
        Controls the verbosity: the higher, the more messages.

    Attributes
    ---------
    _path_to_root_dir : pathlib.Path
        Path to the root directory of the measurement data.
    _tool_dict : dict
        Measurement series accessible by the corresponding tools' names
    _cache_folder : pathlib.Path
        Temporal location for caching files.
    _caching : bool, default=True
        If True, caching for the read function is enabled.
    """
    _VALID_EXTENSIONS = [DATA_FILE_EXTENSION,
                         ANNOTATION_HEADER_FILE_EXTENSION,
                         ANNOTATION_DATA_FILE_EXTENSION,
                         ]

    def __init__(self, path_to_root_dir=None, verbose=2):

        if path_to_root_dir is None:
            self._path_to_root_dir = pathlib.Path(os.path.dirname(__file__)).parent.parent
        else:
            self._path_to_root_dir = pathlib.Path(path_to_root_dir)

        print_msg.set_verbosity(verbose)

        if not self._path_to_root_dir.exists():
            raise ValueError(f"Provided path ({self._path_to_root_dir}) does not exist!")

        # read all available_tools
        tools = [p for p in self._path_to_root_dir.iterdir() if p.is_dir() and
                 all(s not in p.name for s in [".", "tests", "info", "data_tools"])]

        self._tool_dict = {p.name: {m.name: m for m in p.iterdir() if m.is_dir() and "." not in m.name}
                           for p in tools}

        self._cache_folder = pathlib.Path(tempfile.gettempdir())

        self._caching = False

    def __del__(self):
        if self._caching:
            for file in self._cache_folder.glob('*_mdr_cache.pkl'):
                try:
                    os.remove(str(file))
                except OSError:
                    pass

            print_msg(f"Removed cached files from {self._cache_folder}", 'info')

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

    def _parse_measurement_series_selection(self, measurement_series, tools):
        """
        Tries to find matching measurement series based on the users input (measurement_series, tools).

        Parameters
        ----------
        measurement_series : list of str
        tools : list of str

        Returns
        -------
        selected_measurement_series : list of pathlib.Path

        """

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

    def _read_measurements(self, measurement_series, data_types):
        """
        All measurements of the selected measurement series are loaded. This includes the data and
        the corresponding labels (if available). If no specific desired data type is passed, all sensor data is loaded.

        Parameters
        ----------
        measurement_series : list of pathlib.Path
        data_types : list of str

        Returns
        -------
        data : collections.defaultdict(DataBunch)

        """

        data_ms = dict()  # data of all selected measurement series
        data_mea = None   # data of all measurements from one measurement series

        for ms_id, ms_path in enumerate(measurement_series):

            data_mea = collections.defaultdict(DataBunch)

            # --------------------------------
            # --- read data from csv files ---
            # --------------------------------

            possible_data_files = list((ms_path / DATA_FOLDER_STR).glob('*.csv'))

            pbar = ProgressBar(total=len(possible_data_files))
            pbar.set_description("[INFO] Read data")

            for mea_path in possible_data_files:
                pbar.set_postfix(file=mea_path.name, refresh=False)

                if mea_path.is_file() and (mea_path.suffix in self._VALID_EXTENSIONS):
                    dt, id_, sr = self._get_meta(mea_path)
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
                        print_msg(f"Skipping {dt} data (not selected).", 'info')
                        continue

                except (TimeColumnError, SamplingRateError) as e:
                    print("\n")
                    print_msg(str(e), 'error')
                    print_msg(f"Unable to load {mea_path.name}. Skipping ...", 'warn')
                    continue
                except ValueError as e:
                    print("\n")
                    print_msg(str(e), 'debug')
                    print_msg(f"Unable to load {mea_path.name}. Skipping ...", 'error')
                    continue
                finally:
                    pbar.update()

            # ----------------------------------------
            # --- add labels from annotation files ---
            # ----------------------------------------

            pbar.set_description("[INFO] Read annotation")
            for id_ in data_mea.keys():

                if (ms_path / DATA_FOLDER_STR / f'data-{id_}.annotation').exists():

                    annotation_file_path = ms_path / DATA_FOLDER_STR / f'data-{id_}.annotation'
                    pbar.set_postfix(file=annotation_file_path.name, refresh=True)

                    annotation_df, classes, _ = self._load_discrete_annotation(annotation_file_path)

                    # set labels for all data
                    for dt in data_mea[id_].data_keys():
                        data_mea[id_][dt].set_y(self._get_y_from_annotation(
                            num_samples=len(data_mea[id_][dt]),
                            annotation_df=annotation_df.copy(),
                            sampling_rate=data_mea[id_][dt].sampling_rate))

                    data_mea[id_].classes = classes

                # exceptional handling of "old" label files
                else:
                    label_file_path = ms_path / DATA_FOLDER_STR / f'data-{id_}.label'
                    pbar.set_postfix(file=label_file_path.name, refresh=False)

                    data_mea[id_].imu.set_y(np.loadtxt(str(label_file_path), dtype=float))
                    data_mea[id_].classes = LABELS_V1

            data_ms[ms_path.name] = data_mea
            pbar.close()

        # --- sanitize and finalize data ---

        if len(data_ms) > 1:
            flattened_data = flatten(data_ms, stop_cls=DataBunch)
            return flattened_data
        else:
            return data_mea

    # --------- static --------

    @staticmethod
    def _parse_type_selection(data_types):
        """
        Checks and parses the desired data types passed by the user.

        Parameters
        ----------
        data_types : list of str

        Returns
        -------
        selected_types : list of str

        """
        if len(data_types) == 0:
            selected_types = DATA_TYPES
        else:
            selected_types = [data_type.lower() for data_type in data_types
                              if data_type.lower() in DATA_TYPES]

        return selected_types

    @staticmethod
    def _load_discrete_annotation(file_path, color_cycle=False):
        """
        Processes the annotation data (discrete) generated with NOVA
        [https://git01.iis.fhg.de/abt-la/ssa/nova] or
        [https://rawgit.com/hcmlab/nova/master/docs/index.html]

        Parameters
        ----------
        file_path : pathlib.Path
            Name and path of the NOVA annotation file.
        color_cycle : bool, optional,
            Enables parsing the related colors of the labels.

        Returns
        -------
        annotation_df : pd.DataFrame
            Dataframe of all segments. The first two columns determine start and end time of a segment.
            The following id column states the type (label) of the segment. The last column indicates the annotator's
            confidence for the given label.
        classes : list of str
            List of all different classes used for the labelling. Their list position correspond to the NOVA-id.
        c_cycle : list of str or None
            List of colors (hex) which correspond to the classes. The colors are determined in NOVA.
        """

        # check for correct file extension
        if file_path.suffix != ANNOTATION_HEADER_FILE_EXTENSION:
            raise TypeError("Select a valid annotation file.")

        # reading segments from data file
        try:
            data = np.loadtxt(fname=str(file_path.with_suffix(ANNOTATION_DATA_FILE_EXTENSION)),
                              delimiter=CSV_DELIMITER)
        except OSError:
            print_msg(f"Missing annotation data: {file_path.with_suffix(ANNOTATION_DATA_FILE_EXTENSION)}", 'error')
            return

        # parse xml annotation file
        tree = xml.etree.ElementTree.parse(file_path)
        root = tree.getroot()

        scheme = [child for child in root if child.tag == 'scheme'][0]

        if scheme.attrib['type'] != 'DISCRETE':
            raise TypeError(f"Annotation scheme is no DISCRETE but {scheme.attrib['type']}.")

        classes = list(range(len(scheme)))
        c_cycle = list(range(len(scheme)))

        for item in scheme:
            classes[int(item.attrib['id'])] = item.attrib['name']
            c_cycle[int(item.attrib['id'])] = item.attrib['color']

        annotation_df = pd.DataFrame(data, columns=DATA_STRUCTURE)

        if color_cycle:
            return annotation_df, classes, c_cycle

        else:
            return annotation_df, classes, None

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

    # ------------------------
    # --- public functions ---
    # ------------------------

    @pickle_cache()
    def read(self, measurement_series=(), tools=(), data_types=(), measurement_campaigns=(),
             remove_garbage=True, remove_not_annotated=False, concatenated=False, columns_without_units=False,
             exclude_classes=(), exclude_labels=(), drop_zero_rows=False, caching=False, **kwargs):
        """
        Reads and prepares measurement data from the measurement-data repository
        with convenient generation of datasets for machine learning in mind.

        Parameters
        ----------
        measurement_series : list of str, optional
            Names (not a paths) of measurement series which should be processed.
            Passed names can be incomplete but should be unambiguous,
            e.g. "bmw" :math:`\\rightarrow` "factory_bmw_regensburg-00-20170830"
            If `tools` is also passed, the cut set is loaded.
        tools : list of str, optional
            Names of the hand tools for which all measurements should be processed.
            Passed names can be incomplete but should be unambiguous,
            e.g. "elec" :math:`\\rightarrow` "electric_screwdriver"
            If `measurement_series` is also passed, the cut set is loaded.
        data_types : list of str, optional
            Abbreviation for data types which should be incorporated. Can be 'IMU', 'MF' or 'AUDIO'.
        measurement_campaigns : list of int, optional
            Measurement campaign id for which all measurements should be processed.
        remove_garbage : bool default=True
            If True, all samples labelled as garbage are removed (i.e. -1 in :code:`.annotation~` file).
        remove_not_annotated : bool default = False
            If True, all samples which are not labelled are removed.
            (i.e. no corresponding label in the :code:`.annotation~` file)
        concatenated : bool, default=False
            If True, all measurements will be concatenated (if matching sampling rates)
            to a single pd.DataFrame-like object.
        columns_without_units : bool, default=False
            If True, all units will be removed from the column names, e.g. "time [s]" :math:`\\rightarrow` "time"
        exclude_classes : list of str, optional
            All samples which are labelled with these classes (str) will be discarded.
        exclude_labels : list of str, optional
            All samples which are labelled with these label (int) will be discarded.
        drop_zero_rows : bool, default=False
            All samples where all feature values are zero will be removed.
        caching : bool, default=False
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
            measurement_series = kwargs.get("ms", ())

        if "mc" in kwargs and len(measurement_campaigns) == 0:
            measurement_campaigns = kwargs.get("mc", ())

        if "dt" in kwargs and len(data_types) == 0:
            data_types = kwargs.get("dt", ())

        if len(measurement_campaigns) != 0:
            raise NotImplementedError(f"This feature is under development."
                                      f" Contact the maintainer {datatools.__maintainer__}")

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

        data = self._read_measurements(measurement_series, dts)

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

    @staticmethod
    def to_ts_data(data, contextual_recarray_dtype=None):
        """
        Converts data from the MeasurementDataReader to a format appropriate for the seglearn framework.

        Parameters
        ----------
        data : dict of DataBunch or DataBunch
            Data from as the read function from MeasurementDataReader provides.
        contextual_recarray_dtype : list of tuple, optional
            Numpy recarray dtype used to construct the contextual data.
            Default dtype is :code:`[('cls', ""), ('sr', float), ('id', int), ('desc', "")]`

        Returns
        -------
        ts_data : tuple of arrays
            Time series data in the seglearn format.
        """

        if contextual_recarray_dtype is None:
            contextual_recarray_dtype = [('cls', np.object), ('sr', float), ('id', int), ('desc', np.object)]

        if not isinstance(data, dict):
            data = {'00': data}

        num_ts = sum(len(data_bunch.data_keys()) for _, data_bunch in data.items())

        # preallocate new time series data
        Xt = [None] * num_ts
        y = [None] * num_ts
        Xc = np.recarray(shape=(num_ts,), dtype=contextual_recarray_dtype)

        k = 0
        for id_, data_bunch in data.items():
            for ts_type in data_bunch.data_keys():

                Xt[k] = data_bunch[ts_type].ts
                y[k] = data_bunch[ts_type].y

                try:
                    Xc[k].id = id_
                except ValueError:
                    Xc[k].id = int(id_.split("_")[-1])

                Xc[k].desc = ts_type
                Xc[k].sr = data_bunch[ts_type].sr

                k += 1

        return Xt, Xc, y
