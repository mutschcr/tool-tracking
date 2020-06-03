# Built-in/Generic Imports
import pathlib
import datetime
import unittest

# Libs
import pytest
from _pytest.capture import capsys
import numpy as np
import pandas as pd


# Own modules
from datatools.data import DataCSV, DataBunch
from datatools.utils import print_msg
from datatools import MeasurementDataReader


print_msg.set_verbosity(3)

num_features = 3
num_samples = 50
sr = 100
time = np.arange(0, num_samples/sr, 1/sr)
X = np.random.rand(num_samples, num_features)
y = np.random.randint(2, size=num_samples)
classes = [f"label{i}" for i, _ in enumerate(np.unique(y))]
feature_names = ["time"] + [f"feature{i}" for i in range(num_features)]


def create_data():
    df_0 = DataCSV(data=np.c_[time, X],
                   columns=feature_names,
                   sampling_rate=100,
                   has_y_column=False,
                   id_='00')

    df_0.set_y(y)

    return df_0


def test_datacsv_creation():
    df_0 = create_data()
    np.testing.assert_array_equal(df_0.X, X)
    np.testing.assert_array_equal(df_0.y, y)
    assert df_0.features == feature_names[1:]


def test_concat():
    df_0 = create_data()
    df_1 = df_0.copy(deep=True)
    df_1._id = '01'

    data = {df._id: df for df in [df_0, df_1]}
    data_concat = pd.concat(data)

    assert isinstance(data_concat, DataCSV)
    assert df_0.sampling_rate == data_concat.sampling_rate


@pytest.mark.xfail(strict=True)
def test_wrong_concat():
    df_0 = create_data()
    df_1 = df_0.copy(deep=True)
    df_1._id = '01'
    df_1.sampling_rate = 101

    data = {df._id: df for df in [df_0, df_1]}

    # should fail here
    pd.concat(data)


# ------------------------------------
# --- tests: MeasurementDataReader ---
# ------------------------------------

def test_mdr_read(tmpdir):
    root_dir = pathlib.Path(tmpdir / 'test_tool' / 'test_measurement_series' / 'measurements')
    root_dir.mkdir(parents=True)
    mdr = MeasurementDataReader(tmpdir)

    df = create_data()

    np.savetxt(root_dir / "data-01.label", y)
    df.to_csv(root_dir / 'IMU-01-100.csv', sep=';', index=False)

    data = mdr.read(tools=['tool'])

    assert data.imu.sampling_rate == 100
    assert isinstance(data.imu, DataCSV)

    np.savetxt(root_dir / "data-02.label", y)
    df.to_csv(root_dir / 'IMU-02-100.csv', sep=';', index=False)
    data = mdr.read(tools=['test_tool'])

    assert isinstance(data, dict)


def test_mdr_selection_by_tool():
    mdr = MeasurementDataReader()

    for tool in mdr._tool_dict.keys():
        mdr.read(tools=[tool])


class ReadAllDataTestCase(unittest.TestCase):
    options = [{'remove_garbage': True, 'drop_zero_rows': False},
               {'remove_garbage': False, 'drop_zero_rows': True},
               {'columns_without_units': True},
               {'concatenated': True, 'data_types': ['imu', 'mf']},
               ]

    def _check_output(self):
        caps = self.capsys.readouterr()
        output = caps.out + caps.err

        if "[WARN]" in output or "[ERROR]" in output:

            # exceptional cases
            if "Dropping time column" in output:
                return True

            # fail
            else:
                print(output)
                return False
        else:
            return True

    def test_mdr(self):
        mdr = MeasurementDataReader()
        ms = [ms for measurement_series in mdr._tool_dict.values() for ms in measurement_series.keys()][0]
        for option in self.options:
            with self.subTest(ms=ms, mdr=mdr, option=option):
                data = mdr.read(ms=[ms], **option, caching=False)
                assert self._check_output()
                if 'concatenated' in option:
                    assert isinstance(data, DataBunch)

    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        self.capsys = capsys


# ------------------------
# --- tests: DataBunch ---
# ------------------------

def test_data_bunch_remove_classes():
    df = create_data()
    data_bunch = DataBunch(imu=df, classes=classes)

    data_bunch.remove(classes=["label0"])
    assert len(np.unique(data_bunch.imu.y)) == 1
    data_bunch.remove(["label1"])
    assert data_bunch.imu.X.size == 0


def test_data_bunch_remove_labels():
    df = create_data()
    data_bunch = DataBunch(imu=df, classes=classes)

    data_bunch.remove(labels=(0, ))
    assert len(np.unique(data_bunch.imu.y)) == 1
    data_bunch.remove(["label1"])
    assert data_bunch.imu.X.size == 0


def test_data_bunch_drop_zero_rows():
    df = create_data()
    df.loc[0:9, df.features] = 0.0
    data_bunch = DataBunch(imu=df, mf=df)

    data_bunch.remove(zero=True)
    for df_dropped in [data_bunch.imu, data_bunch.mf]:
        assert np.array_equal(df.y[10:20], df_dropped.y[0:10])
        assert np.array_equal(df.X[10:20, :], df_dropped.X[0:10, :])
        assert not np.isnan(df.X).all()
        assert not np.isnan(df.y).all()


def test_data_bunch_drop_nan():
    df = create_data()
    y = df.y.astype(float)
    length = len(y)
    y[length // 2] = np.nan
    df['label'] = y
    data_bunch = DataBunch(imu=df, mf=df)

    assert np.isnan(data_bunch.imu.y).any()

    data_bunch.remove(nan=True)
    for df_dropped in [data_bunch.imu, data_bunch.mf]:
        assert length - 1 == len(df_dropped.y)
        assert not np.isnan(df_dropped.y).any()


# ------------------------
# --- tests: structure ---
# ------------------------

class FolderStructureError(Exception):
    def __init__(self, message=''):
        super().__init__(message)


class NamingError(Exception):
    def __init__(self, message=''):
        super().__init__(message)


ROOT_FOLDERS = {'data_tools', 'info'}
MS_FOLDERS = {'info', 'measurements'}


def get_folders():
    mdr = MeasurementDataReader()
    repro_path = mdr._path_to_root_dir
    _folders = [folder.name for folder in repro_path.iterdir() if folder.is_dir()]

    tool_folders = [tool for tool in repro_path.iterdir()
                    if tool.is_dir() and "." not in tool.name and tool.name not in ROOT_FOLDERS]

    ms_folders = [ms for tool in tool_folders for ms in tool.iterdir() if ms.is_dir() and '.' not in ms.name]

    return _folders, ms_folders


def test_repository_structure():
    _folders, _ = get_folders()

    root_folders = {folder for folder in ROOT_FOLDERS if folder in _folders}

    if ROOT_FOLDERS != root_folders:
        missing = ROOT_FOLDERS - root_folders
        raise FolderStructureError("There are missing folder(s) at root level: {}".format(", ".join(missing)))


def test_integrity():
    _, ms_folders = get_folders()

    for ms_folder in ms_folders:
        m_folders = [f.name for f in ms_folder.iterdir() if f.is_dir()]
        ms_subfolders = {folder for folder in MS_FOLDERS if folder in m_folders}

        if MS_FOLDERS != ms_subfolders:
            missing = MS_FOLDERS - ms_subfolders
            raise FolderStructureError("The following folder(s) are missing in the measurement series '{}': {}".format(
                ms_folder.name, ", ".join(missing)))


def test_naming_ms():
    _, ms_folders = get_folders()

    for ms_folder in ms_folders:

        name = ms_folder.name.split("-")

        if len(name) != 3:
            raise NamingError("The naming of measurement series '{}' is incorrect."
                              " Should be <location>-<mc_id>-<YYYYMMDD>".format(ms_folder.name))

        location, _id, date = name

        if len(_id) != 2:
            raise NamingError("The measurement campaign id of measurement series '{}'"
                              " has to be an identifier with two digits".format(ms_folder.name))

        try:
            datetime.datetime.strptime(date, '%Y%m%d')
        except ValueError:
            raise NamingError("The date of measurement series '{}'"
                              " has to be in the format 'YYYYMMDD'".format(ms_folder.name))


def test_naming_data():
    _, ms_folders = get_folders()

    for ms_folder in ms_folders:
        measurements = ms_folder / "measurements"

        files = [f for f in measurements.iterdir() if f.is_file() and f.name != '.directory']

        for file in files:

            name = file.name.split(".")[0].split("-")

            if len(name) != 3 and name[0] != 'data':
                raise NamingError("Please check naming of file '{}' in '{}'".format(file.name, ms_folder.name))

            kind, _id, *_ = name

            if len(_id) != 2:
                raise NamingError("The measurement id of file '{}' in '{}'"
                                  " has to be a identifier with to digits".format(file.name, ms_folder.name))


def test_nova_completeness():
    _, ms_folders = get_folders()

    for ms_folder in ms_folders:
        measurements = ms_folder / "measurements"

        files = [f for f in measurements.iterdir() if f.is_file() and f.name != '.directory']

        counter = 0
        for file in files:
            if file.suffix == '.annotation' or file.suffix == '.annotation~' or file.suffix == '.nova':
                counter += 1

        if not counter % 3 == 0:
            raise FileNotFoundError(f"At least one of the three nova files ('.nova', '.annotation', '.annotation~') "
                                    f"in {ms_folder} is missing")
