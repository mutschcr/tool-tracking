from .__version__ import __version__, __maintainer__, __license__, __copyright__
from .reader import MeasurementDataReader
from .data import DataBunch, ACC, GYR, MAG, POS, VEL, MIC
from .query import DataTypes, Tool, Action, Config, Measurement, MeasurementSeries, Query
from .utils import print_msg
from .convert import to_ts_data


def show_versions():
    print(__version__)
