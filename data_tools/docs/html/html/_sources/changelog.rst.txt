Change Log
==========

Version 0.5.1

* added option for reading all measurement series of a certain tool to :meth:`.MeasurementDataReader.read`
* fixed :meth:`.MeasurementDataReader.to_ts_data` for the case of multiple measurements series loaded


Version 0.5.0

* fixed occasionally missing labels when reading data
* refactoring of :meth:`.MeasurementDataReader._read_measurements`
* enhance test by checking stdout on warnings and errors


Version 0.4.5

* added two new options to :meth:`.MeasurementDataReader.read`:
    * :code:`remove_not_annotated`
    * :code:`exclude_labels`

* refactoring in :code:`data.py` and :code:`reader.py`


Version 0.4.4

* added a staticmethod :meth:`.MeasurementDataReader.to_ts_data` to convert the read data for usage with seglearn


Version 0.4.3

* added new caching feature when reading with :meth:`.MeasurementDataReader.read`


Version 0.4.2

* fixed concatenation option of :meth:`.MeasurementDataReader.read`
* replaced use of :meth:`pandas.Dataframe.from_csv` (deprecated) to :func:`pandas.read_csv`
* fixed option to select multiple measurement series in :meth:`.MeasurementDataReader.read`
