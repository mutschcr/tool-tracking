User Guide
==========

Load data with the :class:`~datatools.reader.MeasurementDataReader`
*******************************************************************

>>> from datatools import MeasurementDataReader
>>> mdr = MeasurementDataReader()
>>> data_dict = mdr.read(measurement_series=["bmw"])  # read_data() is deprecated

.. code-block:: none

   [INFO] Preparing measurements from:
     /home/wllr/measurement-data/electric_screwdriver/factory_bmw_regensburg-17-08
   [INFO] Reading: 100%|███████████| 21/21 [00:02<00:00,  7.64it/s, file=MF-02-100.csv]
   [INFO] Finished with 7 measurement(s).

.. note::

   The passed measurement series name can be incomplete but should be unambiguous.
   E.g.: "bmw" :math:`\rightarrow` "factory_bmw_regensburg-17-08".

The selected measurement series (factory_bmw_regensburg-17-08) contains 7 datasets with data from the accelerometer and the magnetic field sensor. Each dataset is identified by a double-digit ID.

- The `data_dict` returned by the :class:`~datatools.reader.MeasurementDataReader` is a :class:`dict` containing the seven datasets as :class:`~datatools.data.DataBunch` which can be accessed by their IDs as keys.
- The :class:`~datatools.data.DataBunch` class stores all the data for one single measurement. To see which data is available check with :code:`info`:

>>> data_bunch = data_dict['01']
>>> data_bunch.info
['imu', 'classes']

Here we have two keys:

- **imu**: contain acc, mf data and the corresponding label/target vector (encoded with values between 0 and n_classes-1)
- **classes**: a list of classes used for labelling (strings)

If audio data is available there is a third key:

- **audio**: contain audio data and the corresponding label/target vector (encoded with values between 0 and n_classes-1)

>>> imu_data = data_bunch.imu
>>> type(imu_data)
datatools.types.DataCSV

The actual measurement data is stored using :class:`~datatools.data.DataCSV` which is actually a :class:`pandas.DataFrame` with some useful extensions, but can be handled like an ordinary :class:`~pandas.DataFrame`.


>>> data_bunch.imu

+---+----------------------------+----------------------------+-----+-------+
|   | acceleration x-axis [m/s²] | acceleration y-axis [m/s²] | ... | label |
+===+============================+============================+=====+=======+
| 0 | 10.146501                  | -1.817820                  | ... | 0     |
+---+----------------------------+----------------------------+-----+-------+
| 1 | ...                        | ...                        | ... | ...   |
+---+----------------------------+----------------------------+-----+-------+

.. note::

   If there is only *one* dataset, a single :class:`~datatools.data.DataBunch` is returned instead of a :class:`dict`!


Interaction with the :class:`~datatools.data.DataBunch` class
*************************************************************

A :class:`~datatools.data.DataBunch` object is a dictionary-like object that exposes its keys as attributes.

- access the data with :code:`data_bunch.imu` or :code:`data_bunch['audio']`
- see the class :class:`documentation <datatools.data.DataBunch>` for more helpful functionality
- it is also possible to *concatenate* two bunches if reasonable via:


>>> data_bunch = data_dict['01'] + data_dict['02']
[WARN] Dropping time column.

or even easier

>>> data_bunch = DataBunch.concat(data_dict)

.. note::
   The :class:`~datatools.data.DataBunch` class is a thin wrapper around :class:`sklearn.datasets.base.Bunch`

Interaction with the :class:`~datatools.data.DataCSV` class
***********************************************************

The :class:`~datatools.data.DataCSV` class stores some useful meta information.
If you are interested in the actual feature names you can simply enter:


>>> data_bunch.imu.features
['acceleration x-axis [m/s²]',
 'acceleration y-axis [m/s²]',
 'acceleration z-axis [m/s²]',
 'angular rate x-axis [°/s]',
 'angular rate y-axis [°/s]',
 'angular rate z-axis [°/s]',
 'magnetic field x-axis [Gs]',
 'magnetic field y-axis [Gs]',
 'magnetic field z-axis [Gs]']

Also the :code:`time` column as well as the :code:`sampling_rate` is available.

To train a machine learning model you might need the design matrix *X* and the target vector *y*:


>>> X = data_bunch.imu.X  # numpy array
>>> y = data_bunch.imu.y  # numpy array
...

you can also use the aliases

>>> y = data_bunch.imu.target
>>> y = data_bunch.imu.label
...

Convert data for a `seglearn <https://dmbee.github.io/seglearn>`_ pipeline
**************************************************************************

First, get measurement data

>>> from datatools import MeasurementDataReader
>>> mdr = MeasurementDataReader()
>>> data_dict = mdr.read(ms=["pythagoras-01-20190705"])

Let's prepare the :code:`data_dict` for seglearn's :class:`~seglearn.pipe.Pype` using :class:`~seglearn.base.TS_Data`

>>> Xt, Xc, y = mdr.to_ts_data(data_dict)
>>> from seglearn.base import TS_Data
>>> X = TS_Data(Xt, Xc)

Now we can for example resample to a common sampling rate using the :mod:`temporal.transform` package from `DAT <https://git01.iis.fhg.de/abt-la/ssa/DataAnalysisToolkit>`_

>>> from temporal.transform import Resample
>>> from seglearn.pipe import Pype
>>> pipe = Pype([('resample', Resample(sr=100))])
>>> X_resample, y_resample = pipe.fit_transform(X, y)

Go to the `basic tutorial <http://abt-la.git01.iis.fhg.de/ssa/DataAnalysisToolkit/temporal.transform_example.html>`_ to learn how to transform time series.
