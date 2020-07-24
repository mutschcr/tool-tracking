# --- build-in ---
from typing import Union, Dict, List, Tuple, Optional

# --- third-party ---
import numpy as np

# --- own ---
from .data import DataBunch
from .utils import separate


def to_ts_data(
        data: Union[DataBunch, Dict[str, DataBunch], Dict[str, List[DataBunch]]],
        contextual_recarray_dtype: Optional[List[Tuple]] = None
) -> Tuple[List, np.recarray, List]:
    """
    Converts data from the MeasurementDataReader to a format appropriate for the seglearn framework.

    Parameters
    ----------
    data : dict of DataBunch or DataBunch
        Data from as the read function from MeasurementDataReader provides.
    contextual_recarray_dtype
        Numpy recarray dtype used to construct the contextual data.
        Default dtype is :code:`[('cls', ""), ('sr', float), ('id', int), ('desc', "")]`

    Returns
    -------
    ts_data : tuple of arrays
        Time series data in the seglearn format.
    """

    if contextual_recarray_dtype is None:
        contextual_recarray_dtype = [('cls', np.object), ('sr', float), ('id', int), ('desc', np.object)]

    # CASE: single DataBunch
    if hasattr(data, "data_keys"):
        data = [('00', data)]

    # CASE: dict
    else:
        data = separate(data, stop_cls=DataBunch)

    num_ts = sum(len(bunch.data_keys()) for _, bunch in data)

    # preallocate new time series data
    Xt = [None] * num_ts
    y = [None] * num_ts
    Xc = np.recarray(shape=(num_ts,), dtype=contextual_recarray_dtype)

    k = 0
    id_: str
    bunch: DataBunch
    for id_, bunch in data:
        for ts_type in bunch.data_keys():

            Xt[k] = bunch[ts_type].ts
            y[k] = bunch[ts_type].y

            try:
                Xc[k].id = id_
            except ValueError:
                try:
                    Xc[k].id = int(id_.split("_")[-1])
                except ValueError:
                    Xc[k].cls = id_

            Xc[k].desc = ts_type
            Xc[k].sr = bunch[ts_type].sr

            k += 1

    return Xt, Xc, y
