import numpy as np
import math
import sys
from functools import lru_cache, reduce
from tqdm import tqdm
import collections
from collections import Counter
from itertools import product
from seglearn.transform import XyTransformerMixin
from seglearn.util import check_ts_data, get_ts_data_parts
from seglearn.base import TS_Data
import sklearn
import logging
from sklearn.utils.validation import check_array, check_is_fitted

n_digits_unix_in_seconds = 10
contextual_recarray_dtype = [('cls', np.object), ('sr', float), ('id', int), ('desc', np.object), ('height', float)]

# create logger
logger = logging.getLogger("transformer")


def most_frequent_label_per_window(y_w, threshold=None):
    """
    Compute the most frequent label for every row (i.e. window) of the input array.

    Parameters
    ----------
    y_w : numpy array, shape (n_windows, window_size)
        Array of windowed labels.
    threshold : float, optional
        Proportion of the most frequent label needed to assign this label for the whole window.

    Returns
    -------
    y_w_labels : numpy array, shape (n_windows,)
        Array of the most frequent label of every row (i.e. window) of the input array.

    Example
    -------
    >>> y_w = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0]).reshape(3, 3)
    >>> y_w
    array([[0, 0, 1],
           [1, 0, 0],
           [1, 1, 0]])
    >>> most_frequent_label_per_window(y_w)
    array([0, 0, 1])
    >>> y_w = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(4, 4)
    >>> most_frequent_label_per_window(y_w, threshold=0.6)
    array([-1.,  1.,  0.,  1.])
    """

    # create empty array and fill it with the most frequent labels per window each
    y_w_labels = np.empty(len(y_w))
    for window_idx in range(len(y_w)):
        labels, counts = np.unique(y_w[window_idx], return_counts=True)
        if threshold is not None:
            idx = np.argmax(counts)
            if counts[idx] >= threshold * np.sum(counts):
                y_w_labels[window_idx] = labels[idx]
            else:
                y_w_labels[window_idx] = -1
        else:
            y_w_labels[window_idx] = labels[np.argmax(counts)]

    return y_w_labels

def filter_by_labels(data, filter_labels):
    """
    Filter labels and remove all windows where:
    1) the label is '-1'
    2) the label occurs less than 50% of the window
    """

    data.y = most_frequent_label_per_window(data.yt, threshold=0.5)

    # get mask to remove all windows with label '-1'
    mask = np.isin(data.y, filter_labels, invert=True)

    return data[mask]


class TqdmLoggingHandler(logging.Handler):
    """
    Parameters
    ----------
    level
        Default log level.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


if len(logger.handlers) == 0:
    logger.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(levelname)s] %(message)s')

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(formatter)

    logger.addHandler(tqdm_handler)

def get_sampling_rate(t, kind='mean', unix=False, decimals=None, t_unit=None):
    """
    Sampling rate from time vector.

    This function calculates the sampling rate for a given time vector. In the case of leaps in time the computation
    method can be chosen with the ``kind`` parameter, e.g. 'mean' or 'median'. Both unix timestamps and relative time
    are supported.

    Parameters
    ----------
    t : array-like, shape [n_time_steps, ]
        Time vector (1D) with either unix timestamps or relative time.
    kind : str
        Method for calculating the sampling rate. E.g. 'mean' or 'median'. Can be an any name of a numpy function
        which accepts ``axis`` and thus can return a scalar value (e.g. 'std', 'amin', etc.).
    unix : bool, optional, default: False
        If True unix timestamps will be assumed.
    decimals : int, optional
        Number of decimal places to round to. If zero than the returned sampling rate will be of type integer.
    t_unit : {'s', 'ms', 'us'}, optional
        In case of relative time specify the unit.

    Returns
    -------
    sr : float or int
        Sampling rate of input t in Hz.

    Examples
    -------
    >>> T = 4  # seconds
    >>> sr = 100.5  # Hz
    >>> x = np.linspace(start=0, stop=T, num=int(T*sr), endpoint=False)
    >>> get_sampling_rate(x, kind='mean', t_unit='s', decimals=1, unix=False)
    100.5
    """
    factor = 1
    if unix:
        n_digits = len(str(int(t[0])))

        # convert time array to units of seconds if necessary
        if n_digits != n_digits_unix_in_seconds:
            t /= 10**(n_digits - n_digits_unix_in_seconds)
    else:
        if t_unit == 's':
            factor = 1
        elif t_unit == 'ms':
            factor = 10 ** -3
        elif t_unit == 'us':
            factor = 10 ** -6
        else:
            raise ValueError("Unknown unit " + str(t_unit) + " for time!")

    try:
        func = getattr(np, kind)
    except AttributeError:
        raise ValueError(f"Unknown kind '{kind}'")

    sr = 1 / func(np.diff(t * factor))

    if decimals is not None:
        sr = np.round(sr, decimals=decimals)
        if decimals == 0:
            return int(sr)
        else:
            return sr
    else:
        return sr


def filter_ts_data(X, y=None, filt=None):
    """
    Separates time series data object into time series variables and contextual variables.
    Applies filtering based on the contextual information.

    Parameters
    ----------
    X : seglearn.TS_Data
       Time series data and (optionally) contextual data
    filt : dict
        Dictionary with filter conditions based on contextual data. Valid keys are {'cls', 'sr', 'id', 'desc'}.

    Returns
    -------
    Xt : array-like, shape [n_series, ]
        Time series data
    Xc : array-like, shape [n_series, ...]
        contextual variables
    y : array-like
        Labels.
    """
    Xt, Xc = X.ts_data, X.context_data

    if filt is not None:
        valid_keys = {key for key, _ in contextual_recarray_dtype}
        if not set(filt.keys()).issubset(valid_keys):
            raise ValueError(f"Possible keys for filtering"
                             f" are {valid_keys} but not {set(filt.keys()) - valid_keys}")

        indices = []
        for key, val in filt.items():
            if isinstance(val, (list, tuple, np.ndarray)):
                indices_per_key = []
                for v in val:
                    indices_per_key.append(set(
                        np.argwhere(Xc[key] == v).flatten().tolist()
                    ))
                indices.append(set.union(*indices_per_key))
            else:
                indices.append(set(np.argwhere(Xc[key] == val).flatten().tolist()))

        idx = list(set.intersection(*indices))

        X = X[idx]

        if y is not None:
            y = np.array(y)[idx].tolist()

    if not isinstance(X, TS_Data):
        return X, None, y
    else:
        return X.ts_data, X.context_data, y


def find_nearest(a, targets):
    """
    Return the indices of the values which come closest to the target values. It is assumed that the array is sorted.

    Parameters
    ----------
    a : array_like
        Input array.
    targets : float or array_like
        Value(s) to find in the input array.

    Returns
    -------
    idx : int or np.ndarray
        Indices of the values which come closest to the target values. In case of only one target value an integer is returned.

    Examples
    ----------
    >>> a = [0.1, 1.3, 2.6, 3.1]
    >>> find_nearest(a, 2)
    2

    >>> find_nearest(a, [-0.1, 1, 2.9])
    array([0, 1, 3])
    """
    targets = np.atleast_1d(targets)
    indices = np.atleast_1d(np.searchsorted(a, targets, side="left")).astype('int')

    for i, idx in enumerate(indices):
        if idx > 0 and (idx == len(a) or math.fabs(targets[i] - a[idx - 1]) < math.fabs(targets[i] - a[idx])):
            indices[i] = idx - 1

    if len(indices) == 1:
        return indices[0]
    else:
        return indices


def argnear(a, target):
    """
    Return the index of the value which comes closest to the target value.

    Parameters
    ----------
    a : np.ndarray
        Input array.
    target : float
        Value to find in the input array.

    Returns
    -------
    idx : int
        Index of the values which comes closest to the target value.

    See Also
    --------
    find_nearest :
        A much faster implementation if the array is already sorted.

    Examples
    ----------
    >>> a = np.array([0.1, 1.3, 2.6, 3.1])
    >>> argnear(a, 2)
    2

    """
    return (np.abs(a - target)).argmin()


@lru_cache(maxsize=None)
def _divs(n):
    """Memoized recursive function returning prime factors of n as a list"""
    for i in range(2, int(math.sqrt(n) + 1)):
        d, m = divmod(n, i)
        if not m:
            return [i] + _divs(d)
    return [n]


def prime_factors(n):
    """Map prime factors to their multiplicity for n"""
    d = _divs(n)
    d = [] if d == [n] else (d[:-1] if d[-1] == d else d)
    pf = Counter(d)
    return dict(pf)


def round2int(a):
    """
    Round the input to the nearest integer value.

    Parameters
    ----------
    a : float or int

    Returns
    -------
    int

    """
    return int(round(a, ndigits=0))


def proper_divs(n):
    """
    Return the set of proper divisors of n.

    Parameters
    ----------
    n : int
        Number for which the divisors will be calculated.

    Returns
    -------
    divisors : set of int

    Examples
    -------
    >>> proper_divs(12)
    {1, 2, 3, 4, 6}

    To get the divisors for a floating point number use

    >>> number = 2.4
    >>> divisors = proper_divs(int(2.4 * 10))
    >>> np.array(list(divisors)) / 10
    array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2])

    """
    pf = prime_factors(n)
    pfactors, occurrences = pf.keys(), pf.values()
    multiplicities = product(*(range(oc + 1) for oc in occurrences))
    divs = {reduce(int.__mul__, (pf ** m for pf, m in zip(pfactors, multis)), 1)
            for multis in multiplicities}
    try:
        divs.remove(n)
    except KeyError:
        pass
    return divs or ({1} if n != 1 else set())


def get_exponent(x):
    return np.floor(np.log10(np.abs(x))).astype(int)


def moving_window(sequence, window_size, step_size=None, incomplete=False):
    """
    Generator that will iterate through the defined slices of input sequence (windows).

    Parameters
    ----------
    sequence : iterable
    window_size : int
    step_size : int, optional
        If no step size is passed it will be set to the window size. This means that there will be no overlap between
        the windows.
    incomplete : bool
        Enables incomplete windows at the end of the sequence.

    Yields
    ----------
    window : array-like
        Slice from the sequence with size window_size.

    Examples
    ----------

    There is no overlap between the windows in the default case:

    >>> a = [1, 2, 3, 4]
    >>> list(moving_window(a, 2))
    [[1, 2], [3, 4]]

    The overlap can be controlled using the step_size parameter:

    >>> s = 'ABCDEFG'
    >>> list(moving_window(s, 3, 2))
    ['ABC', 'CDE', 'EFG']

    With incomplete windows:

    >>> x = np.array([1, 2, 3, 4, 5])
    >>> list(moving_window(x, 3, 3, incomplete=True))
    [array([1, 2, 3]), array([4, 5])]

    """

    # input validation
    if not isinstance(sequence, collections.abc.Iterable):
        raise TypeError("Input sequence has to be an iterable")

    if not isinstance(window_size, int) or (not isinstance(step_size, int) and step_size is not None):
        raise TypeError("Window size and step size must be integers")
    elif step_size is None:
        step_size = window_size

    if step_size > window_size:
        raise ValueError("Step size must not be larger than window size.")

    if window_size > len(sequence):
        raise ValueError("Window size must not be larger than sequence length.")

    # pre-compute number of chunks to emit
    num_chunks = math.ceil(((len(sequence) - window_size) / step_size) + 1)

    # windowing
    for i in range(0, num_chunks * step_size, step_size):
        window = sequence[i:i + window_size]
        if len(window) == window_size or (len(window) < window_size and incomplete):
            yield window
        else:
            return


class SubsequentTransformer:
    _N_SDT = 2
    _N_OBS = 2

    def __setattr__(self, key, value):
        if key in ['n', 'nobs', 'nsdt', 't_unit'] and value is None:
            if key in ['n', 'nsdt']:
                value = self._N_SDT
            elif key == 'nobs':
                value = self._N_OBS
            elif key == 't_unit':
                value = self._T_UNIT
            else:
                raise Exception("[CRITICAL] Should not happen.")

        elif key in ['n', 'nobs', 'nsdt'] and value == 0:
            raise ValueError(f"{key} must be greater than zero.")

        super().__setattr__(key, value)


class BaseEstimator(sklearn.base.BaseEstimator):
    _VERBOSE = True
    _T_UNIT = 's'

    def __setattr__(self, key, value):
        if key == 't_unit' and value is None:
            value = self._T_UNIT
        super().__setattr__(key, value)


class Segment(BaseEstimator, XyTransformerMixin, SubsequentTransformer):
    """
    Transformer for sliding window segmentation, optionally with contextual data. The target y and the contextual data
    is mapped to all segments from their parent series. Using the parameter 'n', n subsequent time series will be
    segmented together. This means that they will be segmented according to their timestamps for best time matching
    segments.

    .. note::

       A constant number of samples per window is not enforced.
       The target must have the same sampling rate as the corresponding time series.

    Parameters
    ----------
    window_length : float
        Desired window length in seconds.
    overlap : float, default=0.0
        Overlap of two consecutive windows, has to be in the interval I = [0, 1).
    n : int, default=2
        Number of subsequent time series to be segmented together.
    """

    def __init__(self, window_length, overlap=0.0, enforce_size=False, n=None):
        self.window_length = window_length
        self.overlap = overlap
        self.enforce_size = enforce_size
        self.n = n

    def fit(self, X, y=None):
        Xt, Xc = get_ts_data_parts(X)
        N = len(Xt)  # Number of time series

        # ensure that Xt is an numpy array. necessary for indexing
        Xt = np.array(Xt)

        # check if 'n' is valid
        if ((Xc is not None) and (np.unique(Xc.desc).size != self.n)) or \
                (not isinstance(self.n, int)) or (N % self.n != 0):
            n_suggestion = np.unique(Xc.desc).size if Xc is not None else proper_divs(N)
            print(f"[WARNING] The value of 'n' ({self.n}) is suspicious. Should be {n_suggestion} most likely.")

        self.reference_windows_ = []  # list of reference windows
        self.num_new_ts_ = 0  # number of new time series after transformation

        for selection_idx in moving_window(range(N), window_size=self.n):
            start = np.min([ts[0, 0] for ts in Xt[selection_idx]])
            stop = np.max([ts[-1, 0] for ts in Xt[selection_idx]])
            duration = stop - start

            if duration > (60 * 60):
                print(f"[WARN] duration for reference time is quite high ({duration / 60 / 60:.2f}h)."
                      f" Most likely the parameter 'n' is wrong.")

            if Xc is not None:
                factor = 10 ** (abs(get_exponent(np.max(Xc[selection_idx].sr)) + 1))

                print(f"[INFO] segment {Xc.desc[selection_idx]} together")
            else:
                factor = 1e5

            divisors = np.array(list(proper_divs(round2int(self.window_length * factor)))) / factor

            if Xc is not None:
                precision = divisors[argnear(divisors, 1 / (np.max(Xc[selection_idx].sr) * 2))]
            else:
                precision = min(divisors)

            num = round2int(duration / precision)

            # generate reference time stamps
            t_ref, step = np.linspace(start=0, stop=duration, num=num, retstep=True)
            t_ref += start

            # window length in seconds to number of samples
            window_size = round2int(self.window_length / step)

            # get window iterator
            wins = moving_window(
                sequence=t_ref,
                window_size=window_size,
                step_size=round2int(window_size * (1 - self.overlap)),
                incomplete=False)

            # remove unused timestamps from reference windows
            win_ref = [(win[0], win[-1]) for win in wins]

            self.reference_windows_.append(win_ref)
            self.num_new_ts_ += (len(win_ref) * self.n)

        return self

    def transform(self, X, y, sample_weight=None):
        check_is_fitted(self, ['reference_windows_', 'num_new_ts_'])

        Xt, Xc = get_ts_data_parts(X)
        yt = y
        N = len(Xt)  # Number of time series

        # preallocate new time series data
        Xt_trans = [None] * self.num_new_ts_
        y_trans = [None] * self.num_new_ts_
        Xc_trans = np.recarray(shape=(self.num_new_ts_,), dtype=contextual_recarray_dtype)

        k = 0
        pbar = tqdm(total=self.num_new_ts_, desc="Segment", disable=(not self._VERBOSE), file=sys.stdout)

        # get time series which should be segmented together
        for window, selection_idx in zip(self.reference_windows_, moving_window(range(N), window_size=self.n)):

            # get reference windows for segmentation
            for starting_timestamp, ending_timestamp in window:

                # segment each time series
                for idx in selection_idx:
                    ts = Xt[idx]
                    start_idx = find_nearest(ts[:, 0], starting_timestamp)

                    if self.enforce_size and Xc is not None:
                        stop_idx = start_idx + round2int(self.window_length * Xc[idx].sr)
                    else:
                        stop_idx = find_nearest(ts[:, 0], ending_timestamp)

                    if stop_idx < start_idx:
                        raise ValueError

                    Xt_trans[k] = ts[start_idx:stop_idx, :]

                    if Xc is not None:
                        Xc_trans[k] = Xc[idx]

                    if yt is not None and len(yt[0].shape) >= 2:
                        start_idx = find_nearest(yt[idx][:, 0], starting_timestamp)

                        if self.enforce_size:
                            stop_idx = start_idx + round2int(
                                self.window_length * get_sampling_rate(yt[idx][:, 0], t_unit=self._T_UNIT)
                            )
                        else:
                            stop_idx = find_nearest(yt[idx][:, 0], ending_timestamp)

                        if stop_idx < start_idx:
                            raise ValueError

                        y_trans[k] = yt[idx][start_idx:stop_idx, :]
                    elif yt is not None:
                        y_trans[k] = yt[idx][start_idx:stop_idx]

                    pbar.update(1)
                    k += 1

        pbar.close()

        assert len([1 for ts, y in zip(Xt_trans, y_trans)
                    if ts is None or y is None]) == 0, "[CRITICAL] Missing segments."

        # --- find empty windows and delete them ---
        empty_windows = []

        for i, (ts, y) in enumerate(zip(Xt_trans, y_trans)):
            if len(ts) <= 1 or len(y) <= 1:
                empty_windows.append(i)

        if len(empty_windows) > 0:
            Xt_trans = np.delete(Xt_trans, empty_windows)

            if Xc is not None:
                Xc_trans = np.delete(Xc_trans, empty_windows)

            if y is not None:
                y_trans = np.delete(y_trans, empty_windows)

        num_empty_windows = len(empty_windows)

        if num_empty_windows > 0:
            logger.warning(f"[{self.__class__.__name__}] {num_empty_windows} windows could not be processed "
                           f"and thus removed")

        # --- finalize ---
        if Xc is not None:
            Xt = TS_Data(Xt_trans, Xc_trans)
        else:
            Xt = Xt_trans

        return Xt, y_trans, sample_weight


