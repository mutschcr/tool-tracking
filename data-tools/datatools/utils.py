# --- built-in ---
import uuid
import pickle
import pathlib
import tempfile
import collections
import functools
import inspect
from typing import Callable, TypeVar, Type, Union, Dict, List, MutableMapping, Tuple

# --- third-party ---
import numpy as np

T = TypeVar('T')


# ---------------
# --- classes ---
# ---------------
class MSGPrinter:
    LEVEL_NAMES = {'error': 0,
                   'warn': 1,
                   'info': 2,
                   'indent': 2,
                   'debug': 3,
                   }

    def __init__(self, verbose=2):
        self.verbose = verbose

    def __call__(self, string, cat=None):
        _ = MSGPrinter.LEVEL_NAMES.keys()

        if MSGPrinter.LEVEL_NAMES.get(cat, 0) <= self.verbose:
            if cat == 'indent':
                cat = '  '
            else:
                cat = f"[{cat.upper()}] "

            print(cat + string)

    def set_verbosity(self, verbose):
        if 3 >= self.verbose >= 0:
            self.verbose = verbose
        else:
            raise ValueError("No valid verbosity level (0-3).")


print_msg = MSGPrinter()  # preferred alias


class SamplingRateError(Exception):
    def __init__(self, message=''):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class TimeColumnError(Exception):
    def __init__(self, message=''):
        super().__init__(message)


class ConcatError(Exception):
    def __init__(self, message=''):
        super().__init__(message)


# -----------------
# --- functions ---
# -----------------
def get_sampling_rate(t, t_unit='s', kind='mean', ndigits=None):
    if t_unit == 's':
        factor = 1
    elif t_unit == 'ms':
        factor = 10**-3
    elif t_unit == 'us':
        factor = 10**-6
    else:
        raise ValueError("Unknown unit " + str(t_unit) + " for time!")

    if kind == 'mean':
        sr = 1 / np.mean(np.diff(t * factor))
    elif kind == 'median':
        sr = 1 / np.median(np.diff(t * factor))
    else:
        raise ValueError(f"Unknown kind '{kind}'")

    if ndigits is None:
        return sr
    elif ndigits == 0:
        return int(round(sr, ndigits=ndigits))
    else:
        return round(sr, ndigits=ndigits)


def flatten(d: MutableMapping, parent_key: str = "", sep: str = "_", stop_cls: Type = None) -> Dict:
    items = []
    for key, value in d.items():
        new_key = parent_key + sep + str(key) if parent_key else key
        if isinstance(value, collections.abc.MutableMapping) and\
                (not isinstance(value, stop_cls) if (stop_cls is not None) else True):
            items.extend(flatten(value, new_key, sep=sep, stop_cls=stop_cls).items())
        else:
            items.append((new_key, value))
    return dict(items)


def separate(d: MutableMapping, parent_key: str = "", sep: str = "_", stop_cls: Type = None) -> List[Tuple]:
    items = []
    for key, value in d.items():
        new_key = parent_key + sep + str(key) if parent_key else key
        if hasattr(value, "__len__") and not isinstance(value, collections.abc.MutableMapping) and\
                (not isinstance(value, stop_cls) if (stop_cls is not None) else True):
            for v in value:
                items.append((new_key, v))
        elif isinstance(value, collections.abc.MutableMapping) and\
                (not isinstance(value, stop_cls) if (stop_cls is not None) else True):
            items.extend(separate(value, new_key))
        else:
            items.append((new_key, value))

    return items


def pickle_cache(path=tempfile.gettempdir(), size=10):
    path = pathlib.Path(path)
    cache_size = size
    maxlen = cache_size * 10

    def decorator(func):
        mapping = {}
        key_history = collections.deque(maxlen=maxlen)

        def _sanitize():
            if len(mapping) >= cache_size:
                key_least_common = collections.Counter(key_history).most_common(maxlen)[-1][0]
                mapping[key_least_common].unlink()
                del mapping[key_least_common]

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if kwargs.get('caching', False):
                try:
                    key = str(args) + str(tuple(sorted(kwargs.items())))

                    if key in mapping:
                        filename = mapping[key]
                        with open(filename, 'rb') as f:
                            res = pickle.load(f)
                        print_msg("Loading from cache", 'info')
                    else:
                        _sanitize()
                        res = func(self, *args, **kwargs)
                        filename = path / f"{uuid.uuid4()}_mdr_cache.pkl"

                        with open(filename, 'wb') as f:
                            pickle.dump(res, f)

                        mapping.update({key: filename})
                        print_msg("Cached result", 'info')

                    key_history.append(key)

                except Exception as e:
                    res = func(self, *args, **kwargs)
                    print_msg(f"Caching failed: {e}", 'warn')
            else:
                res = func(self, *args, **kwargs)

            return res
        return wrapper
    return decorator


def convert2path(func: Callable):
    if hasattr(func, "__annotations__"):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            for i, (param_name, param) in enumerate(signature.parameters.items()):
                if param_name == "self":
                    continue

                if "Path" in str(param.annotation):
                    if param_name in kwargs:
                        kwargs[param_name] = pathlib.Path(kwargs[param_name])
                    else:
                        args[i] = pathlib.Path(args[i])

            return func(*args, **kwargs)

        return wrapper
    else:
        return func


def count_objects(target: Union[Dict, List], cls: Type[T]):
    count = 0

    def _check(value_):
        nonlocal count
        if isinstance(value_, cls):
            count += 1
        elif type(value_) is dict:
            dict_search(value_)
        elif isinstance(value_, list):
            list_search(value_)
        else:
            return

    def dict_search(target_):
        for _, value in target_.items():
            _check(value)

    def list_search(target_):
        for value in target_:
            _check(value)

    if type(target) is collections.defaultdict:
        target = dict(target)

    if type(target) is dict:
        dict_search(target)
    else:
        list_search(target)

    return count
