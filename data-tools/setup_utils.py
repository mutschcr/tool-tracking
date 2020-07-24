from pathlib import Path


class Bunch(dict):
    """
    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            print(f"[WARN] Unknown key '{key}' return empty string!")
            return ""


def get_meta_info_from_version_file(file: str, pkg_location: str) -> Bunch:
    here = Path(file).resolve().parent
    version_py_file = here / pkg_location / '__version__.py'

    if version_py_file.is_file():
        info = {}
        with open(str(version_py_file), 'rb') as f:
            exec(f.read(), info)

        b = Bunch()
        for key, value in info.items():
            b[key.strip("_")] = value

        return b

    else:
        raise ValueError(f"Could not find version file at '{version_py_file}'")
