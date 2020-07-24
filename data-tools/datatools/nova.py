# --- built-in ---
from __future__ import annotations
from typing import Dict, Union, List, Tuple, Set, Optional
import xml.etree.ElementTree as ElementTree
import xml.dom.minidom
from pathlib import Path

# --- third-party ---
import numpy as np
import pandas as pd

# --- own --
from .utils import convert2path


class Item:
    """
    Annotation item.

    Parameters
    ----------
    class_name
        Class name.
    label
        Label (integer ID).
    color
        Color associated with this item.

    """
    def __init__(self, class_name: str, label: int, color: str = None):
        self.class_name = class_name
        self.label = label
        self.color = color

    def __eq__(self, other) -> bool:
        condition = True
        for attrib in ["class_name", "label", "color"]:
            condition = condition and getattr(self, attrib) == getattr(other, attrib)
        return condition

    def __hash__(self):
        return hash(self.class_name) + hash(self.label)


class DiscreteAnnotation:
    """
    Representation of a discrete annotation, as defined by `NOVA <https://rawgit.com/hcmlab/nova/master/docs/>`_.

    Actually, it is stored in two separate files: a header file (.annotation) and a data file (.annotation~)
    with the same file name. The header contains information about the scheme, while the data file holds the actual
    annotation data in ASCII (default) format.

    A discrete scheme consists of a list of predefined labels represented by a name and an unique identifier (id).
    Optionally, colour values can be assigned to the background and each label.

    The data file contains of one segment per line, which are structured as follows:
    <beginning of a segment in seconds> ; <end of a segment in seconds> ;
    <label id as specified in the header> ; <confidence value>

    Parameters
    ----------
    scheme_name
        Name of the annotation scheme.
    label_mapping
        Mapping of labels (IDs as integer) and their corresponding class names.
    annotator
        Name of the person which created this annotation file.

    """
    _header_file_extension = ".annotation"  # = xml
    _data_file_extension = ".annotation~"  # = csv
    _columns = ("beginning of a segment in seconds", "end of a segment in seconds", "label id", "confidence value")
    _version = "3"
    _csv_delimiter = ";"
    # define colors (first color is grey, then the default matplotlib colors are used)
    _colors = ['#FFD3D3D3', '#FF1F77B4', '#FFFF7F0E', '#FF2CA02C', '#FFD62728', '#FF9467BD', '#FF8C564B', '#FFE377C2',
               '#FF7F7F7F', '#FFBCBD22', '#FF17BECF']

    def __init__(self, scheme_name: str, label_mapping: Dict[int, str], annotator: str = "Annotator"):
        self._scheme_name: str = scheme_name
        self._annotator = annotator

        self._items: Set[Item] = {Item(class_name, label, self._colors[color_id])
                                  for color_id, (label, class_name) in enumerate(label_mapping.items())}

        self._data: pd.DataFrame = pd.DataFrame(columns=self._columns)

    def __eq__(self, other) -> bool:
        condition = True
        for attrib in ["_scheme_name", "_annotator", "_items"]:
            condition = condition and getattr(self, attrib) == getattr(other, attrib)
        condition = condition and np.array_equal(self._data.values, other._data.values)
        return condition

    @property
    def data(self) -> pd.DataFrame:
        """
        Table with the actual annotation information.
        """
        return self._data.astype({"label id": "int32"})

    @property
    def items(self) -> Set[Item]:
        """
        Items defining the annotation scheme.
        """
        return self._items

    # --- public ---

    def define_label_mapping(self, mapping: Dict[int, str]):
        """
        Overwrite existing label mapping. Consistency with the data is not checked, so use with care!
        """
        self._items = {Item(class_name, label) for label, class_name in mapping.items()}

    def update_label_mapping(self, mapping: Optional[Dict[int, str]]):
        """
        Update existing label mapping by remapping labels in data according to the passed label mapping.
        """

        if mapping is None:
            return self

        label_mapping = {}

        # Update ITEMS
        for label, class_name in mapping.items():
            for item in self._items:
                if item.class_name == class_name:
                    label_mapping.update({item.label: int(label)})
                    item.label = int(label)

        label_mapping.update({-1: -1})

        # Update DATA
        for idx, row in self._data.iterrows():
            self._data.at[idx, "label id"] = label_mapping[int(row["label id"])]

        return self

    def add_annotation(self, row: Union[Tuple[float, float, int], Tuple[float, float, int, float]]):
        """
        Add a new annotated segment. Passing the confidence (default=1) is optional.
        """
        assert 2 < len(row) <= 4, f"Invalid annotation. Should be {self._columns}"
        if len(row) == 3:
            row = list(row)
            row.append(1.0)
        self._data = self._data.append(row)

    def add_annotations(self, data: np.ndarray):
        """
        Add multiple annotated segments. Passing the confidence (default=1) is optional.
        """
        assert 2 < data.shape[1] <= 4, f"Invalid annotation. Should be {self._columns}"

        if data.shape[1] == 3:
            data_full = np.ones(shape=(data.shape[0], 4))
            data_full[:, :3] = data
        else:
            data_full = data

        new_data = pd.DataFrame(data=data_full, columns=self._columns)
        self._data = self._data.append(new_data, ignore_index=True)

    def add_annotations_from_prediction(self, y: np.ndarray, t: np.ndarray):
        """
        Add annotations based on the prediction of a machine learning pipeline.
        """
        self.add_annotations(self._pred_to_discrete(y, t))

    @convert2path
    def save(self, output: Union[str, Path]):
        """
        Save this annotation as two separate files with the passed name: a header file (.annotation)
        and a data file (.annotation~) with the same file name.
        """
        assert output.parent.is_dir()

        # HEADER FILE
        with open(output.with_suffix(".annotation"), "w") as header_file:
            header_file.write(self._create_header())

        # DATA FILE
        self._data["label id"] = self._data["label id"].astype(int)

        self._data.to_csv(output.with_suffix(".annotation~"), sep=";", header=False, index=False)

    @classmethod
    @convert2path
    def from_files(cls, filepath: Union[str, Path]) -> Optional[DiscreteAnnotation]:
        """
        Creates an DiscreteAnnotation object from a annotation file pair.

        Parameters
        ----------
        filepath
            Name and path of the NOVA annotation file.

        Returns
        -------
        annotation
            DiscreteAnnotation object with the annotations from the input files.
        """

        # check for correct file extension
        if filepath.suffix != cls._header_file_extension and str(filepath) != cls._header_file_extension:
            raise TypeError("Select a valid annotation file.")

        # reading segments from data file
        try:
            data = np.loadtxt(fname=str(filepath.with_suffix(cls._data_file_extension)),
                              delimiter=cls._csv_delimiter)
        except OSError:
            print(f"Missing annotation data: {filepath.with_suffix(cls._data_file_extension)}", 'error')
            return None

        # parse xml annotation file
        tree = xml.etree.ElementTree.parse(filepath)
        root = tree.getroot()

        scheme = [child for child in root if child.tag == 'scheme'][0]
        meta = [child for child in root if child.tag == 'meta'][0]

        if scheme.attrib['type'] != 'DISCRETE':
            raise TypeError(f"Annotation scheme is no DISCRETE but {scheme.attrib['type']}.")

        annotation = cls(scheme.attrib['name'], {}, meta.attrib['annotator'])
        annotation._items = {
            Item(item.attrib['name'], int(item.attrib['id']), item.attrib.get('color', None)) for item in scheme
        }

        annotation.add_annotations(data)

        return annotation

    # --- private ---

    def _create_header(self) -> str:
        """
        Create a string representation of the xml header file.
        """
        header = ElementTree.Element("annotation", attrib={"ssi-v": self._version})
        _ = ElementTree.SubElement(header, "info", attrib={"ftype": "ASCII", "size": "71"})
        _ = ElementTree.SubElement(header, "meta", attrib={"annotator": self._annotator})
        scheme = ElementTree.SubElement(
            header, "scheme", attrib={"name": self._scheme_name, "type": "DISCRETE", "color": "#FFFFFFFF"}
        )

        for item in sorted(self._items, key=lambda item_: item_.label):
            ElementTree.SubElement(scheme, "item", attrib={
                "name": item.class_name, "id": str(item.label), "color": item.color})

        dom = xml.dom.minidom.parseString(ElementTree.tostring(header))

        return dom.toprettyxml()

    @staticmethod
    def _sections_start_stop_indices(y: np.ndarray) -> List[Tuple[int, int]]:
        """
        Return the start and stop indices for every section.
        A section describes an interval of consecutive identical values within a vector.

        Parameters
        ----------
        y : array-like
            Multi label encoded target vector.

        Returns
        -------
        indices
            Dictionary with the keys being the unique labels of interest and the values being lists of tuples.
            One tuple contains the start and the stop index of one section.
        """

        lims = list(np.where(np.diff(y) != 0)[0] + 1)
        lims = [0] + lims + [len(y)-1]
        lims_diff = np.diff(lims)

        return [(lims[idx], lims[idx + 1]) for idx in range(len(lims_diff))]

    @classmethod
    def _pred_to_discrete(cls, y: np.ndarray, t: np.ndarray) -> np.ndarray:

        assert y.shape == t.shape, "Shapes of predictions and time are not matching"
        assert y.ndim == 1 and t.ndim == 1, "Predictions and time has to be 1d"
        time = t - t[0]

        section_indices = cls._sections_start_stop_indices(y)
        data = np.zeros(shape=(len(section_indices), 4))

        for i, (start, stop) in enumerate(section_indices):
            data[i, :] = (time[start], time[stop], y[start], 1)

        return data
