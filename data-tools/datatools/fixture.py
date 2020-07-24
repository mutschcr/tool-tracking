# --- build-in ---
from pathlib import Path
import tempfile
import json

# --- third-party ---
import numpy as np

# --- own ---
from .data import DataCSV
from .nova import DiscreteAnnotation


num_features = 3
num_samples = 50
num_classes = 2
sr = 100
time = np.arange(0, num_samples/sr, 1/sr)

np.random.seed(42)
X = np.random.rand(num_samples, num_features)
y = np.random.randint(num_classes, size=num_samples)

classes = {0: "no_action", 1: "action"}
feature_names = ["time"] + [f"feature{i}" for i in range(num_features)]


def create_data():
    df = DataCSV(
        data=np.c_[time, X],
        columns=feature_names,
        sampling_rate=100,
        has_y_column=False,
        id_='00'
    )

    df.set_y(y)

    anno = DiscreteAnnotation("test scheme", label_mapping=classes)
    anno.add_annotations_from_prediction(y, time)

    return df, anno


def create_info_dict(n_measurements):
    info_dict = dict()
    info_dict["date"] = "01/01/2020"
    info_dict["tool"] = "test_tool"
    info_dict["model"] = "model xyz"
    info_dict["work_piece"] = "work piece xyz"
    # add measurement specific information
    info_dict["measurements"] = list()
    for n in range(1, n_measurements + 1):
        measurement_dict = dict()
        measurement_dict["id"] = f"0{n}"
        measurement_dict["test_person"] = "xyz"
        measurement_dict["time"] = f"0{n}:00"
        measurement_dict["module"] = "module xyz"
        measurement_dict["model_id"] = "01"
        # add tool and measurement specific information
        measurement_dict["tool_settings"] = dict()
        measurement_dict["tool_settings"][f"tool_setting_1"] = 0
        measurement_dict["tool_settings"][f"tool_setting_2"] = 1

        info_dict['measurements'].append(measurement_dict)

    return info_dict


def setup_dummy_data(n=1):
    assert n < 10, "The number of test measurements must be smaller than 10."

    tmpdir = tempfile.mkdtemp()

    data_dir = Path(tmpdir) / 'test_tool' / 'test-00-20200202'
    data_dir.mkdir(parents=True)

    info_dict = create_info_dict(n_measurements=n)
    with open(data_dir / 'info.json', 'w') as write_file:
        json.dump(info_dict, write_file, indent=4)

    df, anno = create_data()
    for i in range(1, n + 1):
        df.to_csv(data_dir / f'ACC-0{i}-100.csv', sep=';', index=False)
        df.to_csv(data_dir / f'GYR-0{i}-100.csv', sep=';', index=False)
        anno.save(data_dir / f"data-0{i}")

    return tmpdir
