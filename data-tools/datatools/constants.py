# --- build-in ---
from pathlib import Path

DATA_FILE_EXTENSION = '.csv'
CSV_DELIMITER = ";"

CONFIG_INI = Path(__file__).parent / "config.ini"

GLOBAL_MAPPING = {
    0: "no_action",
    1: "action",
    2: "tightening",
    3: "untightening",
    4: "motor_activity_cw",
    5: "motor_activity_ccw",
    6: "manual_motor_rotation",
    7: "shaking",
    8: "undefined",
    9: "screwing_in",
    10: "screwing_out",
    11: "screwing_air",
    12: "screwing_air_in",
    13: "screwing_air_out",
    14: "tightening_double",
    15: "positioning",
    16: "threading",
    17: "aligning",
    18: "dropping",
    19: "transport_by_hand",
    20: "transport_by_car",
    21: "idle",
    22: "motor_run_down_screw",
    23: "motor_run_down_air",
    24: "untightening_stuck",
    25: "pull_trigger",
    26: "hold_trigger",
    27: "release_trigger",
    28: "pull_trigger_not_removed",
    29: "pull_trigger_air",
    30: "pull_trigger_air_with_rivet",
    31: "pull_trigger_air_with_rivet_not_removed",
    32: "change_nose_piece",
    33: "tightening_pre",
    34: "tightening_click",
    35: "tightening_inter",
    36: "tightening_clack",
    37: "untightening_pre",
    38: "impact"
}
