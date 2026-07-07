from enum import Enum

from .control_task_common import (
    ANOMALY_LABEL,
    ANOMAL_STATISTICS_FILE,
    META_DATASET_FILE,
    NORMAL_LABEL,
    NORMAL_STATISTICS_FILE,
    construct_meta_data,
    get_stats,
    obtain_meta_data,
    parse_meta_data,
)


class DMCTask(Enum):
    """
    Enum representing all DMC tasks.
    Used to select which environment/task dataset to load.
    """
    CARTPOLE_BALANCE = 0
    CARTPOLE_SWINGUP = 1
    CHEETAH_RUN = 2
    HOPPER_STAND = 3
    FINGER_SPIN = 4
    QUADRUPED_RUN = 5
    QUADRUPED_WALK = 6
    WALKER_WALK = 7
