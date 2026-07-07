from enum import Enum


class MetaWorldTask(Enum):
    """
    Enum representing all Meta World tasks.
    Used to select which environment/task dataset to load.
    """
    ASSEMBLY_V3 = 0
    DISASSEMBLE_V3 = 1
    HAMMER_V3 = 2
    HAND_INSERT_V3 = 3
    PUSH_V3 = 4
    SWEEP_V3 = 5