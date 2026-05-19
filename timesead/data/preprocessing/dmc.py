from enum import Enum

class DMCTask(Enum):
    """
    Enum representing all DMC tasks.
    Used to select which environment/task dataset to load.
    """
    ACROBOT_SWINGUP = 0
    CHEETAH_RUN = 1
    FINGER_TURN_HARD = 2
    PENDULUM_SWINGUP = 3
    QUADRUPED_RUN = 4
    REACHER_HARD = 5
    CARTPOLE_BALANCE = 6
    CUP_CATCH = 7
    HOPPER_STAND = 8
    POINTMASS_EASY = 9
    QUADRUPED_WALK = 10
    WALKER_WALK = 11
    CARTPOLE_SWINGUP = 12
    FINGER_SPIN = 13
    MANIPULATOR_BRING_BALL = 14
    POINTMASS_HARD = 15
    REACHER_EASY = 16

