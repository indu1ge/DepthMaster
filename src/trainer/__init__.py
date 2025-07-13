# Author: Bingxin Ke
# Last modified: 2024-05-17

from .trainer_s1 import DepthMasterTrainerS1
from .trainer_s2 import DepthMasterTrainerS2


trainer_cls_name_dict = {
    "DepthMasterTrainerS1": DepthMasterTrainerS1,
    "DepthMasterTrainerS2": DepthMasterTrainerS2,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
