from utils import PathUtil as _PathUtil
from utils import IDUtil as _IDUtil
import torch

ID = _IDUtil.get_random_id_bytime()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PATH SETTINGS
ROOT_PATH = _PathUtil.get_main_folder_path()  # the absolute path of main.py
DATA_FOLDER_PATH = ROOT_PATH + "/../data"
TMPOUT_FOLDER_PATH = ROOT_PATH + "/../temp"
CONFIG_FOLDER_PATH = ROOT_PATH + "/conf"
OUTPUT_FOLDER_PATH = ROOT_PATH + "/../output"

_PathUtil.auto_create_folder_path(TMPOUT_FOLDER_PATH, OUTPUT_FOLDER_PATH)
_PathUtil.check_path_exist(DATA_FOLDER_PATH)

