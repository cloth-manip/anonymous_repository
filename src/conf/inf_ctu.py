import time as _time
import torch as _torch
import socket as _socket

#from src.networks import OldWholeNetwork as _net
#from src.networks import WholeNetwork as _net
from src.lm_networks import LandmarkExpNetwork as _net
from src.lm_networks import LandmarkBranchUpsample as _lm_branch
from src.utils import InferenceEvaluator as _evaluator
from src.utils import LandmarkEvaluator as _lmevaluator


_hostname = str(_socket.gethostname())

_name = 'inference'
_time = _time.strftime('%m-%d_%H:%M:%S', _time.localtime())


# Dataset
#gaussian_R = 8
DATASET_PROC_METHOD_INF = 'BBOXRESIZE'
SWITCH_LEFT_RIGHT = True
########

# Network
USE_NET = _net
LM_SELECT_VGG = 'conv4_3'
LM_SELECT_VGG_SIZE = 28
LM_SELECT_VGG_CHANNEL = 512
LM_BRANCH = _lm_branch
EVALUATOR = _lmevaluator

INIT_MODEL = './lm_ctu_r_rr'#'./ctu_lm_ew_a_50_s_10'
USE_IORN = True

#INIT_MODEL = './DF_whole_elastic_IORN_sig_16'
#INIT_MODEL = './DF_whole_normal_Liu'

#INIT_MODEL = './DF_whole_rot_IORN_attention'
#INIT_MODEL = './DF_whole_normal_Liu'

#INIT_MODEL = './DF_whole_elastic_IORN_noFreeze'
#INIT_MODEL = './WHOLE_DF_normal_IORN'
FREEZE_LM_NETWORK = True
SWITCH_LEFT_RIGHT = True
SE_REDUCTION = 16 
INF_BATCH_SIZE = 64
#################

INF_DIR = 'runs/%s/' % _name + _time
TRAIN_DIR = 'runs/%s/' % _name + _time
MODEL_NAME = 'vgg16.pkl'


_dataset_folder = 'CTU/'
#_dataset_folder = 'robot/'
if _hostname == 'ThinkPad-X1-Yoga':
    base_path = '/home/zieglert/ETH/SA-FL/data/' + _dataset_folder
elif _hostname == 'mordor':
    base_path = '/home/mwelle/WACV2020/data/' + _dataset_folder
else:
    base_path = '/cluster/scratch/zieglert/' + _dataset_folder

device = _torch.device('cuda:0' if _torch.cuda.is_available() else 'cpu')

lm2name = ['L.Col', 'R.Col', 'L.Sle', 'R.Sle', 'L.Wai', 'R.Wai', 'L.Hem', 'R.Hem']
attrtype2name = {1: 'texture', 2: 'fabric', 3: 'shape', 4: 'part', 5: 'style'}

USE_CSV = 'info_3.csv'
#USE_CSV = 'info.csv'
USE_CSV = 'new_info.csv'

LM_TRAIN_USE = 'vis'
LM_EVAL_USE = 'vis'


