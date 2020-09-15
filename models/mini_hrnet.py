import torch

from .lib.config import cfg
from .lib.config import update_config
from .lib.core.loss import JointsMSELoss
from .lib.utils.utils import create_logger
from .lib.models.pose_hrnet import get_pose_net

from yacs.config import CfgNode as CN


def get_pretrained_model():
    args = CN()
    args.cfg = '/data/usr/yikanchen/deep-high-resolution-net.pytorch/experiments/coco/hrnet/mini_128x64_adam_lr1e-3.yaml'
    args.opts = ''
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    update_config(cfg, args)

    model = get_pose_net(cfg, is_train=False)
    model.load_state_dict(torch.load(
        '/data/usr/yikanchen/deep-high-resolution-net.pytorch/output/coco/pose_hrnet/mini_128x64_adam_lr1e-3/model_best.pth'))
    model.eval()

    return model
