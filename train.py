import os.path as osp

import ffhq_degradation_dataset  # noqa: F401
import gfpgan_model  # noqa: F401
import gfpganv1_arch  # noqa: F401
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
