import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import dataset modules for registry
# scan all the files that end with '_dataset.py' under the data folder
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'gfpgan.data.{file_name}') for file_name in dataset_filenames]
