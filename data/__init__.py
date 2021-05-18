import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]
