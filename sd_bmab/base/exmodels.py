import os
import sys
import glob
import importlib.util

from PIL import Image

import sd_bmab
from sd_bmab import constants
from sd_bmab.util import debug_print




def get_external_model(name):
	if name == 'None':
		return None
	debug_print('External', name)
	path = os.path.dirname(sd_bmab.__file__)
	path = os.path.normpath(os.path.join(path, '../exmodels'))
	filter_path = f'{path}/{name}.py'
	mod = load_module(filter_path, name)
	return mod


def load_module(file_name, module_name):
	spec = importlib.util.spec_from_file_location(module_name, file_name)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module
