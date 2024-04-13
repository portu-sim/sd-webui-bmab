import os
import sys
import glob
import importlib.util

from PIL import Image

import sd_bmab
from sd_bmab import constants
from sd_bmab.util import debug_print
from sd_bmab import controlnet


filters = [constants.filter_default]


class BaseFilter(object):

	def __init__(self) -> None:
		super().__init__()

	def configurations(self):
		return {}

	def is_controlnet_required(self):
		return False

	def preprocess(self, context, image, *args, **kwargs):
		pass

	def process(self, context, base: Image, processed: Image, *args, **kwargs):
		return processed

	def postprocess(self, context, *args, **kwargs):
		pass

	def finalprocess(self, context, *args, **kwargs):
		pass


class NoneFilter(BaseFilter):

	def process_filter(self, context, base: Image, processed: Image, *args, **kwargs):
		return processed


def reload_filters():
	global filters
	filters = [constants.filter_default]

	path = os.path.dirname(sd_bmab.__file__)
	path = os.path.normpath(os.path.join(path, '../filter'))
	files = sorted(glob.glob(f'{path}/*.py'))
	for file in files:
		fname = os.path.splitext(os.path.basename(file))[0]
		filters.append(fname)


def get_filter(name):
	if name == 'None':
		return NoneFilter()
	debug_print('Filter', name)
	path = os.path.dirname(sd_bmab.__file__)
	path = os.path.normpath(os.path.join(path, '../filter'))
	filter_path = f'{path}/{name}.py'
	mod = load_module(filter_path, 'filter')
	return eval(f'mod.Filter()')


def load_module(file_name, module_name):
	spec = importlib.util.spec_from_file_location(module_name, file_name)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def preprocess_filter(bmab_filter, context, image, *args, **kwargs):
	bmab_filter.preprocess(context, image, *args, **kwargs)


def process_filter(bmab_filter, context, base: Image, processed: Image, *args, **kwargs):
	return bmab_filter.process(context, base, processed, *args, **kwargs)


def postprocess_filter(bmab_filter, context, *args, **kwargs):
	bmab_filter.postprocess(context, *args, **kwargs)


def finalprocess_filter(bmab_filter, context, *args, **kwargs):
	bmab_filter.finalprocess(context, *args, **kwargs)

