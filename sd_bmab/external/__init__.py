import os
import sys
import importlib.util


def load_external_module(module, name):
	path = os.path.dirname(__file__)
	path = os.path.normpath(os.path.join(path, f'{module}/{name}.py'))
	return load_module(path, 'module')


def load_module(file_name, module_name):
	spec = importlib.util.spec_from_file_location(module_name, file_name)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


class ModuleAutoLoader(object):

	def __init__(self, module, name) -> None:
		super().__init__()
		self.mod = load_external_module(module, name)

	def __enter__(self):
		return self.mod

	def __exit__(self, *args, **kwargs):
		self.mod.release()
