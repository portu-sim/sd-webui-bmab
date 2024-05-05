from copy import copy

from modules import shared
from modules import processing
from modules import img2img
from modules.processing import Processed

from sd_bmab.util import debug_print, get_cn_args


controlnet_args = (0, 0)


class FakeControlNet:
	def __init__(self, p, cn_enabled=False) -> None:
		super().__init__()
		self.process = p
		self.all_prompts = None
		self.all_negative_prompts = None
		self.enabled = self.is_controlnet_enabled() if cn_enabled else False
		debug_print('FakeControlNet', self.enabled, cn_enabled)

	def __enter__(self):
		if self.enabled:
			dummy = Processed(self.process, [], self.process.seed, "")
			self.all_prompts = copy(self.process.all_prompts)
			self.all_negative_prompts = copy(self.process.all_negative_prompts)
			self.process.scripts.postprocess(copy(self.process), dummy)
			for idx, obj in enumerate(self.process.script_args):
				if 'controlnet' in obj.__class__.__name__.lower():
					if hasattr(obj, 'enabled') and obj.enabled:
						debug_print('Use controlnet True')
				elif isinstance(obj, dict) and 'model' in obj and obj['enabled']:
					obj['enabled'] = False

	def __exit__(self, *args, **kwargs):
		if self.enabled:
			copy_p = copy(self.process)
			self.process.all_prompts = self.all_prompts
			self.process.all_negative_prompts = self.all_negative_prompts
			if hasattr(self.process.scripts, "before_process"):
				self.process.scripts.before_process(copy_p)
			self.process.scripts.process(copy_p)

	def is_controlnet_enabled(self):
		global controlnet_args
		for idx in range(controlnet_args[0], controlnet_args[1]):
			obj = self.process.script_args[idx]
			if isinstance(obj, dict):
				return True
			if 'controlnet' in obj.__class__.__name__.lower():
				if hasattr(obj, 'enabled'):
					return True
		return False


class PreventControlNet(FakeControlNet):
	process_images_inner = processing.process_images_inner
	process_batch = img2img.process_batch

	def __init__(self, p, cn_enabled=False) -> None:
		super().__init__(p, cn_enabled)
		self._process_images_inner = processing.process_images_inner
		self._process_batch = img2img.process_batch
		self.allow_script_control = None
		self.p = p
		self.all_prompts = copy(p.all_prompts)
		self.all_negative_prompts = copy(p.all_negative_prompts)

	def is_controlnet_used(self):
		if not self.p.script_args:
			return False

		for idx, obj in enumerate(self.p.script_args):
			if 'controlnet' in obj.__class__.__name__.lower():
				if hasattr(obj, 'enabled') and obj.enabled:
					debug_print('Use controlnet True')
					return True
			elif isinstance(obj, dict) and 'module' in obj and obj['enabled']:
				debug_print('Use controlnet True')
				return True

		debug_print('Use controlnet False')
		return False

	def __enter__(self):
		model = self.p.sd_model.model.diffusion_model
		if hasattr(model, '_original_forward'):
			model._old_forward = self.p.sd_model.model.diffusion_model.forward
			model.forward = getattr(model, '_original_forward')

		processing.process_images_inner = PreventControlNet.process_images_inner
		img2img.process_batch = PreventControlNet.process_batch
		if 'control_net_allow_script_control' in shared.opts.data:
			self.allow_script_control = shared.opts.data['control_net_allow_script_control']
			shared.opts.data['control_net_allow_script_control'] = True
		self.multiple_tqdm = shared.opts.data.get('multiple_tqdm', True)
		shared.opts.data['multiple_tqdm'] = False
		super().__enter__()

	def __exit__(self, *args, **kwargs):
		processing.process_images_inner = self._process_images_inner
		img2img.process_batch = self._process_batch
		if 'control_net_allow_script_control' in shared.opts.data:
			shared.opts.data['control_net_allow_script_control'] = self.allow_script_control
		shared.opts.data['multiple_tqdm'] = self.multiple_tqdm
		model = self.p.sd_model.model.diffusion_model
		if hasattr(model, '_original_forward') and hasattr(model, '_old_forward'):
			self.p.sd_model.model.diffusion_model.forward = model._old_forward
		super().__exit__(*args, **kwargs)


def update_controlnet_args(p):
	cn_arg_index = []
	for idx, obj in enumerate(p.script_args):
		if 'controlnet' in obj.__class__.__name__.lower():
			cn_arg_index.append(idx)
	global controlnet_args
	controlnet_args = (cn_arg_index[0], cn_arg_index[-1])


def get_controlnet_index(p):
	cn_args = get_cn_args(p)
	controlnet_count = 0
	for num in range(*cn_args):
		obj = p.script_args[num]
		if hasattr(obj, 'enabled') and obj.enabled:
			controlnet_count += 1
		elif isinstance(obj, dict) and 'model' in obj and obj['enabled']:
			controlnet_count += 1
		else:
			break
	return cn_args[0] + controlnet_count
