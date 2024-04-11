import os
from PIL import Image

from modules import shared

import sd_bmab
from sd_bmab import util
from sd_bmab.util import debug_print
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


class LineartNoise(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.controlnet_opt = {}
		self.enabled = False
		self.with_refiner = False
		self.noise_enabled = False
		self.noise_strength = 0.4
		self.noise_begin = 0.1
		self.noise_end = 0.9

	@staticmethod
	def with_refiner(context: Context):
		controlnet_opt = context.args.get('module_config', {}).get('controlnet', {})
		enabled = controlnet_opt.get('enabled', False)
		with_refiner = controlnet_opt.get('with_refiner', False)
		debug_print('with refiner', enabled, with_refiner)
		return enabled and with_refiner

	def preprocess(self, context: Context, image: Image):
		self.controlnet_opt = context.args.get('module_config', {}).get('controlnet', {})
		self.enabled = self.controlnet_opt.get('enabled', False)
		self.with_refiner = self.controlnet_opt.get('with_refiner', False)
		self.noise_enabled = self.controlnet_opt.get('noise', False)
		self.noise_strength = self.controlnet_opt.get('noise_strength', 0.4)
		self.noise_begin = self.controlnet_opt.get('noise_begin', 0.1)
		self.noise_end = self.controlnet_opt.get('noise_end', 0.9)

		debug_print('Noise', context.is_refiner_context(), context.with_refiner(), self.with_refiner)
		if context.is_refiner_context():
			return self.enabled and self.with_refiner
		elif context.with_refiner() and self.with_refiner:
			return False
		return self.enabled

	@staticmethod
	def get_noise_args(image, weight, begin, end):
		cn_args = {
			'input_image': util.b64_encoding(image),
			'model': shared.opts.bmab_cn_lineart,
			'weight': weight,
			"guidance_start": begin,
			"guidance_end": end,
			'resize_mode': 'Just Resize',
			'pixel_perfect': False,
			'control_mode': 'ControlNet is more important',
			'processor_res': 512,
			'threshold_a': 64,
			'threshold_b': 64,
		}
		return cn_args

	def get_controlnet_args(self, context):
		img = util.generate_noise(context.sdprocessing.seed, context.sdprocessing.width, context.sdprocessing.height)
		noise = img.convert('L').convert('RGB')
		return self.get_noise_args(noise, self.noise_strength, self.noise_begin, self.noise_end)

	def get_noise_from_cache(self, seed, width, height):
		path = os.path.dirname(sd_bmab.__file__)
		cache_dir = os.path.normpath(os.path.join(path, '..', 'cache'))
		if not os.path.isdir(cache_dir):
			os.mkdir(cache_dir)
		cache_file = os.path.join(cache_dir, f'noise_{width}_{height}.png')
		if os.path.isfile(cache_file):
			img = Image.open(cache_file)
			noise = img.convert('L').convert('RGB')
			return noise
		img = util.generate_noise(seed, width, height)
		img.save(cache_file)
		return img

	def process(self, context: Context, image: Image):
		context.add_generation_param('BMAB_controlnet_option', util.dict_to_str(self.controlnet_opt))

		debug_print('Seed', context.sdprocessing.seed)
		debug_print('AllSeeds', context.sdprocessing.all_seeds)

		cn_args = util.get_cn_args(context.sdprocessing)
		debug_print('ControlNet', cn_args)
		for num in range(*cn_args):
			obj = context.sdprocessing.script_args[num]
			if hasattr(obj, 'enabled') and obj.enabled:
				context.controlnet_count += 1
			elif isinstance(obj, dict) and 'module' in obj and obj['enabled']:
				context.controlnet_count += 1
			else:
				break

		debug_print('noise enabled.', self.noise_strength)
		context.add_generation_param('BMAB controlnet mode', 'lineart')
		context.add_generation_param('BMAB noise strength', self.noise_strength)
		context.add_generation_param('BMAB noise begin', self.noise_begin)
		context.add_generation_param('BMAB noise end', self.noise_end)

		img = self.get_noise_from_cache(context.sdprocessing.seed, context.sdprocessing.width, context.sdprocessing.height)
		cn_op_arg = self.get_noise_args(img, self.noise_strength, self.noise_begin, self.noise_end)
		idx = cn_args[0] + context.controlnet_count
		context.controlnet_count += 1
		sc_args = list(context.sdprocessing.script_args)
		sc_args[idx] = cn_op_arg
		context.sdprocessing.script_args = tuple(sc_args)

	def postprocess(self, context: Context, image: Image):
		pass


