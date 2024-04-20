import os
import glob
import random

from PIL import Image

from modules import shared

import sd_bmab
from sd_bmab import util
from sd_bmab.util import debug_print
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


class Openpose(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.controlnet_opt = {}
		self.enabled = False
		self.pose_enabled = False
		self.pose_strength = 0.3
		self.pose_begin = 0.0
		self.pose_end = 1.0
		self.pose_face_only = False
		self.pose_selected = 'Random'

	def preprocess(self, context: Context, image: Image):
		self.controlnet_opt = context.args.get('module_config', {}).get('controlnet', {})
		self.enabled = self.controlnet_opt.get('enabled', False)
		self.pose_enabled = self.controlnet_opt.get('pose', False)
		self.pose_strength = self.controlnet_opt.get('pose_strength', self.pose_strength)
		self.pose_begin = self.controlnet_opt.get('pose_begin', self.pose_begin)
		self.pose_end = self.controlnet_opt.get('pose_end', self.pose_end)
		self.pose_face_only = self.controlnet_opt.get('pose_face_only', self.pose_face_only)
		self.pose_selected = self.controlnet_opt.get('pose_selected', self.pose_selected)
		return self.enabled and self.pose_enabled

	def get_openpose_args(self, image):
		cn_args = {
			'input_image': util.b64_encoding(image),
			'module': 'openpose_faceonly' if self.pose_face_only else 'openpose',
			'model': shared.opts.bmab_cn_openpose,
			'weight': 1,
			"guidance_start": 0,
			"guidance_end": 1,
			'resize_mode': 'Just Resize',
			'pixel_perfect': False,
			'control_mode': 'My prompt is more important',
			'processor_res': 512,
			'threshold_a': 64,
			'threshold_b': 64,
			'hr_option': 'Low res only'
		}
		return cn_args

	def process(self, context: Context, image: Image):
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
				
		debug_print('pose enabled.', self.pose_strength)
		context.add_generation_param('BMAB controlnet pose mode', 'openpose')
		context.add_generation_param('BMAB pose strength', self.pose_strength)
		context.add_generation_param('BMAB pose begin', self.pose_begin)
		context.add_generation_param('BMAB pose end', self.pose_end)

		img = self.load_random_image(context)
		if img is None:
			return

		cn_op_arg = self.get_openpose_args(img)
		idx = cn_args[0] + context.controlnet_count
		context.controlnet_count += 1
		sc_args = list(context.sdprocessing.script_args)
		sc_args[idx] = cn_op_arg
		context.sdprocessing.script_args = tuple(sc_args)

	def postprocess(self, context: Context, image: Image):
		pass

	def load_random_image(self, context):
		path = os.path.dirname(sd_bmab.__file__)
		path = os.path.normpath(os.path.join(path, '../pose'))
		if os.path.exists(path) and os.path.isdir(path):
			file_mask = f'{path}/*.*'
			files = glob.glob(file_mask)
			if not files:
				debug_print(f'Not found pose files in {path}')
				return None
			if self.pose_selected == 'Random':
				file = random.choice(files)
				debug_print(f'Random pose {file}')
				return self.get_cache(context, file)
			else:
				img_name = f'{path}/{self.pose_selected}'
				return self.get_cache(context, img_name)
		debug_print(f'Not found directory {path}')
		return None

	def get_cache(self, context, file):
		if self.pose_face_only:
			path = os.path.dirname(sd_bmab.__file__)
			path = os.path.normpath(os.path.join(path, '../cache'))
			if os.path.exists(path) and os.path.isdir(path):
				b = os.path.basename(file)
				file_mask = f'{path}/pose_face_{b}'
				if os.path.exists(file_mask):
					return Image.open(file_mask)
		return Image.open(file)


	@staticmethod
	def list_pose():
		path = os.path.dirname(sd_bmab.__file__)
		path = os.path.normpath(os.path.join(path, '../pose'))
		if os.path.exists(path) and os.path.isdir(path):
			file_mask = f'{path}/*.*'
			files = glob.glob(file_mask)
			return [os.path.basename(f) for f in files]
		debug_print(f'Not found directory {path}')
		return []

	@staticmethod
	def get_pose(f):
		if f == 'Random':
			return Image.new('RGB', (512, 512), 0)
		path = os.path.dirname(sd_bmab.__file__)
		path = os.path.normpath(os.path.join(path, '../pose'))
		if os.path.exists(path) and os.path.isdir(path):
			img_name = f'{path}/{f}'
			return Image.open(img_name)
		return Image.new('RGB', (512, 512), 0)
