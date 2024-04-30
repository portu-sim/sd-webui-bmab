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

weight_type = [
	('normal', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
	('ease in', [1.0, 0.94, 0.88, 0.82, 0.76, 0.7, 0.64, 0.58, 0.53, 0.47, 0.41, 0.35, 0.29, 0.23, 0.17, 0.11]),
	('ease out', [0.05, 0.11, 0.17, 0.23, 0.29, 0.35, 0.41, 0.47, 0.53, 0.58, 0.64, 0.7, 0.76, 0.82, 0.88, 0.94]),
	('ease in-out', [0.05, 0.17, 0.29, 0.41, 0.53, 0.64, 0.76, 0.88, 1.0, 0.88, 0.76, 0.64, 0.53, 0.41, 0.29, 0.17]),
	('reverse in-out', [1.0, 0.88, 0.76, 0.64, 0.53, 0.41, 0.29, 0.17, 0.05, 0.17, 0.29, 0.41, 0.53, 0.64, 0.76, 0.88]),
	('weak input', [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
	('weak output', [1, 1, 1, 1, 1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
	('weak middle', [1, 1, 1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
	('strong middle', [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
	('style transfer', [1, 1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 1]),
	('composition', [0.0, 0.0, 0.0, 0.0, 0.25, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
	('strong style transfer', [1, 1, 1, 1, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
]


class IpAdapter(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.controlnet_opt = {}
		self.enabled = False
		self.ipadapter_enabled = False
		self.ipadapter_strength = 0.3
		self.ipadapter_begin = 0.0
		self.ipadapter_end = 1.0
		self.ipadapter_face_only = False
		self.ipadapter_selected = 'Random'
		self.ipadapter_weight_type = 'normal'

	def preprocess(self, context: Context, image: Image):
		self.controlnet_opt = context.args.get('module_config', {}).get('controlnet', {})
		self.enabled = self.controlnet_opt.get('enabled', False)
		self.ipadapter_enabled = self.controlnet_opt.get('ipadapter', False)
		self.ipadapter_strength = self.controlnet_opt.get('ipadapter_strength', self.ipadapter_strength)
		self.ipadapter_begin = self.controlnet_opt.get('ipadapter_begin', self.ipadapter_begin)
		self.ipadapter_end = self.controlnet_opt.get('ipadapter_end', self.ipadapter_end)
		self.ipadapter_face_only = self.controlnet_opt.get('ipadapter_face_only', self.ipadapter_face_only)
		self.ipadapter_selected = self.controlnet_opt.get('ipadapter_selected', self.ipadapter_selected)
		self.ipadapter_weight_type = self.controlnet_opt.get('ipadapter_weight_type', self.ipadapter_weight_type)
		return self.enabled and self.ipadapter_enabled

	def get_openipadapter_args(self, image):
		cn_args = {
			'enabled': True,
			'input_image': util.b64_encoding(image),
			'module': 'ip-adapter-auto',
			'model': shared.opts.bmab_cn_ipadapter,
			'weight': self.ipadapter_strength,
			"guidance_start": self.ipadapter_begin,
			"guidance_end": self.ipadapter_end,
			'resize_mode': 'Just Resize',
			'pixel_perfect': False,
			'control_mode': 'My prompt is more important',
			'processor_res': 1024,
			'threshold_a': 64,
			'threshold_b': 64,
			'hr_option': 'Low res only',
			'advanced_weighting': self.get_weight_type(self.ipadapter_weight_type, self.ipadapter_strength)
		}
		return cn_args

	def process(self, context: Context, image: Image):
		debug_print('Seed', context.sdprocessing.seed)
		debug_print('AllSeeds', context.sdprocessing.all_seeds)

		cn_args = util.get_cn_args(context.sdprocessing)
		debug_print('ControlNet', cn_args)
		controlnet_count = 0
		for num in range(*cn_args):
			obj = context.sdprocessing.script_args[num]
			if hasattr(obj, 'enabled') and obj.enabled:
				controlnet_count += 1
			elif isinstance(obj, dict) and obj.get('enabled', False):
				controlnet_count += 1
			else:
				break
				
		context.add_generation_param('BMAB controlnet ipadapter mode', 'ip-adapter-auto')
		context.add_generation_param('BMAB ipadapter strength', self.ipadapter_strength)
		context.add_generation_param('BMAB ipadapter begin', self.ipadapter_begin)
		context.add_generation_param('BMAB ipadapter end', self.ipadapter_end)
		context.add_generation_param('BMAB ipadapter image', self.ipadapter_selected)
		context.add_generation_param('BMAB ipadapter weight type', self.ipadapter_weight_type)

		img = self.load_random_image(context)
		if img is None:
			return

		cn_op_arg = self.get_openipadapter_args(img)
		idx = cn_args[0] + controlnet_count
		debug_print(f'IpAdapter Enabled {idx}')
		sc_args = list(context.sdprocessing.script_args)
		sc_args[idx] = cn_op_arg
		context.sdprocessing.script_args = tuple(sc_args)

	def postprocess(self, context: Context, image: Image):
		pass

	def load_random_image(self, context):
		path = os.path.dirname(sd_bmab.__file__)
		path = os.path.normpath(os.path.join(path, '../resources/ipadapter'))
		if os.path.exists(path) and os.path.isdir(path):
			file_mask = f'{path}/*.*'
			files = [os.path.basename(f) for f in glob.glob(file_mask) if not f.endswith('.txt')]
			if not files:
				debug_print(f'Not found ipadapter files in {path}')
				return None
			if self.ipadapter_selected == 'Random':
				file = random.choice(files)
				debug_print(f'Random ipadapter {file}')
				return self.get_image(file)
			else:
				return self.get_image(self.ipadapter_selected)
		debug_print(f'Not found directory {path}')
		return None

	@staticmethod
	def list_images():
		path = os.path.dirname(sd_bmab.__file__)
		path = os.path.normpath(os.path.join(path, '../resources/ipadapter'))
		if os.path.exists(path) and os.path.isdir(path):
			file_mask = f'{path}/*.*'
			files = glob.glob(file_mask)
			return [os.path.basename(f) for f in files if not f.endswith('.txt')]
		debug_print(f'Not found directory {path}')
		return []

	@staticmethod
	def get_image(f):
		if f == 'Random':
			return Image.new('RGB', (512, 512), 0)
		path = os.path.dirname(sd_bmab.__file__)
		path = os.path.normpath(os.path.join(path, '../resources/ipadapter'))
		if os.path.exists(path) and os.path.isdir(path):
			img_name = f'{path}/{f}'
			return Image.open(img_name)
		return Image.new('RGB', (512, 512), 0)

	@staticmethod
	def get_weight_type_list():
		return [wt[0] for wt in weight_type]

	@staticmethod
	def get_weight_type(weight_type_name, weight):
		for wt in weight_type:
			if wt[0] == weight_type_name:
				return [x * weight for x in wt[1]]
		return [x * weight for x in weight_type[0][1]]
