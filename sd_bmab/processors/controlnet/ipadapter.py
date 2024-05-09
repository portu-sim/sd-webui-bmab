import os
import glob
import random

from PIL import Image

from modules import shared

import sd_bmab
from sd_bmab import util, controlnet
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
			'image': image if isinstance(image, str) and os.path.exists(image) else util.b64_encoding(image.convert('RGB')),
			'module': 'ip-adapter-auto',
			'model': shared.opts.bmab_cn_ipadapter,
			'weight': self.ipadapter_strength,
			"guidance_start": self.ipadapter_begin,
			"guidance_end": self.ipadapter_end,
			'resize_mode': 'Just Resize',
			'pixel_perfect': False,
			'control_mode': 'My prompt is more important',
			'processor_res': 1024,
			'threshold_a': 0.5,
			'threshold_b': 0.5,
			'hr_option': 'Low res only',
			'advanced_weighting': self.get_weight_type(self.ipadapter_weight_type, self.ipadapter_strength)
		}
		return cn_args

	def process(self, context: Context, image: Image):
		context.add_generation_param('BMAB controlnet ipadapter mode', 'ip-adapter-auto')
		context.add_generation_param('BMAB ipadapter strength', self.ipadapter_strength)
		context.add_generation_param('BMAB ipadapter begin', self.ipadapter_begin)
		context.add_generation_param('BMAB ipadapter end', self.ipadapter_end)
		context.add_generation_param('BMAB ipadapter image', self.ipadapter_selected)
		context.add_generation_param('BMAB ipadapter weight type', self.ipadapter_weight_type)

		img = self.load_image(context)
		if img is None:
			return

		index = controlnet.get_controlnet_index(context.sdprocessing)
		cn_op_arg = self.get_openipadapter_args(img)
		debug_print(f'IpAdapter Enabled {index}')
		sc_args = list(context.sdprocessing.script_args)
		sc_args[index] = cn_op_arg
		context.sdprocessing.script_args = tuple(sc_args)

	def postprocess(self, context: Context, image: Image):
		pass

	def load_image(self, context):
		if self.ipadapter_selected == 'Random':
			images = IpAdapter.list_images()
			img = random.choice(images)
			return self.get_image(img)
		else:
			return self.get_image(self.ipadapter_selected)

	@staticmethod
	def list_images():
		root_path = os.path.dirname(sd_bmab.__file__)
		root_path = os.path.normpath(os.path.join(root_path, '../resources/ipadapter'))
		if not os.path.exists(root_path) or not os.path.isdir(root_path):
			return []
		return [os.path.relpath(f, root_path) for f in IpAdapter.list_images_in_dir(root_path)]

	@staticmethod
	def list_images_in_dir(path):
		files = []
		dirs = []
		for file in glob.glob(f'{path}/*'):
			if os.path.isdir(file):
				dirs.append(file)
				continue
			if not file.endswith('.txt'):
				files.append(file)

		files = sorted(files)
		for dir in dirs:
			files.append(dir)
			files.extend(IpAdapter.list_images_in_dir(dir))

		return files

	@staticmethod
	def get_image(f, displayed=False):
		if displayed and (f is None or f == 'Random'):
			return Image.new('RGB', (512, 512))
		root_path = os.path.dirname(sd_bmab.__file__)
		root_path = os.path.normpath(os.path.join(root_path, '../resources/ipadapter'))
		image_path = os.path.join(root_path, f)
		if os.path.isdir(image_path):
			if displayed:
				return Image.new('RGB', (512, 512))
			files = [os.path.relpath(f, root_path) for f in IpAdapter.list_images_in_dir(image_path) if not f.endswith('.txt')]
			if not files:
				debug_print(f'Not found ipadapter files in {image_path}')
				return Image.new('RGB', (512, 512)) if displayed else None
			file = random.choice(files)
			return IpAdapter.get_image(file)
		else:
			if os.path.exists(image_path):
				return Image.open(image_path)
			else:
				return Image.new('RGB', (512, 512)) if displayed else None

	@staticmethod
	def get_weight_type_list():
		return [wt[0] for wt in weight_type]

	@staticmethod
	def get_weight_type(weight_type_name, weight):
		for wt in weight_type:
			if wt[0] == weight_type_name:
				return [x * weight for x in wt[1]]
		return [x * weight for x in weight_type[0][1]]

	@staticmethod
	def ipadapter_selected(*args):
		return IpAdapter.get_image(args[0], displayed=True)

