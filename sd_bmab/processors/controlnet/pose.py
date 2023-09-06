
from PIL import Image

from modules import shared
from modules import images

from sd_bmab import util
from sd_bmab.util import debug_print
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


class Openpose(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		self.controlnet_opt = context.args.get('module_config', {}).get('controlnet', {})
		self.enabled = self.controlnet_opt.get('enabled', False)
		self.pose = self.controlnet_opt.get('pose', False)
		return self.enabled

	@staticmethod
	def get_openpose_args(image):
		cn_args = {
			'input_image': util.b64_encoding(image),
			'module': 'openpose',
			'model': shared.opts.bmab_cn_openpose,
			'weight': 1,
			"guidance_start": 0,
			"guidance_end": 1,
			'resize mode': 'Just Resize',
			'allow preview': False,
			'pixel perfect': False,
			'control mode': 'My prompt is more important',
			'processor_res': 512,
			'threshold_a': 64,
			'threshold_b': 64,
		}
		return cn_args

	def process(self, context: Context, image: Image):
		context.add_generation_param('BMAB_controlnet_option', util.dict_to_str(self.controlnet_opt))

		debug_print('Seed', context.sdprocessing.seed)
		debug_print('AllSeeds', context.sdprocessing.all_seeds)

		cn_args = util.get_cn_args(context.sdprocessing)
		debug_print('ControlNet', cn_args)

		noise_strength = self.controlnet_opt.get('noise_strength', 0.4)
		debug_print('noise enabled.', noise_strength)
		context.add_generation_param('BMAB controlnet mode', 'lineart')
		context.add_generation_param('BMAB noise strength', noise_strength)

		img = util.generate_noise(context.sdprocessing.width, context.sdprocessing.height)
		cn_op_arg = self.get_openpose_args(img)
		idx = cn_args[0] + context.controlnet_count
		context.controlnet_count += 1
		sc_args = list(context.sdprocessing.script_args)
		sc_args[idx] = cn_op_arg
		context.sdprocessing.script_args = tuple(sc_args)

	def postprocess(self, context: Context, image: Image):
		pass


