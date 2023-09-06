from PIL import Image

from modules import images

from sd_bmab.base import dino
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase
from sd_bmab import constants, util
from sd_bmab.util import debug_print


class BeforeProcessUpscaler(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.ratio = 1.5
		self.upscaler = 'None'

	def preprocess(self, context: Context, image: Image):
		self.ratio = context.args['upscale_ratio']
		self.upscaler = context.args['upscaler_name']
		return context.args['upscale_enabled'] and not context.args['detailing_after_upscale']

	def process(self, context: Context, image: Image):
		debug_print(f'Upscale ratio {self.ratio} Upscaler {self.upscaler}')
		context.add_generation_param('BMAB_upscale_option', f'Upscale ratio {self.ratio} Upscaler {self.upscaler}')

		if self.ratio < 1.0 or self.ratio > 4.0:
			debug_print('upscale out of range')
			return image
		image = image.convert('RGB')
		context.add_generation_param('BMAB process upscale', self.ratio)
		context.args['upscale_limit'] = True

		w = image.width
		h = image.height
		img = images.resize_image(0, image, int(w * self.ratio), int(h * self.ratio), self.upscaler)
		return img.convert('RGB')

	def postprocess(self, context: Context, image: Image):
		pass
