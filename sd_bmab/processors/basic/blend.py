from PIL import Image

from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase
from sd_bmab.util import debug_print


class BlendImage(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.enabled = False
		self.input_image = None
		self.alpha = 0

	def preprocess(self, context: Context, image: Image):
		self.enabled = context.args['blend_enabled']
		self.input_image = context.args['input_image']
		self.alpha = context.args['blend_alpha']
		return self.enabled and self.input_image is not None and 0 <= self.alpha <= 1

	def process(self, context: Context, image: Image):
		context.add_generation_param('BMAB blend alpha', self.alpha)
		#blend = Image.fromarray(self.input_image, mode='RGB')
		debug_print(self.input_image)
		blend = self.input_image
		img = Image.new(mode='RGB', size=image.size)
		img.paste(image, (0, 0))
		img.paste(blend)
		image = Image.blend(image, img, alpha=self.alpha)
		return image

	def postprocess(self, context: Context, image: Image):
		pass
