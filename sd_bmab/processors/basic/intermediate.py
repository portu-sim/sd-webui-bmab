from PIL import Image
from sd_bmab import util
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


class NoiseAlpha(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.noise_alpha = 0

	def preprocess(self, context: Context, image: Image):
		self.noise_alpha = context.args['noise_alpha']
		return self.noise_alpha != 0

	def process(self, context: Context, image: Image):
		context.add_generation_param('BMAB noise alpha final', self.noise_alpha)
		img_noise = util.generate_noise(image.size[0], image.size[1])
		return Image.blend(image, img_noise, alpha=self.noise_alpha)

	def postprocess(self, context: Context, image: Image):
		pass
