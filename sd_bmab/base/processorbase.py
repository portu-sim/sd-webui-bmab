from PIL import Image
from sd_bmab.base.context import Context


class ProcessorBase(object):
	def __init__(self) -> None:
		super().__init__()

	def use_controlnet(self, context: Context):
		return False

	def preprocess(self, context: Context, image: Image):
		pass

	def process(self, context: Context, image: Image):
		pass

	def postprocess(self, context: Context, image: Image):
		pass
