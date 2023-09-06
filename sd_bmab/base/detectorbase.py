from PIL import Image
from sd_bmab.base.context import Context


class DetectorBase(object):
	def __init__(self, **kwargs) -> None:
		super().__init__()

	def target(self):
		pass

	def description(self):
		pass

	def predict(self, context: Context, image: Image):
		pass
