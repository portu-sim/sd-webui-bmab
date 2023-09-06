import cv2
import numpy as np

from PIL import Image
from PIL import ImageOps
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


class EdgeEnhancement(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

		self.edge_low_threadhold = 50
		self.edge_high_threadhold = 200
		self.edge_strength = 0.5

	def preprocess(self, context: Context, image: Image):
		if context.args['edge_flavor_enabled']:
			self.edge_low_threadhold = context.args['edge_low_threadhold']
			self.edge_high_threadhold = context.args['edge_high_threadhold']
			self.edge_strength = context.args['edge_strength']
		return context.args['edge_flavor_enabled']

	def process(self, context: Context, image: Image):
		context.add_generation_param('BMAB edge flavor low threadhold', self.edge_low_threadhold)
		context.add_generation_param('BMAB edge flavor high threadhold', self.edge_high_threadhold)
		context.add_generation_param('BMAB edge flavor strength', self.edge_strength)

		numpy_image = np.array(image)
		base = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
		arcanny = cv2.Canny(base, self.edge_low_threadhold, self.edge_high_threadhold)
		canny = Image.fromarray(arcanny)
		canny = ImageOps.invert(canny)

		newdata = [(0, 0, 0) if mdata == 0 else ndata for mdata, ndata in zip(canny.getdata(), image.getdata())]
		newbase = Image.new('RGB', image.size)
		newbase.putdata(newdata)
		return Image.blend(image, newbase, alpha=self.edge_strength).convert("RGB")
