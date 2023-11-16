from PIL import Image

from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase
from sd_bmab.pipeline.internal import process_intermediate_step1, process_intermediate_step2


class Preprocess(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		return not context.is_hires_fix() and not context.is_img2img()

	def process(self, context: Context, image: Image):
		image = process_intermediate_step1(context, image)
		return process_intermediate_step2(context, image)

	def postprocess(self, context: Context, image: Image):
		pass
