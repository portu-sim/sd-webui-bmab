from PIL import Image

from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase
from sd_bmab.pipeline.internal import process_intermediate_before_upscale, process_intermediate_after_upscale


class Preprocess(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		return context.is_txtimg() and not context.is_hires_fix()

	def process(self, context: Context, image: Image):
		image = process_intermediate_before_upscale(context, image)
		return process_intermediate_after_upscale(context, image)

	def postprocess(self, context: Context, image: Image):
		pass
