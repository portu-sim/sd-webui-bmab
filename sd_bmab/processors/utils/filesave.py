from PIL import Image

from modules import shared
from modules import images

from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


class BeforeProcessFileSaver(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		return shared.opts.bmab_save_image_before_process
	
	def process(self, context: Context, image: Image):
		images.save_image(
			image, context.sdprocessing.outpath_samples, '',
			context.sdprocessing.all_seeds[context.index], context.sdprocessing.all_prompts[context.index],
			shared.opts.samples_format, p=context.sdprocessing, suffix='-before-bmab')
		return image

	def postprocess(self, context: Context, image: Image):
		pass


class AfterProcessFileSaver(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		return shared.opts.bmab_save_image_after_process

	def process(self, context: Context, image: Image):
		images.save_image(
			image, context.sdprocessing.outpath_samples, '',
			context.sdprocessing.all_seeds[context.index], context.sdprocessing.all_prompts[context.index],
			shared.opts.samples_format, p=context.sdprocessing, suffix="-after-bmab")
		return image

	def postprocess(self, context: Context, image: Image):
		pass
