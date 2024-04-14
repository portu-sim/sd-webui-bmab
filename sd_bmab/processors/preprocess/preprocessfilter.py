from PIL import Image
from sd_bmab.base import Context, ProcessorBase
from sd_bmab.base import filter


class PreprocessFilter(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.filter_name = 'None'
		self.filter = None

	def preprocess(self, context: Context, image: Image):
		self.filter_name = context.args.get('txt2img_filter_hresfix_before_upscale', self.filter_name)
		return not context.is_hires_fix() and self.filter_name != 'None'

	def process(self, context: Context, image: Image):
		self.filter = filter.get_filter(self.filter_name)
		if self.filter is not None:
			filter.preprocess_filter(self.filter, context, image)
			image = filter.process_filter(self.filter, context, None, image)
			filter.postprocess_filter(self.filter, context)
		return image

	def finalprocess(self, context: Context, image: Image):
		if self.filter is not None:
			filter.finalprocess_filter(self.filter, context)
		self.filter = None
