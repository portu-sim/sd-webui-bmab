from modules.processing import StableDiffusionProcessingImg2Img
from modules.processing import StableDiffusionProcessingTxt2Img


class StableDiffusionProcessingImg2ImgOv(StableDiffusionProcessingImg2Img):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.block_tqdm = False


class StableDiffusionProcessingTxt2ImgOv(StableDiffusionProcessingTxt2Img):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.block_tqdm = False

