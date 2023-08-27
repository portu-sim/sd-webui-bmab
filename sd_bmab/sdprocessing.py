from modules.processing import StableDiffusionProcessingImg2Img


class StableDiffusionProcessingImg2ImgOv(StableDiffusionProcessingImg2Img):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.block_tqdm = False

