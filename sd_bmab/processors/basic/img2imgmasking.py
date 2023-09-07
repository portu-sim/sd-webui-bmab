from PIL import Image

from modules import devices
from modules.processing import StableDiffusionProcessingImg2Img

from sd_bmab.base import sam, dino
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


class Img2imgMasking(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		self.enabled = context.args['dino_detect_enabled']
		self.prompt = context.args['dino_prompt']
		self.input_image = context.args['input_image']
		return isinstance(context.sdprocessing, StableDiffusionProcessingImg2Img) and self.enabled

	def sam(self, prompt, input_image):
		boxes, logits, phrases = dino.dino_predict(input_image, prompt, 0.35, 0.25)
		mask = sam.sam_predict(input_image, boxes)
		return mask

	def process(self, context: Context, image: Image):
		if context.sdprocessing.image_mask is not None:
			context.sdprocessing.image_mask = self.sam(self.prompt, context.sdprocessing.init_images[0])
			context.script.extra_image.append(context.sdprocessing.image_mask)
		if context.sdprocessing.image_mask is None and self.input_image is not None:
			mask = self.sam(self.prompt, context.sdprocessing.init_images[0])
			newpil = Image.new('RGB', context.sdprocessing.init_images[0].size)
			newdata = [bdata if mdata == 0 else ndata for mdata, ndata, bdata in
			           zip(mask.getdata(), context.sdprocessing.init_images[0].getdata(), self.input_image.getdata())]
			newpil.putdata(newdata)
			context.script.extra_image.append(newpil)
			return newpil
		return image

	def postprocess(self, context: Context, image: Image):
		devices.torch_gc()
