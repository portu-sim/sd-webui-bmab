from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from modules import shared
from modules import devices
from modules.processing import process_images, StableDiffusionProcessingImg2Img

from sd_bmab import constants, util
from sd_bmab.base import apply_extensions, build_img2img, Context, ProcessorBase, VAEMethodOverride, dino

from sd_bmab.util import debug_print
from sd_bmab.detectors.detector import get_detector


class InpaintResize(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.resize_by_person_opt = None
		self.value = 0,
		self.denoising_strength = 0.4
		self.dilation = 4
		self.mode = 'Inpaint'

	def preprocess(self, context: Context, image: Image):
		enabled = context.args.get('resize_by_person_enabled', False)
		self.resize_by_person_opt = context.args.get('module_config', {}).get('resize_by_person_opt', {})
		self.value = self.resize_by_person_opt.get('scale', 0)
		self.denoising_strength = self.resize_by_person_opt.get('denoising_strength', 0.4)
		self.dilation = self.resize_by_person_opt.get('dilation', 0.4)
		self.mode = self.resize_by_person_opt.get('mode', self.mode)

		return enabled and self.mode == 'Inpaint'

	def process(self, context: Context, image: Image):

		debug_print('prepare dino')
		dino.dino_init()
		boxes, logits, phrases = dino.dino_predict(image, 'person')
		if shared.opts.bmab_optimize_vram != 'None':
			dino.release()

		org_size = image.size
		debug_print('size', org_size)

		largest = (0, None)
		for box in boxes:
			x1, y1, x2, y2 = box
			size = (x2 - x1) * (y2 - y1)
			if size > largest[0]:
				largest = (size, box)

		if largest[0] == 0:
			return image

		x1, y1, x2, y2 = largest[1]
		ratio = (y2 - y1) / image.height
		debug_print('ratio', ratio)
		debug_print('org_size', org_size)
		if ratio <= self.value:
			return image
		image_ratio = ratio / self.value
		if image_ratio < 1.0:
			return image
		debug_print('scale', image_ratio)
		ratio = image_ratio

		org_size = image.size
		dw, dh = org_size

		context.add_generation_param('BMAB controlnet mode', 'inpaint')
		context.add_generation_param('BMAB resize by person ratio', '%.3s' % ratio)

		resized_width = int(dw / ratio)
		resized_height = int(dh / ratio)
		resized = image.resize((resized_width, resized_height), resample=util.LANCZOS)
		context.sdprocessing.resize_mode = 2
		input_image = util.resize_image(2, resized, dw, dh)

		offset_x = int((dw - resized_width) / 2)
		offset_y = dh - resized_height

		mask = Image.new('L', (dw, dh), 255)
		dr = ImageDraw.Draw(mask, 'L')
		dr.rectangle((offset_x, offset_y, offset_x + resized_width, offset_y + resized_height), fill=0)
		mask = mask.resize(org_size, resample=util.LANCZOS)
		mask = util.dilate_mask(mask, self.dilation)

		opt = dict(mask=mask, denoising_strength=self.denoising_strength)
		i2i_param = build_img2img(context.sdprocessing, input_image, opt)

		img2img = StableDiffusionProcessingImg2Img(**i2i_param)
		img2img.cached_c = [None, None]
		img2img.cached_uc = [None, None]
		img2img.scripts, img2img.script_args = apply_extensions(context.sdprocessing, cn_enabled=False)

		processed = process_images(img2img)
		img = processed.images[0]

		img2img.close()

		return img

	def postprocess(self, context: Context, image: Image):
		devices.torch_gc()
