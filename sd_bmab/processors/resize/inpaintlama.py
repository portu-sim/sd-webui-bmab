from PIL import Image
from PIL import ImageDraw

from modules import shared
from modules import devices
from modules.processing import process_images, StableDiffusionProcessingImg2Img

from sd_bmab import util
from sd_bmab.base import apply_extensions, build_img2img, Context, ProcessorBase, dino


class InpaintLamaResize(ProcessorBase):
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

		return enabled and self.mode == 'ControlNet inpaint+lama'

	@staticmethod
	def get_inpaint_lama_args(image, mask):
		cn_args = {
			'input_image': util.b64_encoding(image),
			'mask': util.b64_encoding(mask),
			'module': 'inpaint_only+lama',
			'model': shared.opts.bmab_cn_inpaint,
			'weight': 1,
			"guidance_start": 0,
			"guidance_end": 1,
			'resize mode': 'Resize and Fill',
			'allow preview': False,
			'pixel perfect': False,
			'control mode': 'ControlNet is more important',
			'processor_res': 512,
			'threshold_a': 64,
			'threshold_b': 64,
		}
		return cn_args

	def get_ratio(self, context, img, p):
		p.extra_generation_params['BMAB process_resize_by_person'] = self.value

		final_ratio = 1
		dino.dino_init()
		boxes, logits, phrases = dino.dino_predict(img, 'person')

		largest = (0, None)
		for box in boxes:
			x1, y1, x2, y2 = box
			size = (x2 - x1) * (y2 - y1)
			if size > largest[0]:
				largest = (size, box)

		if largest[0] == 0:
			return final_ratio

		x1, y1, x2, y2 = largest[1]
		ratio = (y2 - y1) / img.height
		print('ratio', ratio)
		dino.release()

		if ratio > self.value:
			image_ratio = ratio / self.value
			if image_ratio < 1.0:
				return final_ratio
			final_ratio = image_ratio
		return final_ratio

	def resize_by_person_using_controlnet(self, context, p):
		if not isinstance(p, StableDiffusionProcessingImg2Img):
			return False

		cn_args = util.get_cn_args(p)

		print('resize_by_person_enabled_inpaint', self.value)
		img = p.init_images[0]
		context.script.extra_image.append(img)

		ratio = self.get_ratio(context, img, p)
		print('image resize ratio', ratio)
		org_size = img.size
		dw, dh = org_size

		if ratio == 1:
			return False

		p.extra_generation_params['BMAB controlnet mode'] = 'inpaint'
		p.extra_generation_params['BMAB resize by person ratio'] = '%.3s' % ratio

		resized_width = int(dw / ratio)
		resized_height = int(dh / ratio)
		resized = img.resize((resized_width, resized_height), resample=util.LANCZOS)
		p.resize_mode = 2
		input_image = util.resize_image(2, resized, dw, dh)
		p.init_images[0] = input_image

		offset_x = int((dw - resized_width) / 2)
		offset_y = dh - resized_height

		mask = Image.new('L', (dw, dh), 255)
		dr = ImageDraw.Draw(mask, 'L')
		dr.rectangle((offset_x, offset_y, offset_x + resized_width, offset_y + resized_height), fill=0)
		mask = mask.resize(org_size, resample=util.LANCZOS)
		mask = util.dilate_mask(mask, self.dilation)

		cn_op_arg = self.get_inpaint_lama_args(input_image, mask)
		idx = cn_args[0]
		sc_args = list(p.script_args)
		sc_args[idx] = cn_op_arg
		p.script_args = tuple(sc_args)
		return True

	def process(self, context: Context, image: Image):
		opt = dict(denoising_strength=self.denoising_strength)
		i2i_param = build_img2img(context.sdprocessing, image, opt)

		img2img = StableDiffusionProcessingImg2Img(**i2i_param)
		img2img.cached_c = [None, None]
		img2img.cached_uc = [None, None]
		img2img.scripts, img2img.script_args = apply_extensions(context.sdprocessing, cn_enabled=True)

		if self.resize_by_person_using_controlnet(context, img2img):
			processed = process_images(img2img)
			image = processed.images[0]
			img2img.close()
			devices.torch_gc()
		return image

	def postprocess(self, context: Context, image: Image):
		pass
