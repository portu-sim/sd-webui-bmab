from PIL import Image

from modules import shared

from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase
from sd_bmab import util
from sd_bmab.base import filter
from sd_bmab.util import debug_print
from sd_bmab.detectors import UltralyticsPersonDetector8n
from sd_bmab.base import process_img2img, process_img2img_with_controlnet
from sd_bmab.external.lama import LamaInpainting


class ResizeIntermidiate(ProcessorBase):
	def __init__(self, step=2) -> None:
		super().__init__()
		self.enabled = False
		self.filter = 'None'
		self.resize_by_person_opt = None
		self.resize_by_person = True
		self.method = 'stretching'
		self.alignment = 'bottom'
		self.value = 0
		self.denoising_strength = 0.75
		self.step = step

	def use_controlnet(self, context: Context):
		self.preprocess(context, None)
		if self.enabled and self.method in ['inpaint_only+lama', 'inpaint_only']:
			return True
		return False

	def preprocess(self, context: Context, image: Image):
		self.enabled = context.args.get('resize_intermediate_enabled', False)
		self.resize_by_person_opt = context.args.get('module_config', {}).get('resize_intermediate_opt', {})

		self.filter = self.resize_by_person_opt.get('filter', self.filter)
		self.resize_by_person = self.resize_by_person_opt.get('resize_by_person', self.resize_by_person)
		self.method = self.resize_by_person_opt.get('method', self.method)
		self.alignment = self.resize_by_person_opt.get('alignment', self.alignment)
		self.value = self.resize_by_person_opt.get('scale', self.value)
		self.denoising_strength = self.resize_by_person_opt.get('denoising_strength', self.denoising_strength)

		if context.is_txtimg():
			if not self.enabled:
				return False
			if self.step == 1 and self.method == 'stretching':
				return False
			if self.step == 2 and self.method != 'stretching':
				return False
			if 0.5 > self.value >= 1.0:
				return False
			return self.enabled
		else:
			return self.enabled

	@staticmethod
	def get_inpaint_lama_args(image, mask, module):
		cn_args = {
			'input_image': util.b64_encoding(image),
			'mask': util.b64_encoding(mask),
			'module': module,
			'model': shared.opts.bmab_cn_inpaint,
			'weight': 1,
			"guidance_start": 0,
			"guidance_end": 1,
			'resize_mode': 'Resize and Fill',
			'pixel_perfect': False,
			'control_mode': 'ControlNet is more important',
			'processor_res': 512,
			'threshold_a': 64,
			'threshold_b': 64,
		}
		return cn_args

	def process(self, context: Context, image: Image):
		bmab_filter = filter.get_filter(self.filter)
		filter.preprocess_filter(bmab_filter, context, image)
		image = self.process_resize(context, image)
		image = filter.process_filter(bmab_filter, context, image, image)
		filter.postprocess_filter(bmab_filter, context)
		return image

	def process_resize(self, context: Context, image: Image):
		context.add_generation_param('BMAB process_resize_by_person', self.value)
		org_size = image.size

		if self.resize_by_person:
			debug_print('prepare detector')
			detector = UltralyticsPersonDetector8n()
			boxes, logits = detector.predict(context, image)

			debug_print('boxes', len(boxes))
			debug_print('logits', len(logits))
			debug_print('alignment', self.alignment)

			debug_print('size', org_size)

			if len(boxes) == 0:
				return image

			largest = []
			for idx, box in enumerate(boxes):
				x1, y1, x2, y2 = box
				largest.append(((y2 - y1), box))
				debug_print(f'ratio {idx}', (y2 - y1) / image.height)
			largest = sorted(largest, key=lambda c: c[0], reverse=True)

			x1, y1, x2, y2 = largest[0][1]
			ratio = (y2 - y1) / image.height
			debug_print('ratio', ratio)
			debug_print('org_size', org_size)
			if ratio > self.value:
				image_ratio = ratio / self.value
				if image_ratio < 1.0:
					return image
			else:
				return image
		else:
			image_ratio = 1 / self.value

		context.add_generation_param('BMAB process_resize_by_person_ratio', '%.3s' % image_ratio)

		debug_print('image resize ratio', image_ratio)
		stretching_image = util.resize_image_with_alignment(image, self.alignment, int(image.width * image_ratio), int(image.height * image_ratio))

		if self.method == 'stretching':
			# image = util.resize_image(2, image, int(image.width * image_ratio), int(image.height * image_ratio))
			debug_print('Stretching')
			return image
		elif self.method == 'inpaint':
			mask = util.get_mask_with_alignment(image, self.alignment, int(image.width * image_ratio), int(image.height * image_ratio))
			debug_print('mask size', mask.size)
			seed, subseed = context.get_seeds()
			options = dict(
				seed=seed, subseed=subseed,
				denoising_strength=self.denoising_strength,
				resize_mode=0,
				mask=mask,
				mask_blur=4,
				inpainting_fill=1,
				inpaint_full_res=True,
				inpaint_full_res_padding=32,
				inpainting_mask_invert=0,
				initial_noise_multiplier=1.0,
				prompt=context.get_prompt_by_index(),
				negative_prompt=context.get_negative_prompt_by_index(),
				batch_size=1,
				n_iter=1,
				restore_faces=False,
				do_not_save_samples=True,
				do_not_save_grid=True,
			)
			context.add_job()
			image = process_img2img(context.sdprocessing, stretching_image, options=options)
			return image
		elif self.method == 'inpaint+lama':
			mask = util.get_mask_with_alignment(image, self.alignment, int(image.width * image_ratio), int(image.height * image_ratio))
			lama = LamaInpainting()
			stretching_image = lama(stretching_image, mask)
			debug_print('mask size', mask.size)
			seed, subseed = context.get_seeds()
			options = dict(
				seed=seed, subseed=subseed,
				denoising_strength=self.denoising_strength,
				resize_mode=0,
				mask=mask,
				mask_blur=4,
				inpainting_fill=1,
				inpaint_full_res=True,
				inpaint_full_res_padding=32,
				inpainting_mask_invert=0,
				initial_noise_multiplier=1.0,
				prompt=context.get_prompt_by_index(),
				negative_prompt=context.get_negative_prompt_by_index(),
				batch_size=1,
				n_iter=1,
				restore_faces=False,
				do_not_save_samples=True,
				do_not_save_grid=True,
			)
			context.add_job()
			image = process_img2img(context.sdprocessing, stretching_image, options=options)
			return image
		elif self.method == 'inpaint_only+lama':
			mask = util.get_mask_with_alignment(image, self.alignment, int(image.width * image_ratio), int(image.height * image_ratio))
			opt = dict(denoising_strength=self.denoising_strength)
			debug_print('Stretching image size', stretching_image.size)
			debug_print('Mask image size', mask.size)
			cnarg = self.get_inpaint_lama_args(stretching_image, mask, 'inpaint_only+lama')
			context.add_job()
			image = process_img2img_with_controlnet(context, image, opt, cnarg)
		elif self.method == 'inpaint_only':
			mask = util.get_mask_with_alignment(image, self.alignment, int(image.width * image_ratio), int(image.height * image_ratio))
			opt = dict(denoising_strength=self.denoising_strength)
			debug_print('Stretching image size', stretching_image.size)
			debug_print('Mask image size', mask.size)
			cnarg = self.get_inpaint_lama_args(stretching_image, mask, 'inpaint_only')
			context.add_job()
			image = process_img2img_with_controlnet(context, image, opt, cnarg)
		return image

	def postprocess(self, context: Context, image: Image):
		pass
