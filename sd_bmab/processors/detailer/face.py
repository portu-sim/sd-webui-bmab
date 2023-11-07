from math import sqrt

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from modules import shared
from modules import devices
from modules.processing import StableDiffusionProcessingImg2Img

from sd_bmab import constants, util
from sd_bmab.base import process_img2img, Context, ProcessorBase, VAEMethodOverride

from sd_bmab.util import debug_print
from sd_bmab.detectors.detector import get_detector


class FaceDetailer(ProcessorBase):
	def __init__(self, step=2) -> None:
		super().__init__()

		self.enabled = False
		self.hiresfix_enabled = False
		self.detailing_opt = None
		self.parameters = None
		self.override_parameter = False
		self.dilation = 4
		self.box_threshold = 0.35
		self.order = 'Score'
		self.limit = 1
		self.sampler = constants.sampler_default
		self.best_quality = False
		self.detection_model = 'GroundingDINO(face)'
		self.max_element = shared.opts.bmab_max_detailing_element
		self.step = step

	def preprocess(self, context: Context, image: Image):
		self.enabled = context.args['face_detailing_enabled']
		self.hiresfix_enabled = context.args['face_detailing_before_hiresfix_enabled']
		self.detailing_opt = context.args.get('module_config', {}).get('face_detailing_opt', {})
		self.parameters = dict(context.args.get('module_config', {}).get('face_detailing', {}))
		self.override_parameter = self.detailing_opt.get('override_parameter', self.override_parameter)
		self.dilation = self.detailing_opt.get('dilation', self.dilation)
		self.box_threshold = self.detailing_opt.get('box_threshold', self.box_threshold)
		self.order = self.detailing_opt.get('sort_by', self.order)
		self.limit = self.detailing_opt.get('limit', self.limit)
		self.sampler = self.detailing_opt.get('sampler', self.sampler)
		self.best_quality = self.detailing_opt.get('best_quality', self.best_quality)
		self.detection_model = self.detailing_opt.get('detection_model', self.detection_model)

		if self.step == 1 and context.is_hires_fix():
			return self.enabled and self.hiresfix_enabled
		return context.args['face_detailing_enabled']

	def process(self, context: Context, image: Image):

		detector = get_detector(context, self.detection_model, box_threshold=self.box_threshold)
		boxes, logits = detector.predict(context, image)

		org_size = image.size
		debug_print('size', org_size, len(boxes), len(logits))

		face_config = {
			'denoising_strength': self.parameters['denoising_strength'],
			'inpaint_full_res': self.parameters['inpaint_full_res'],
			'inpaint_full_res_padding': self.parameters['inpaint_full_res_padding'],
		}

		if self.override_parameter:
			face_config = dict(self.parameters)
		else:
			if self.best_quality or shared.opts.bmab_keep_original_setting:
				face_config['width'] = image.width
				face_config['height'] = image.height
			else:
				if isinstance(context.sdprocessing, StableDiffusionProcessingImg2Img):
					face_config['width'] = context.sdprocessing.init_images[0].width
					face_config['height'] = context.sdprocessing.init_images[0].height
				else:
					face_config['width'] = context.sdprocessing.width
					face_config['height'] = context.sdprocessing.height

		if self.sampler != constants.sampler_default:
			face_config['sampler_name'] = self.sampler

		context.add_generation_param('BMAB_face_option', util.dict_to_str(self.detailing_opt))
		context.add_generation_param('BMAB_face_parameter', util.dict_to_str(face_config))

		candidate = []
		if self.order == 'Left':
			for box, logit in zip(boxes, logits):
				x1, y1, x2, y2 = box
				value = x1 + (x2 - x1) // 2
				debug_print('detected(from left)', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0])
		elif self.order == 'Right':
			for box, logit in zip(boxes, logits):
				x1, y1, x2, y2 = box
				value = x1 + (x2 - x1) // 2
				debug_print('detected(from right)', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		elif self.order == 'Center':
			for box, logit in zip(boxes, logits):
				x1, y1, x2, y2 = box
				cx = image.width / 2
				cy = image.height / 2
				ix = x1 + (x2 - x1) // 2
				iy = y1 + (y2 - y1) // 2
				value = sqrt(abs(cx - ix) ** 2 + abs(cy - iy) ** 2)
				debug_print('detected(from center)', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0])
		elif self.order == 'Size':
			for box, logit in zip(boxes, logits):
				x1, y1, x2, y2 = box
				value = (x2 - x1) * (y2 - y1)
				debug_print('detected(size)', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		else:
			for box, logit in zip(boxes, logits):
				value = float(logit)
				debug_print(f'detected({self.order})', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)

		context.add_job(min(self.limit, len(candidate)))

		for idx, (size, box, logit) in enumerate(candidate):
			if self.limit != 0 and idx >= self.limit:
				debug_print(f'Over limit {self.limit}')
				break

			if self.max_element != 0 and idx >= self.max_element:
				debug_print(f'Over limit MAX Element {self.max_element}')
				break

			prompt = self.detailing_opt.get(f'prompt{idx}')
			if prompt is not None:
				if prompt.find('#!org!#') >= 0:
					current_prompt = context.get_prompt_by_index()
					face_config['prompt'] = prompt.replace('#!org!#', current_prompt)
					debug_print('prompt for face', face_config['prompt'])
				elif prompt != '':
					face_config['prompt'] = prompt
				else:
					face_config['prompt'] = context.get_prompt_by_index()

			ne_prompt = self.detailing_opt.get(f'negative_prompt{idx}')
			if ne_prompt is not None and ne_prompt != '':
				face_config['negative_prompt'] = ne_prompt
			else:
				face_config['negative_prompt'] = context.get_negative_prompt_by_index()

			debug_print('render', float(logit))
			debug_print('delation', self.dilation)

			face_mask = Image.new('L', image.size, color=0)
			dr = ImageDraw.Draw(face_mask, 'L')
			dr.rectangle(box, fill=255)
			face_mask = util.dilate_mask(face_mask, self.dilation)

			seed, subseed = context.get_seeds()
			options = dict(mask=face_mask, seed=seed, subseed=subseed, **face_config)
			with VAEMethodOverride(hiresfix=self.best_quality):
				img2img_imgage = process_img2img(context.sdprocessing, image, options=options)

			x1, y1, x2, y2 = util.fix_box_size(box)
			face_mask = Image.new('L', image.size, color=0)
			dr = ImageDraw.Draw(face_mask, 'L')
			dr.rectangle((x1, y1, x2, y2), fill=255)
			blur = ImageFilter.GaussianBlur(3)
			mask = face_mask.filter(blur)
			image.paste(img2img_imgage, mask=mask)
		return image

	def postprocess(self, context: Context, image: Image):
		devices.torch_gc()
