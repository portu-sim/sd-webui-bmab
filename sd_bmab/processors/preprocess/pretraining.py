import torch
import modules
from ultralytics import YOLO

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from modules import devices

from sd_bmab import constants, util
from sd_bmab.base import process_img2img, Context, ProcessorBase, VAEMethodOverride
from sd_bmab.util import debug_print


class PretrainingDetailer(ProcessorBase):
	def __init__(self, step=2) -> None:
		super().__init__()
		self.pretraining_opt = {}

		self.enabled = False
		self.hiresfix_enabled = False
		self.pretraining_model = None
		self.prompt = ''
		self.negative_prompt = ''
		self.sampler = constants.sampler_default
		self.scheduler = constants.scheduler_default
		self.steps = 20
		self.cfg_scale = 7
		self.denoising_strength = 0.75
		self.confidence = 0.35
		self.dilation = 4

		self.preprocess_step = step

	def predict(self, context: Context, image: Image, ptmodel, confidence):
		yolo = util.load_pretraining_model(ptmodel)
		boxes = []
		confs = []
		load = torch.load
		torch.load = modules.safe.unsafe_torch_load
		try:
			model = YOLO(yolo)
			pred = model(image, conf=confidence, device='')
			boxes = pred[0].boxes.xyxy.cpu().numpy()
			boxes = boxes.tolist()
			confs = pred[0].boxes.conf.tolist()
		except:
			pass
		torch.load = load
		devices.torch_gc()

		return boxes, confs

	def preprocess(self, context: Context, image: Image):
		self.enabled = context.args['pretraining_enabled']
		self.pretraining_opt = context.args.get('module_config', {}).get('pretraining_opt', {})
		self.hiresfix_enabled = self.pretraining_opt.get('hiresfix_enabled', self.hiresfix_enabled)
		self.pretraining_model = self.pretraining_opt.get('pretraining_model', self.pretraining_model)
		self.prompt = self.pretraining_opt.get('prompt', self.prompt)
		self.negative_prompt = self.pretraining_opt.get('negative_prompt', self.negative_prompt)
		self.sampler = self.pretraining_opt.get('sampler', self.sampler)
		self.scheduler = self.pretraining_opt.get('scheduler', self.scheduler)
		self.steps = self.pretraining_opt.get('steps', self.steps)
		self.cfg_scale = self.pretraining_opt.get('cfg_scale', self.cfg_scale)
		self.denoising_strength = self.pretraining_opt.get('denoising_strength', self.denoising_strength)
		self.confidence = self.pretraining_opt.get('box_threshold', 0.35)
		self.dilation = self.pretraining_opt.get('dilation', self.dilation)

		if self.enabled and self.preprocess_step == 1:
			return context.is_hires_fix() and self.hiresfix_enabled
		if self.enabled and self.preprocess_step == 2 and self.hiresfix_enabled:
			return False
		return self.enabled

	def process(self, context: Context, image: Image):
		boxes, logits = self.predict(context, image, self.pretraining_model, self.confidence)

		org_size = image.size
		debug_print('size', org_size, len(boxes), len(logits))
		debug_print('sampler', context.sdprocessing.sampler_name if self.sampler == constants.sampler_default else self.sampler)

		pretraining_config = {
			'steps': self.steps,
			'cfg_scale': self.cfg_scale,
			'sampler_name': context.sdprocessing.sampler_name if self.sampler == constants.sampler_default else self.sampler,
			'scheduler': util.get_scheduler(context.sdprocessing) if self.scheduler == constants.scheduler_default else self.scheduler,
			'denoising_strength': self.denoising_strength,
			'width': context.sdprocessing.width,
			'height': context.sdprocessing.height,
		}

		candidate = []
		for box, logit in zip(boxes, logits):
			value = float(logit)
			candidate.append((value, box, logit))
		candidate = sorted(candidate, key=lambda c: c[0], reverse=True)

		for idx, (size, box, logit) in enumerate(candidate):
			context.add_job()
			'''
			prompt = self.detailing_opt.get(f'prompt{idx}')
			if prompt is not None:

			'''
			prompt = self.prompt
			if prompt.find('#!org!#') >= 0:
				current_prompt = context.get_prompt_by_index()
				pretraining_config['prompt'] = prompt.replace('#!org!#', current_prompt)
				debug_print('prompt for detection', pretraining_config['prompt'])
			elif prompt != '':
				pretraining_config['prompt'] = prompt
			else:
				pretraining_config['prompt'] = context.get_prompt_by_index()

			pretraining_config['negative_prompt'] = context.get_negative_prompt_by_index()

			debug_print('prompt', pretraining_config['prompt'])
			debug_print('negative_prompt', pretraining_config['negative_prompt'])

			debug_print('render', float(logit))
			debug_print('delation', self.dilation)

			debug_print('box', box)

			detected_mask = Image.new('L', image.size, color=0)
			dr = ImageDraw.Draw(detected_mask, 'L')
			dr.rectangle(box, fill=255)
			detected_mask = util.dilate_mask(detected_mask, self.dilation)

			seed, subseed = context.get_seeds()
			options = dict(mask=detected_mask, seed=seed, subseed=subseed, **pretraining_config)
			with VAEMethodOverride():
				img2img_imgage = process_img2img(context.sdprocessing, image, options=options)

			x1, y1, x2, y2 = util.fix_box_size(box)
			x1 -= int(detected_mask.width / 2)
			x2 += int(detected_mask.width / 2)
			y1 -= int(detected_mask.height / 2)
			y2 += int(detected_mask.height / 2)
			
			detected_mask = Image.new('L', image.size, color=0)
			dr = ImageDraw.Draw(detected_mask, 'L')
			dr.rectangle((x1, y1, x2, y2), fill=255)
			blur = ImageFilter.GaussianBlur(3)
			mask = detected_mask.filter(blur)
			image.paste(img2img_imgage, mask=mask)
		return image

	def postprocess(self, context: Context, image: Image):
		devices.torch_gc()
