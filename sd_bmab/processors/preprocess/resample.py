from PIL import Image

from modules import shared
from modules import devices
from modules import images

from sd_bmab import util
from sd_bmab import constants
from sd_bmab.base import filter
from sd_bmab.util import debug_print
from sd_bmab.base import process_txt2img, process_img2img_with_controlnet, Context, ProcessorBase


class ResamplePreprocessor(ProcessorBase):
	def __init__(self, step=2) -> None:
		super().__init__()

		self.resample_opt = {}
		self.enabled = False
		self.hiresfix_enabled = False
		self.save_image = False
		self.method = 'txt2img-1pass'
		self.checkpoint = constants.checkpoint_default
		self.vae = constants.vae_default
		self.filter = 'None'
		self.prompt = None
		self.negative_prompt = None
		self.sampler = None
		self.scheduler = None
		self.upscaler = None
		self.steps = 20
		self.cfg_scale = 0.7
		self.denoising_strength = 0.75
		self.strength = 0.5
		self.begin = 0.0
		self.end = 1.0

		self.base_sd_model = None
		self.preprocess_step = step

	def use_controlnet(self, context: Context):
		return self.preprocess(context, None)

	def preprocess(self, context: Context, image: Image):
		self.enabled = context.args['resample_enabled']
		self.resample_opt = context.args.get('module_config', {}).get('resample_opt', {})

		self.hiresfix_enabled = self.resample_opt.get('hiresfix_enabled', self.hiresfix_enabled)
		self.save_image = self.resample_opt.get('save_image', self.save_image)
		self.method = self.resample_opt.get('method', self.method)
		self.checkpoint = self.resample_opt.get('checkpoint', self.checkpoint)
		self.vae = self.resample_opt.get('vae', self.vae)
		self.filter = self.resample_opt.get('filter', self.filter)
		self.prompt = self.resample_opt.get('prompt', self.prompt)
		self.negative_prompt = self.resample_opt.get('negative_prompt', self.negative_prompt)
		self.sampler = self.resample_opt.get('sampler', self.sampler)
		self.scheduler = self.resample_opt.get('scheduler', self.scheduler)
		self.upscaler = self.resample_opt.get('upscaler', self.upscaler)
		self.steps = self.resample_opt.get('steps', self.steps)
		self.cfg_scale = self.resample_opt.get('cfg_scale', self.cfg_scale)
		self.denoising_strength = self.resample_opt.get('denoising_strength', self.denoising_strength)
		self.strength = self.resample_opt.get('scale', self.strength)
		self.begin = self.resample_opt.get('width', self.begin)
		self.end = self.resample_opt.get('height', self.end)

		if self.enabled and self.preprocess_step == 1:
			return context.is_hires_fix() and self.hiresfix_enabled
		if self.enabled and self.preprocess_step == 2 and self.hiresfix_enabled:
			return False
		return self.enabled

	@staticmethod
	def get_resample_args(image, weight, begin, end):
		cn_args = {
			'input_image': util.b64_encoding(image),
			'module': 'tile_resample',
			'model': shared.opts.bmab_cn_tile_resample,
			'weight': weight,
			"guidance_start": begin,
			"guidance_end": end,
			'resize_mode': 'Just Resize',
			'pixel_perfect': False,
			'control_mode': 'ControlNet is more important',
			'processor_res': -1,
			'threshold_a': 1,
			'threshold_b': 1,
		}
		return cn_args

	def process(self, context: Context, image: Image):
		if self.prompt == '':
			self.prompt = context.get_prompt_by_index()
			debug_print('prompt', self.prompt)
		elif self.prompt.find('#!org!#') >= 0:
			current_prompt = context.get_prompt_by_index()
			self.prompt = self.prompt.replace('#!org!#', current_prompt)
			debug_print('Prompt', self.prompt)
		if self.negative_prompt == '':
			self.negative_prompt = context.sdprocessing.negative_prompt
		if self.checkpoint == constants.checkpoint_default:
			self.checkpoint = context.sdprocessing.sd_model
		if self.sampler == constants.sampler_default:
			self.sampler = context.sdprocessing.sampler_name
		if self.scheduler == constants.scheduler_default:
			self.scheduler = util.get_scheduler(context.sdprocessing)

		bmab_filter = filter.get_filter(self.filter)

		seed, subseed = context.get_seeds()
		options = dict(
			seed=seed, subseed=subseed,
			denoising_strength=self.denoising_strength,
			prompt=self.prompt,
			negative_prompt=self.negative_prompt,
			sampler_name=self.sampler,
			scheduler=self.scheduler,
			steps=self.steps,
			cfg_scale=self.cfg_scale,
		)

		if self.checkpoint != constants.checkpoint_default:
			override_settings = options.get('override_settings', {})
			override_settings['sd_model_checkpoint'] = self.checkpoint
			options['override_settings'] = override_settings
		if self.vae != constants.vae_default:
			override_settings = options.get('override_settings', {})
			override_settings['sd_vae'] = self.vae
			options['override_settings'] = override_settings

		filter.preprocess_filter(bmab_filter, context, image, options)

		context.add_job()
		if self.save_image:
			saved = image.copy()
			images.save_image(
				saved, context.sdprocessing.outpath_samples, '',
				context.sdprocessing.all_seeds[context.index], context.sdprocessing.all_prompts[context.index],
				shared.opts.samples_format, p=context.sdprocessing, suffix='-before-resample')
			context.add_extra_image(saved)
		cn_op_arg = self.get_resample_args(image, self.strength, self.begin, self.end)

		processed = image.copy()
		if self.hiresfix_enabled:
			if self.method == 'txt2img-1pass' or self.method == 'txt2img-2pass':
				options['width'] = context.sdprocessing.width
				options['height'] = context.sdprocessing.height
				processed = process_txt2img(context, options=options, controlnet=[cn_op_arg])
			elif self.method == 'img2img-1pass':
				del cn_op_arg['input_image']
				options['width'] = context.sdprocessing.width
				options['height'] = context.sdprocessing.height
				processed = process_img2img_with_controlnet(context, image, options=options, controlnet=[cn_op_arg])
		else:
			if self.method == 'txt2img-1pass':
				if context.is_hires_fix():
					if context.sdprocessing.hr_resize_x != 0 or context.sdprocessing.hr_resize_y != 0:
						options['width'] = context.sdprocessing.hr_resize_x
						options['height'] = context.sdprocessing.hr_resize_y
					else:
						options['width'] = int(context.sdprocessing.width * context.sdprocessing.hr_scale)
						options['height'] = int(context.sdprocessing.height * context.sdprocessing.hr_scale)
				processed = process_txt2img(context, options=options, controlnet=[cn_op_arg])
			elif self.method == 'txt2img-2pass':
				if context.is_txtimg() and context.is_hires_fix():
					options.update(dict(
						enable_hr=context.sdprocessing.enable_hr,
						hr_scale=context.sdprocessing.hr_scale,
						hr_resize_x=context.sdprocessing.hr_resize_x,
						hr_resize_y=context.sdprocessing.hr_resize_y,
					))
				processed = process_txt2img(context, options=options, controlnet=[cn_op_arg])
			elif self.method == 'img2img-1pass':
				del cn_op_arg['input_image']
				processed = process_img2img_with_controlnet(context, image, options=options, controlnet=[cn_op_arg])

		image = filter.process_filter(bmab_filter, context, image, processed)
		filter.postprocess_filter(bmab_filter, context)

		return image

	def postprocess(self, context: Context, image: Image):
		devices.torch_gc()
