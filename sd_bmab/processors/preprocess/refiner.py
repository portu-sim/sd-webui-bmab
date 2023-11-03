from PIL import Image

from modules import shared
from modules import devices
from modules import sd_models
from modules import images

from sd_bmab import constants
from sd_bmab.util import debug_print
from sd_bmab.base import process_img2img, Context, ProcessorBase
from sd_bmab.processors.resize import IntermidiateResize
from sd_bmab.processors.basic import EdgeEnhancement, NoiseAlpha, Img2imgMasking
from sd_bmab.processors.controlnet import LineartNoise


def change_model(name):
	if name is None:
		return
	info = sd_models.get_closet_checkpoint_match(name)
	if info is None:
		debug_print(f'Unknown model: {name}')
		return
	sd_models.reload_model_weights(shared.sd_model, info)


def process_intermediate_step2(context, image):
	all_processors = [
		EdgeEnhancement(),
		IntermidiateResize(),
		Img2imgMasking(),
		NoiseAlpha(),
	]

	processed = image.copy()

	for proc in all_processors:
		result = proc.preprocess(context, processed)
		if result is None or not result:
			continue
		ret = proc.process(context, processed)
		proc.postprocess(context, processed)
		processed = ret

	return processed


class Refiner(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

		self.refiner_opt = {}
		self.enabled = False
		self.checkpoint = None
		self.keep_checkpoint = True
		self.prompt = None
		self.negative_prompt = None
		self.sampler = None
		self.upscaler = None
		self.steps = 20
		self.cfg_scale = 0.7
		self.denoising_strength = 0.75
		self.scale = 1
		self.width = 0
		self.height = 0

		self.base_sd_model = None

	def preprocess(self, context: Context, image: Image):
		self.enabled = context.args['refiner_enabled']
		self.refiner_opt = context.args.get('module_config', {}).get('refiner_opt', {})

		self.checkpoint = self.refiner_opt.get('checkpoint', None)
		self.keep_checkpoint = self.refiner_opt.get('keep_checkpoint', True)
		self.prompt = self.refiner_opt.get('prompt', '')
		self.negative_prompt = self.refiner_opt.get('negative_prompt', '')
		self.sampler = self.refiner_opt.get('sampler', None)
		self.upscaler = self.refiner_opt.get('upscaler', None)
		self.steps = self.refiner_opt.get('steps', None)
		self.cfg_scale = self.refiner_opt.get('cfg_scale', None)
		self.denoising_strength = self.refiner_opt.get('denoising_strength', None)
		self.scale = self.refiner_opt.get('scale', None)
		self.width = self.refiner_opt.get('width', None)
		self.height = self.refiner_opt.get('height', None)

		if self.enabled:
			context.refiner = self

		return self.enabled

	def process(self, context: Context, image: Image):

		if self.checkpoint != constants.checkpoint_default:
			self.base_sd_model = shared.opts.data['sd_model_checkpoint']
			debug_print('base sd model', self.base_sd_model)
			change_model(self.checkpoint)

		output_width = image.width
		output_height = image.height

		if not (self.width == 0 and self.height == 0 and self.scale == 1):
			if (self.width == 0 or self.height == 0) and self.scale != 1:
				output_width = int(image.width * self.scale)
				output_height = int(image.height * self.scale)
			elif self.width != 0 and self.height != 0:
				output_width = self.width
				output_height = self.height

			if image.width != output_width or image.height != output_height:
				LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
				if self.upscaler == constants.fast_upscaler:
					image = image.resize((output_width, output_height), resample=LANCZOS)
				else:
					image = images.resize_image(0, image, output_width, output_height, self.upscaler)

		if not context.is_hires_fix():
			image = process_intermediate_step2(context, image)

		if self.prompt == '':
			self.prompt = context.get_prompt_by_index()
			debug_print('prompt', self.prompt)
		elif self.prompt.find('#!org!#') >= 0:
			current_prompt = context.get_prompt_by_index()
			self.prompt = self.prompt.replace('#!org!#', current_prompt)
			print('Prompt', self.prompt)
		if self.negative_prompt == '':
			self.negative_prompt = context.sdprocessing.negative_prompt
		if self.checkpoint == constants.checkpoint_default:
			self.checkpoint = context.sdprocessing.sd_model
		if self.sampler == constants.sampler_default:
			self.sampler = context.sdprocessing.sampler_name

		seed, subseed = context.get_seeds()
		options = dict(
			seed=seed, subseed=subseed,
			denoising_strength=self.denoising_strength,
			resize_mode=0,
			mask=None,
			mask_blur=4,
			inpainting_fill=1,
			inpaint_full_res=True,
			inpaint_full_res_padding=32,
			inpainting_mask_invert=0,
			initial_noise_multiplier=1.0,
			sd_model=self.checkpoint,
			prompt=self.prompt,
			negative_prompt=self.negative_prompt,
			sampler_name=self.sampler,
			batch_size=1,
			n_iter=1,
			steps=self.steps,
			cfg_scale=self.cfg_scale,
			width=output_width,
			height=output_height,
			restore_faces=False,
			do_not_save_samples=True,
			do_not_save_grid=True,
		)
		context.add_job()

		if LineartNoise.with_refiner(context):
			image = process_img2img(context.sdprocessing, image, options=options, use_cn=True, callback=self.process_callback, callback_args=[self, context])
		else:
			image = process_img2img(context.sdprocessing, image, options=options)

		if not self.keep_checkpoint and self.base_sd_model is not None:
			debug_print('Rollback model')
			change_model(self.base_sd_model)

		return image

	@staticmethod
	def process_callback(self, context, img2img):
		ctx = Context.newContext(self, img2img, context.args, 0)
		ctx.refiner = self
		ln = LineartNoise()
		if ln.preprocess(ctx, None):
			ln.process(ctx, None)
			ln.postprocess(ctx, None)

	def postprocess(self, context: Context, image: Image):
		devices.torch_gc()


class RefinerRollbackModel(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		if context.refiner is None:
			return False
		return context.refiner.keep_checkpoint

	def process(self, context: Context, image: Image):
		debug_print('Rollback model')
		if context.refiner.base_sd_model is not None:
			change_model(context.refiner.base_sd_model)
		return image

	def postprocess(self, context: Context, image: Image):
		pass
