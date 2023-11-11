from copy import copy

from modules import scripts
from modules import shared
from modules import script_callbacks
from modules import processing
from modules import img2img
from modules import images

from sd_bmab import parameters
from sd_bmab.base import context, filter
from sd_bmab.util import debug_print

from sd_bmab import pipeline
from sd_bmab import internalpipeline
from sd_bmab import masking
from sd_bmab import ui
from sd_bmab import util
from sd_bmab.sd_override import override_sd_webui, StableDiffusionProcessingTxt2ImgOv


override_sd_webui()
filter.reload_filters()
util.check_models()


class PreventControlNet:
	process_images_inner = processing.process_images_inner
	process_batch = img2img.process_batch

	def __init__(self, p) -> None:
		self._process_images_inner = processing.process_images_inner
		self._process_batch = img2img.process_batch
		self.allow_script_control = None
		self.p = p
		self.all_prompts = copy(p.all_prompts)
		self.all_negative_prompts = copy(p.all_negative_prompts)

	def is_controlnet_used(self):
		if not self.p.script_args:
			return False

		for idx, obj in enumerate(self.p.script_args):
			if 'controlnet' in obj.__class__.__name__.lower():
				if hasattr(obj, 'enabled') and obj.enabled:
					debug_print('Use controlnet True')
					return True
			elif isinstance(obj, dict) and 'module' in obj and obj['enabled']:
				debug_print('Use controlnet True')
				return True

		debug_print('Use controlnet False')
		return False

	def __enter__(self):
		model = self.p.sd_model.model.diffusion_model
		if hasattr(model, '_original_forward'):
			model._old_forward = self.p.sd_model.model.diffusion_model.forward
			model.forward = getattr(model, '_original_forward')

		processing.process_images_inner = PreventControlNet.process_images_inner
		img2img.process_batch = PreventControlNet.process_batch
		if 'control_net_allow_script_control' in shared.opts.data:
			self.allow_script_control = shared.opts.data['control_net_allow_script_control']
			shared.opts.data['control_net_allow_script_control'] = True
		self.multiple_tqdm = shared.opts.data.get('multiple_tqdm', True)
		shared.opts.data['multiple_tqdm'] = False

	def __exit__(self, *args, **kwargs):
		processing.process_images_inner = self._process_images_inner
		img2img.process_batch = self._process_batch
		if 'control_net_allow_script_control' in shared.opts.data:
			shared.opts.data['control_net_allow_script_control'] = self.allow_script_control
		shared.opts.data['multiple_tqdm'] = self.multiple_tqdm
		model = self.p.sd_model.model.diffusion_model
		if hasattr(model, '_original_forward') and hasattr(model, '_old_forward'):
			self.p.sd_model.model.diffusion_model.forward = model._old_forward


class BmabExtScript(scripts.Script):

	def __init__(self) -> None:
		super().__init__()
		self.extra_image = []
		self.config = {}
		self.index = 0

	def title(self):
		return 'BMAB Extension'

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		return ui.create_ui(is_img2img)

	def before_process(self, p, *args):
		self.extra_image = []
		ui.final_images = []
		ui.last_process = p
		ui.bmab_script = self
		self.index = 0
		self.config, a = parameters.parse_args(args)
		if not a['enabled']:
			return

		ctx = context.Context.newContext(self, p, a, 0, hiresfix=True)
		if isinstance(p, StableDiffusionProcessingTxt2ImgOv):
			p.bscript = self
			p.bscript_args = a
			p.initial_noise_multiplier = a.get('txt2img_noise_multiplier', 1)
			p.extra_noise = a.get('txt2img_extra_noise_multiplier', 0)
		else:
			internalpipeline.process_img2img(ctx)
		pipeline.process_controlnet(ctx)

	def postprocess_image(self, p, pp, *args):
		self.config, a = parameters.parse_args(args)
		if not a['enabled']:
			ui.final_images.append(pp.image)
			return

		if shared.state.interrupted or shared.state.skipped:
			return

		with PreventControlNet(p):
			ctx = context.Context.newContext(self, p, a, self.index)
			pp.image = pipeline.process(ctx, pp.image)
			ui.final_images.append(pp.image)
		self.index += 1

	def postprocess(self, p, processed, *args):
		if shared.opts.bmab_show_extends:
			processed.images.extend(self.extra_image)

		pipeline.release()
		masking.release()

	def describe(self):
		return 'This stuff is worth it, you can buy me a beer in return.'

	def resize_image(self, p, a, resize_mode, idx, image, width, height, upscaler_name):
		if not a['enabled']:
			return images.resize_image(resize_mode, image, width, height, upscaler_name=upscaler_name)

		ctx = context.Context.newContext(self, p, a, idx)
		image = internalpipeline.process_intermediate_step1(ctx, image)
		image = images.resize_image(resize_mode, image, width, height, upscaler_name=upscaler_name)
		image = internalpipeline.process_intermediate_step2(ctx, image)
		return image


script_callbacks.on_ui_settings(ui.on_ui_settings)
