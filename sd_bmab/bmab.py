import gradio as gr
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

from sd_bmab import processors
from sd_bmab import masking
from sd_bmab import ui
from sd_bmab.processors import interprocess
from sd_bmab.sd_override import override_sd_webui, StableDiffusionProcessingTxt2ImgOv


override_sd_webui()
filter.reload_filters()


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
			interprocess.process_img2img(ctx)
		processors.process_controlnet(ctx)

	def postprocess_image(self, p, pp, *args):
		self.config, a = parameters.parse_args(args)
		if not a['enabled']:
			ui.final_images.append(pp.image)
			return

		if shared.state.interrupted or shared.state.skipped:
			return

		with PreventControlNet(p):
			ctx = context.Context.newContext(self, p, a, self.index)
			pp.image = processors.process(ctx, pp.image)
			ui.final_images.append(pp.image)
		self.index += 1

	def postprocess(self, p, processed, *args):
		if shared.opts.bmab_show_extends:
			processed.images.extend(self.extra_image)

		processors.release()
		masking.release()

	def describe(self):
		return 'This stuff is worth it, you can buy me a beer in return.'

	def resize_image(self, p, a, resize_mode, idx, image, width, height, upscaler_name):
		if not a['enabled']:
			return images.resize_image(resize_mode, image, width, height, upscaler_name=upscaler_name)

		ctx = context.Context.newContext(self, p, a, idx)
		image = processors.interprocess.process_intermediate_step1(ctx, image)
		image = images.resize_image(resize_mode, image, width, height, upscaler_name=upscaler_name)
		image = processors.interprocess.process_intermediate_step2(ctx, image)
		return image


def on_ui_settings():
	shared.opts.add_option('bmab_debug_print', shared.OptionInfo(False, 'Print debug message.', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_show_extends', shared.OptionInfo(False, 'Show before processing image. (DO NOT ENABLE IN CLOUD)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_test_function', shared.OptionInfo(False, 'Show Test Function', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_keep_original_setting', shared.OptionInfo(False, 'Keep original setting', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_save_image_before_process', shared.OptionInfo(False, 'Save image that before processing', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_save_image_after_process', shared.OptionInfo(False, 'Save image that after processing (some bugs)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_for_developer', shared.OptionInfo(False, 'Show developer hidden function.', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_max_detailing_element', shared.OptionInfo(
		default=0, label='Max Detailing Element', component=gr.Slider, component_args={'minimum': 0, 'maximum': 10, 'step': 1}, section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_detail_full', shared.OptionInfo(True, 'Allways use FULL, VAE type for encode when detail anything. (v1.6.0)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_optimize_vram', shared.OptionInfo(default='None', label='Checkpoint for Person, Face, Hand', component=gr.Radio, component_args={'choices': ['None', 'low vram', 'med vram']}, section=('bmab', 'BMAB')))
	mask_names = masking.list_mask_names()
	shared.opts.add_option('bmab_mask_model', shared.OptionInfo(default=mask_names[0], label='Masking model', component=gr.Radio, component_args={'choices': mask_names}, section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_use_specific_model', shared.OptionInfo(False, 'Use specific model', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_model', shared.OptionInfo(default='', label='Checkpoint for Person, Face, Hand', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_openpose', shared.OptionInfo(default='control_v11p_sd15_openpose_fp16 [73c2b67d]', label='ControlNet openpose model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_lineart', shared.OptionInfo(default='control_v11p_sd15_lineart [43d4be0d]', label='ControlNet lineart model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_inpaint', shared.OptionInfo(default='control_v11p_sd15_inpaint_fp16 [be8bc0ed]', label='ControlNet inpaint model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_tile_resample', shared.OptionInfo(default='control_v11f1e_sd15_tile_fp16 [3b860298]', label='ControlNet tile model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))


script_callbacks.on_ui_settings(on_ui_settings)
