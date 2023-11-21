from modules import scripts
from modules import shared
from modules import script_callbacks
from modules import images

from sd_bmab import parameters
from sd_bmab.base import context, filter

from sd_bmab.pipeline import post
from sd_bmab.pipeline import internal
from sd_bmab import masking
from sd_bmab import ui
from sd_bmab import util
from sd_bmab import controlnet
from sd_bmab.sd_override import override_sd_webui, StableDiffusionProcessingTxt2ImgOv


override_sd_webui()
filter.reload_filters()

if not shared.opts.data.get('bmab_for_developer', False):
	util.check_models()


class BmabExtScript(scripts.Script):

	def __init__(self) -> None:
		super().__init__()
		self.extra_image = []
		self.config = {}
		self.index = 0

	def title(self):
		return 'BMAB'

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

		controlnet.update_controlnet_args(p)

		ctx = context.Context.newContext(self, p, a, 0, hiresfix=True)
		if isinstance(p, StableDiffusionProcessingTxt2ImgOv):
			p.bscript = self
			p.bscript_args = a
			p.initial_noise_multiplier = a.get('txt2img_noise_multiplier', 1)
			p.extra_noise = a.get('txt2img_extra_noise_multiplier', 0)
		else:
			internal.process_img2img(ctx)
		post.process_controlnet(ctx)

	def postprocess_image(self, p, pp, *args):
		self.config, a = parameters.parse_args(args)
		if not a['enabled']:
			ui.final_images.append(pp.image)
			return

		if shared.state.interrupted or shared.state.skipped:
			return

		ctx = context.Context.newContext(self, p, a, self.index)
		with controlnet.PreventControlNet(p, cn_enabled=post.is_controlnet_required(ctx)):
			pp.image = post.process(ctx, pp.image)
			ui.final_images.append(pp.image)
		self.index += 1

	def postprocess(self, p, processed, *args):
		if shared.opts.bmab_show_extends:
			processed.images.extend(self.extra_image)

		post.release()
		masking.release()

	def describe(self):
		return 'This stuff is worth it, you can buy me a beer in return.'

	def resize_image(self, p, a, resize_mode, idx, image, width, height, upscaler_name):
		if not a['enabled']:
			return images.resize_image(resize_mode, image, width, height, upscaler_name=upscaler_name)

		ctx = context.Context.newContext(self, p, a, idx)
		with controlnet.PreventControlNet(p, cn_enabled=internal.is_controlnet_required(ctx)):
			image = internal.process_intermediate_step1(ctx, image)
			image = images.resize_image(resize_mode, image, width, height, upscaler_name=upscaler_name)
			image = internal.process_intermediate_step2(ctx, image)
		return image


script_callbacks.on_ui_settings(ui.on_ui_settings)
