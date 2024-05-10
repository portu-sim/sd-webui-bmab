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
from sd_bmab.sd_override import sd_models
from sd_bmab.compat import check_directory
from sd_bmab.processors.basic import preprocessfilter


check_directory()
override_sd_webui()
filter.reload_filters()

if not shared.opts.data.get('bmab_for_developer', False):
	util.check_models()

if shared.opts.data.get('bmab_additional_checkpoint_path', '') != '':
	sd_models.override()


class BmabExtScript(scripts.Script):

	def __init__(self) -> None:
		super().__init__()
		self.extra_image = []
		self.config = {}
		self.index = 0
		self.stop_generation = False

	def title(self):
		return 'BMAB'

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		return ui.create_ui(self, is_img2img)

	def before_process(self, p, *args):
		self.stop_generation = False
		self.extra_image = []
		ui.final_images = []
		ui.last_process = p
		ui.bmab_script = self
		self.index = 0
		self.config, a = parameters.parse_args(args)
		if not a['enabled']:
			return

		controlnet.update_controlnet_args(p)
		if not hasattr(p, 'context') or p.context is None:
			ctx = context.Context.newContext(self, p, a, 0, hiresfix=True)
			p.context = ctx
			preprocessfilter.run_preprocess_filter(ctx)
			post.process_controlnet(p.context)
			internal.process_img2img(p.context)
			if isinstance(p, StableDiffusionProcessingTxt2ImgOv):
				p.initial_noise_multiplier = a.get('txt2img_noise_multiplier', 1)
				p.extra_noise = a.get('txt2img_extra_noise_multiplier', 0)

	def postprocess_image(self, p, pp, *args):
		self.config, a = parameters.parse_args(args)
		if not a['enabled']:
			ui.final_images.append(pp.image)
			return

		if shared.state.interrupted or shared.state.skipped:
			return

		p.context.index = self.index
		with controlnet.PreventControlNet(p.context, cn_enabled=post.is_controlnet_required(p.context)):
			pp.image = post.process(p.context, pp.image)
			ui.final_images.append(pp.image)
		self.index += 1
		if self.stop_generation:
			shared.state.interrupted = True

	def postprocess(self, p, processed, *args):
		if shared.opts.bmab_show_extends:
			processed.images.extend(self.extra_image)

		post.release()
		masking.release()

	def describe(self):
		return 'This stuff is worth it, you can buy me a beer in return.'

	def resize_image(self, ctx: context.Context, resize_mode, idx, image, width, height, upscaler_name):
		if not ctx.args['enabled']:
			return images.resize_image(resize_mode, image, width, height, upscaler_name=upscaler_name)
		with controlnet.PreventControlNet(ctx, cn_enabled=internal.is_controlnet_required(ctx)):
			image = internal.process_intermediate_before_upscale(ctx, image)
			image = images.resize_image(resize_mode, image, width, height, upscaler_name=upscaler_name)
			image = internal.process_intermediate_after_upscale(ctx, image)
		return image


script_callbacks.on_ui_settings(ui.on_ui_settings)
