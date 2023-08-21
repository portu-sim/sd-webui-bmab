import gradio as gr

from modules import scripts
from modules import shared
from modules import script_callbacks
from modules.processing import StableDiffusionProcessingImg2Img
from modules.processing import StableDiffusionProcessingTxt2Img

from sd_bmab import samplers, util, process, detailing

bmab_version = 'v23.08.22.0'
samplers.override_samplers()


class BmabExtScript(scripts.Script):

	def __init__(self) -> None:
		super().__init__()
		self.extra_image = []
		self.config = {}

	def title(self):
		return 'BMAB Extension'

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		return self._create_ui()

	def parse_args(self, args):
		params = [
			('enabled', False),
			('contrast', 1),
			('brightness', 1),
			('sharpeness', 1),
			('color_temperature', 0),
			('noise_alpha', 0),
			('noise_alpha_final', 0),
			('edge_flavor_enabled', False),
			('edge_low_threadhold', 50),
			('edge_high_threadhold', 200),
			('edge_strength', 0.5),
			('input_image', None),
			('blend_enabled', False),
			('blend_alpha', 1),
			('dino_detect_enabled', False),
			('dino_prompt', ''),
			('face_detailing_enabled', False),
			('face_detailing_before_hresfix_enabled', False),
			('face_lighting', 0.0),
			('hand_detailing_enabled', False),
			('module_config.hand_detailing_opt.block_overscaled_image', True),
			('module_config.hand_detailing_opt.detailing_method', 'subframe'),
			('module_config.hand_detailing.prompt', ''),
			('module_config.hand_detailing.negative_prompt', ''),
			('module_config.hand_detailing.denoising_strength', 0.4),
			('module_config.hand_detailing.cfg_scale', 7),
			('module_config.hand_detailing_opt.auto_upscale', True),
			('module_config.hand_detailing_opt.scale', 2),
			('module_config.hand_detailing_opt.box_threshold', 0.3),
			('module_config.hand_detailing_opt.dilation', 0.1),
			('module_config.hand_detailing.inpaint_full_res', 0),
			('module_config.hand_detailing.inpaint_full_res_padding', 32),
			('module_config.hand_detailing_opt.additional_parameter', ''),
			('resize_by_person_enabled', False),
			('resize_by_person', 0.85),
		]

		ext_params = [
		]

		if len(args) != len(params):
			print('Refresh webui first.')
			raise Exception('Refresh webui first.')

		if args[0]:
			args_list = [(params[idx][0], v) for idx, v in enumerate(args)]
			args_list.extend(ext_params)
			ar = util.get_dict_from_args(args_list, None)
		else:
			params.extend(ext_params)
			ar = util.get_dict_from_args(params, None)

		if self.config:
			cfgarg = util.get_param_from_dict('', self.config)
			ar = util.get_dict_from_args(cfgarg, ar)
			ar['enabled'] = True
		return ar

	def before_process(self, p, *args):
		self.extra_image = []

		prompt, self.config = util.get_config(p.prompt)
		a = self.parse_args(args)
		if not a['enabled']:
			return

		p.prompt = prompt
		p.setup_prompts()

		if a['face_detailing_before_hresfix_enabled'] and isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
			process.process_detailing_before_hires_fix(self, p, a)

		if isinstance(p, StableDiffusionProcessingImg2Img):
			process.process_dino_detect(p, self, a)

	def process_batch(self, p, *args, **kwargs):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		process.process_img2img_process_all(p, self, a)

	def postprocess_image(self, p, pp, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		pp.image = detailing.process_face_detailing(pp.image, self, p, a)
		pp.image = detailing.process_hand_detailing(pp.image, self, p, a)
		pp.image = process.after_process(pp.image, self, p, a)

	def postprocess(self, p, processed, *args):
		if shared.opts.bmab_show_extends:
			processed.images.extend(self.extra_image)

	def before_hr(self, p, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if not process.check_process(a, p):
			return

		process.process_txt2img_hires_fix(p, self, a)

	def describe(self):
		return 'This stuff is worth it, you can buy me a beer in return.'

	def _create_ui(self):
		class ListOv(list):
			def __iadd__(self, x):
				self.append(x)
				return self
		elem = ListOv()
		with gr.Group():
			with gr.Accordion(f'BMAB', open=False):
				with gr.Row():
					with gr.Column():
						elem += gr.Checkbox(label=f'Enabled {bmab_version}', value=False)
				with gr.Row():
					with gr.Tabs(elem_id='tabs'):
						with gr.Tab('Basic', elem_id='basic_tabs'):
							with gr.Row():
								elem += gr.Slider(minimum=0, maximum=2, value=1, step=0.05, label='Contrast')
							with gr.Row():
								elem += gr.Slider(minimum=0, maximum=2, value=1, step=0.05, label='Brightness')
							with gr.Row():
								elem += gr.Slider(minimum=-5, maximum=5, value=1, step=0.1, label='Sharpeness')
							with gr.Row():
								elem += gr.Slider(
									minimum=-2000, maximum=+2000, value=0, step=1, label='Color temperature')
							with gr.Row():
								elem += gr.Slider(minimum=0, maximum=1, value=0, step=0.05, label='Noise alpha')
							with gr.Row():
								elem += gr.Slider(minimum=0, maximum=1, value=0, step=0.05, label='Noise alpha at final stage')
						with gr.Tab('Edge', elem_id='edge_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable edge enhancement', value=False)
							with gr.Row():
								elem += gr.Slider(minimum=1, maximum=255, value=50, step=1, label='Edge low threshold')
								elem +=  gr.Slider(minimum=1, maximum=255, value=200, step=1, label='Edge high threshold')
							with gr.Row():
								elem += gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label='Edge strength')
						with gr.Tab('Imaging', elem_id='imaging_tabs'):
							with gr.Row():
								elem += gr.Image(source='upload')
							with gr.Row():
								elem += gr.Checkbox(label='Blend enabled', value=False)
							with gr.Row():
								elem += gr.Slider(minimum=0, maximum=1, value=1, step=0.05, label='Blend alpha')
							with gr.Row():
								elem += gr.Checkbox(label='Enable dino detect', value=False)
							with gr.Row():
								elem += gr.Textbox(placeholder='1girl:0:0.4:0', visible=True, value='',  label='Prompt')
						with gr.Tab('Face', elem_id='face_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable face detailing', value=False)
							with gr.Row():
								elem += gr.Checkbox(label='Enable face detailing before hires.fix (EXPERIMENTAL)', value=False)
							with gr.Row():
								elem += gr.Slider(minimum=-1, maximum=1, value=0, step=0.05, label='Face lighting (EXPERIMENTAL)')
						with gr.Tab('Hand', elem_id='hand_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable hand detailing (EXPERIMENTAL)', value=False)
								elem += gr.Checkbox(label='Block over-scaled image', value=True)
							with gr.Row():
								elem += gr.Dropdown(label='Method', visible=True, interactive=True, value='subframe', choices=['subframe', 'each hand', 'inpaint each hand', 'at once'])
							with gr.Row():
								elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', visible=True, value='', label='Prompt')
							with gr.Row():
								elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', visible=True, value='', label='Negative Prompt')
							with gr.Row():
								with gr.Column():
									elem += gr.Slider(minimum=0, maximum=1, value=0.4, step=0.01, label='Denoising Strength')
									elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='CFG Scale')
								with gr.Column():
									elem += gr.Checkbox(label='Auto Upscale if Block over-scaled image enabled', value=True)
									elem += gr.Slider(minimum=1, maximum=4, value=2, step=0.01, label='Upscale Ratio')
									elem += gr.Slider(minimum=0, maximum=1, value=0.3, step=0.01, label='Box Threshold')
									elem += gr.Slider(minimum=0, maximum=0.3, value=0.1, step=0.01, label='Box Dilation')
								with gr.Column():
									elem += gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture")
									elem += gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32)
							with gr.Row():
								elem += gr.Textbox(placeholder='Additional parameter for advanced user', visible=True, value='', label='Additional Parameter')
						with gr.Tab('Resize', elem_id='resize_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable resize by person', value=False)
							with gr.Row():
								elem += gr.Slider(minimum=0.80, maximum=0.95, value=0.85, step=0.01, label='Resize by person')
				return elem


def on_ui_settings():
	shared.opts.add_option('bmab_show_extends', shared.OptionInfo(False, 'Show before processing image. (DO NOT ENABLE IN CLOUD)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_test_function', shared.OptionInfo(False, 'Show Test Function', section=('bmab', 'BMAB')))


script_callbacks.on_ui_settings(on_ui_settings)


