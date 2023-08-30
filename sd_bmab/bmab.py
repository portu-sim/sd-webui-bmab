import gradio as gr

from modules import scripts
from modules import shared
from modules import script_callbacks
from modules.processing import StableDiffusionProcessingImg2Img
from modules.processing import StableDiffusionProcessingTxt2Img

from sd_bmab import samplers, dinosam, process, detailing, parameters, util, controlnet, constants

bmab_version = 'v23.08.30.3'
samplers.override_samplers()


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
		return self._create_ui(is_img2img)

	def parse_args(self, args):
		self.config = parameters.Parameters().load_preset(args)
		ar = parameters.Parameters().get_dict(args, self.config)
		return ar

	def before_process(self, p, *args):
		self.extra_image = []
		self.index = 0
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if isinstance(p, StableDiffusionProcessingTxt2Img):
			process.override_sample(self, p, a)

		controlnet.process_controlnet(self, p, a)

		if isinstance(p, StableDiffusionProcessingImg2Img):
			process.process_dino_detect(p, self, a)

	def process_batch(self, p, *args, **kwargs):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
			a['max_area'] = p.hr_upscale_to_x * p.hr_upscale_to_y

		process.process_img2img_process_all(p, self, a)

	def postprocess_image(self, p, pp, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if shared.state.interrupted or shared.state.skipped:
			return

		modelname = None
		if shared.opts.bmab_use_specific_model:
			modelname = shared.opts.data['sd_model_checkpoint']
			util.change_model(shared.opts.bmab_model)

		pp.image = process.process_upscale_before_detailing(pp.image, self, p, a)
		pp.image = detailing.process_person_detailing(pp.image, self, p, a)
		pp.image = detailing.process_face_detailing(pp.image, self, p, a)
		pp.image = detailing.process_hand_detailing(pp.image, self, p, a)
		pp.image = process.process_upscale_after_detailing(pp.image, self, p, a)
		pp.image = process.after_process(pp.image, self, p, a)

		if modelname is not None:
			util.change_model(modelname)

		self.index += 1

	def postprocess(self, p, processed, *args):
		if shared.opts.bmab_show_extends:
			processed.images.extend(self.extra_image)
		dinosam.release()

	def describe(self):
		return 'This stuff is worth it, you can buy me a beer in return.'

	def _create_ui(self, is_img2img):
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
								with gr.Column():
									elem += gr.Slider(minimum=0, maximum=2, value=1, step=0.05, label='Contrast')
									elem += gr.Slider(minimum=0, maximum=2, value=1, step=0.05, label='Brightness')
									elem += gr.Slider(minimum=-5, maximum=5, value=1, step=0.1, label='Sharpeness')
									elem += gr.Slider(minimum=0, maximum=2, value=1, step=0.01, label='Color')
								with gr.Column():
									elem += gr.Slider(minimum=-2000, maximum=+2000, value=0, step=1, label='Color temperature')
									elem += gr.Slider(minimum=0, maximum=1, value=0, step=0.05, label='Noise alpha')
									elem += gr.Slider(minimum=0, maximum=1, value=0, step=0.05, label='Noise alpha at final stage')
						with gr.Tab('Edge', elem_id='edge_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable edge enhancement', value=False)
							with gr.Row():
								elem += gr.Slider(minimum=1, maximum=255, value=50, step=1, label='Edge low threshold')
								elem += gr.Slider(minimum=1, maximum=255, value=200, step=1, label='Edge high threshold')
							with gr.Row():
								elem += gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label='Edge strength')
								gr.Markdown('')
						with gr.Tab('Imaging', elem_id='imaging_tabs'):
							with gr.Row():
								elem += gr.Image(source='upload', type='pil')
							with gr.Row():
								elem += gr.Checkbox(label='Blend enabled', value=False)
							with gr.Row():
								with gr.Column():
									elem += gr.Slider(minimum=0, maximum=1, value=1, step=0.05, label='Blend alpha')
								with gr.Column():
									gr.Markdown('')
							with gr.Row():
								elem += gr.Checkbox(label='Enable dino detect', value=False)
							with gr.Row():
								elem += gr.Textbox(placeholder='1girl', visible=True, value='',  label='Prompt')
						with gr.Tab('Person', elem_id='person_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable person detailing for landscape', value=False)
							with gr.Row():
								elem += gr.Checkbox(label='Block over-scaled image', value=True)
								elem += gr.Checkbox(label='Auto Upscale if Block over-scaled image enabled', value=True)
							with gr.Row():
								with gr.Column(min_width=100):
									elem += gr.Slider(minimum=1, maximum=8, value=4, step=0.01, label='Upscale Ratio')
									elem += gr.Slider(minimum=0, maximum=20, value=3, step=1, label='Dilation mask')
									elem += gr.Slider(minimum=0.01, maximum=1, value=0.1, step=0.01, label='Large person area limit')
									elem += gr.Slider(minimum=0, maximum=20, value=1, step=1, label='Limit')
								with gr.Column(min_width=100):
									elem += gr.Slider(minimum=0, maximum=1, value=0.4, step=0.01, label='Denoising Strength')
									elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='CFG Scale')
									gr.Markdown('')
						with gr.Tab('Face', elem_id='face_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable face detailing', value=False)
							with gr.Row():
								elem += gr.Checkbox(label='Enable face detailing before hires.fix (EXPERIMENTAL)', value=False)
							with gr.Row():
								with gr.Column(min_width=100):
									elem += gr.Dropdown(label='Face detailing sort by', choices=['Score', 'Size', 'Left', 'Right'], type='value', value='Score')
								with gr.Column(min_width=100):
									elem += gr.Slider(minimum=0, maximum=20, value=1, step=1, label='Limit')
							with gr.Tab('Face1', elem_id='face1_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Tab('Face2', elem_id='face2_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Tab('Face3', elem_id='face3_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Tab('Face4', elem_id='face4_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Tab('Face5', elem_id='face5_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Row():
								with gr.Tab('Parameters', elem_id='parameter_tabs'):
									with gr.Row():
										elem += gr.Checkbox(label='Overide Parameters', value=False)
									with gr.Row():
										with gr.Column(min_width=100):
											elem += gr.Slider(minimum=64, maximum=2048, value=512, step=8, label='Width')
											elem += gr.Slider(minimum=64, maximum=2048, value=512, step=8, label='Height')
										with gr.Column(min_width=100):
											elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='CFG Scale')
											elem += gr.Slider(minimum=1, maximum=150, value=20, step=1, label='Steps')
											elem += gr.Slider(minimum=0, maximum=64, value=4, step=1, label='Mask Blur')
							with gr.Row():
								with gr.Column(min_width=100):
									asamplers = [constants.sampler_default]
									asamplers.extend([x.name for x in shared.list_samplers()])
									elem += gr.Dropdown(label='Sampler', visible=True, value=asamplers[0], choices=asamplers)
									inpaint_area = gr.Radio(label='Inpaint area', choices=['Whole picture', 'Only masked'], type='value', value='Only masked')
									elem += inpaint_area
									elem += gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32)
								with gr.Column():
									elem += gr.Slider(minimum=0, maximum=1, value=0.4, step=0.01, label='Denoising Strength')
									elem += gr.Slider(minimum=0, maximum=64, value=4, step=1, label='Dilation')
									elem += gr.Slider(minimum=0.1, maximum=1, value=0.35, step=0.01, label='Box threshold')
						with gr.Tab('Hand', elem_id='hand_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable hand detailing (EXPERIMENTAL)', value=False)
								elem += gr.Checkbox(label='Block over-scaled image', value=True)
							with gr.Row():
								elem += gr.Dropdown(label='Method', visible=True, interactive=True, value='subframe', choices=['subframe', 'each hand', 'inpaint each hand', 'at once'])
							with gr.Row():
								elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
							with gr.Row():
								elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Row():
								with gr.Column():
									elem += gr.Slider(minimum=0, maximum=1, value=0.4, step=0.01, label='Denoising Strength')
									elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='CFG Scale')
									elem += gr.Checkbox(label='Auto Upscale if Block over-scaled image enabled', value=True)
								with gr.Column():
									elem += gr.Slider(minimum=1, maximum=4, value=2, step=0.01, label='Upscale Ratio')
									elem += gr.Slider(minimum=0, maximum=1, value=0.3, step=0.01, label='Box Threshold')
									elem += gr.Slider(minimum=0, maximum=0.3, value=0.1, step=0.01, label='Box Dilation')
							with gr.Row():
								inpaint_area = gr.Radio(label='Inpaint area', choices=['Whole picture', 'Only masked'], type='value', value='Whole picture')
								elem += inpaint_area
							with gr.Row():
								with gr.Column():
									elem += gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32)
								with gr.Column():
									gr.Markdown('')
							with gr.Row():
								elem += gr.Textbox(placeholder='Additional parameter for advanced user', visible=True, value='', label='Additional Parameter')
						with gr.Tab('Resize', elem_id='resize_tabs'):
							with gr.Row():
								with gr.Tab('Resize by person', elem_id='resize1_tab'):
									with gr.Row():
										elem += gr.Checkbox(label='Enable resize by person', value=False)
									with gr.Row():
										with gr.Column():
											elem += gr.Slider(minimum=0.80, maximum=0.95, value=0.85, step=0.01, label='Resize by person')
										with gr.Column():
											gr.Markdown('')
							with gr.Row():
								with gr.Tab('Upscale', elem_id='resize2_tab'):
									with gr.Row():
										with gr.Column(min_width=100):
											elem += gr.Checkbox(label='Enable upscale at final stage', value=False)
											elem += gr.Checkbox(label='Detailing after upscale', value=True)
										with gr.Column(min_width=100):
											gr.Markdown('')
									with gr.Row():
										with gr.Column(min_width=100):
											upscalers = [x.name for x in shared.sd_upscalers]
											elem += gr.Dropdown(label='Upscaler', visible=True, value=upscalers[0], choices=upscalers)
											elem += gr.Slider(minimum=1, maximum=4, value=1.5, step=0.1, label='Upscale ratio')
						with gr.Tab('ControlNet', elem_id='controlnet_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable ControlNet access (EXPERIMENTAL, TESTING)', value=False)
							with gr.Row():
								with gr.Tab('Resize', elem_id='cn_resize_tabs'):
									with gr.Row():
										elem += gr.Checkbox(label='Enable resize by person', value=False)
									with gr.Row():
										with gr.Column():
											elem += gr.Slider(minimum=0.1, maximum=0.95, value=0.4, step=0.01, elem_id='cn_resize', label='Resize person using openpose')
										with gr.Column():
											gr.Markdown('')
							with gr.Row():
								with gr.Tab('Noise', elem_id='cn_noise_tabs'):
									with gr.Row():
										elem += gr.Checkbox(label='Enable noise (EXPERIMENTAL)', value=False)
									with gr.Row():
										with gr.Column():
											elem += gr.Slider(minimum=0.0, maximum=2, value=0.4, step=0.05, elem_id='cn_noise', label='Noise strength')
										with gr.Column():
											gr.Markdown('')
						with gr.Tab('Config', elem_id='config_tab'):
							configs = parameters.Parameters().list_config()
							config = '' if not configs else configs[0]
							with gr.Row():
								with gr.Tab('Configuration', elem_id='configuration_tabs'):
									with gr.Row():
										with gr.Column(min_width=100):
											config_dd = gr.Dropdown(label='Configuration', visible=True, interactive=True, allow_custom_value=True, value=config, choices=configs)
											elem += config_dd
										with gr.Column(min_width=100):
											gr.Markdown('')
										with gr.Column(min_width=100):
											gr.Markdown('')
									with gr.Row():
										with gr.Column(min_width=100):
											load_btn = gr.Button('Load', visible=True, interactive=True)
										with gr.Column(min_width=100):
											save_btn = gr.Button('Save', visible=True, interactive=True)
										with gr.Column(min_width=100):
											reset_btn = gr.Button('Reset', visible=True, interactive=True)
							with gr.Row():
								with gr.Tab('Preset', elem_id='configuration_tabs'):
									with gr.Row():
										with gr.Column(min_width=100):
											gr.Markdown('Preset Loader : preset override UI configuration.')
									with gr.Row():
										presets = parameters.Parameters().list_preset()
										with gr.Column(min_width=100):
											preset_dd = gr.Dropdown(label='Preset', visible=True, interactive=True, allow_custom_value=True, value=presets[0], choices=presets)
											elem += preset_dd
											refresh_btn = gr.Button('Refresh', visible=True, interactive=True)

			def load_config(*args):
				name = args[0]
				ret = parameters.Parameters().load_config(name)
				return ret

			def save_config(*args):
				name = parameters.Parameters().get_save_config_name(args)
				parameters.Parameters().save_config(args)
				return {
					config_dd: {
						'choices': parameters.Parameters().list_config(),
						'value': name,
						'__type__': 'update'
					}
				}

			def reset_config(*args):
				return parameters.Parameters().get_default()

			def refresh_preset(*args):
				return {
					preset_dd: {
						'choices': parameters.Parameters().list_preset(),
						'value': 'None',
						'__type__': 'update'
					}
				}

			load_btn.click(load_config, inputs=[config_dd], outputs=elem)
			save_btn.click(save_config, inputs=elem, outputs=[config_dd])
			reset_btn.click(reset_config, outputs=elem)
			refresh_btn.click(refresh_preset, outputs=elem)

		return elem


def on_ui_settings():
	shared.opts.add_option('bmab_show_extends', shared.OptionInfo(False, 'Show before processing image. (DO NOT ENABLE IN CLOUD)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_test_function', shared.OptionInfo(False, 'Show Test Function', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_keep_original_setting', shared.OptionInfo(False, 'Keep original setting', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_detailing_method', shared.OptionInfo('Grounding DINO', 'Detailing method', component=gr.Radio, component_args={'choices': ('Grounding DINO', 'YOLO')}, section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_max_detailing_element', shared.OptionInfo(
		default=0, label='Max Detailing Element', component=gr.Slider, component_args={'minimum': 0, 'maximum': 10, 'step': 1}, section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_use_specific_model', shared.OptionInfo(False, 'Use specific model', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_model', shared.OptionInfo(default='', label='Checkpoint for Person, Face, Hand', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))


script_callbacks.on_ui_settings(on_ui_settings)


