import gradio as gr
from copy import copy

from modules import scripts
from modules import shared
from modules import script_callbacks
from modules import processing
from modules import img2img
from modules import sd_models

from sd_bmab import parameters, util, constants
from sd_bmab.base import context
from sd_bmab.util import debug_print


from sd_bmab import processors
from sd_bmab import detectors
from sd_bmab import masking


bmab_version = 'v23.11.03.0'


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

		ctx = context.Context.newContext(self, p, a, 0, hiresfix=True)
		processors.process_hiresfix(ctx)
		processors.process_img2img(ctx)
		processors.process_controlnet(ctx)

	def postprocess_image(self, p, pp, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if shared.state.interrupted or shared.state.skipped:
			return

		with PreventControlNet(p):
			ctx = context.Context.newContext(self, p, a, self.index)
			pp.image = processors.process(ctx, pp.image)
		self.index += 1

	def postprocess(self, p, processed, *args):
		if shared.opts.bmab_show_extends:
			processed.images.extend(self.extra_image)

		processors.release()
		masking.release()

	def describe(self):
		return 'This stuff is worth it, you can buy me a beer in return.'

	def _create_ui(self, is_img2img):
		class ListOv(list):
			def __iadd__(self, x):
				self.append(x)
				return self
		elem = ListOv()
		with gr.Group():
			elem += gr.Checkbox(label=f'Enable BMAB {bmab_version}', value=False)
			with gr.Accordion(f'BMAB Preprocessor', open=False):
				with gr.Row():
					with gr.Tab('Pretraining', id='bmab_pretraining', elem_id='bmab_pretraining_tabs'):
						with gr.Row():
							elem += gr.Checkbox(label='Enable pretraining detailer (EXPERIMENTAL)', value=False)
						with gr.Column(min_width=100):
							models = ['Select Model']
							models.extend(util.list_pretraining_models())
							elem += gr.Dropdown(label='Pretraining Model', visible=True, value=models[0], choices=models, elem_id='bmab_pretraining_models')
						with gr.Row():
							elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Pretraining prompt')
						with gr.Row():
							elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Pretraining negative prompt')
						with gr.Row():
							with gr.Column(min_width=100):
								asamplers = [constants.sampler_default]
								asamplers.extend([x.name for x in shared.list_samplers()])
								elem += gr.Dropdown(label='Sampling method', visible=True, value=asamplers[0], choices=asamplers)
						with gr.Row():
							with gr.Column(min_width=100):
								elem += gr.Slider(minimum=1, maximum=150, value=20, step=1, label='Pretraining sampling steps', elem_id='bmab_pretraining_steps')
								elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='Pretraining CFG scale', elem_id='bmab_pretraining_cfg_scale')
								elem += gr.Slider(minimum=0, maximum=1, value=0.75, step=0.01, label='Pretraining denoising Strength', elem_id='bmab_pretraining_denoising')
								elem += gr.Slider(minimum=0, maximum=128, value=4, step=1, label='Pretraining dilation', elem_id='bmab_pretraining_dilation')
								elem += gr.Slider(minimum=0.1, maximum=1, value=0.35, step=0.01, label='Pretraining box threshold', elem_id='bmab_pretraining_box_threshold')

					with gr.Tab('Refiner', id='bmab_refiner', elem_id='bmab_refiner_tabs'):
						with gr.Row():
							elem += gr.Checkbox(label='Enable refiner (EXPERIMENTAL)', value=False)
						with gr.Row():
							with gr.Column(min_width=100):
								checkpoints = [constants.checkpoint_default]
								checkpoints.extend([str(x) for x in sd_models.checkpoints_list.keys()])
								elem += gr.Dropdown(label='CheckPoint', visible=True, value=checkpoints[0], choices=checkpoints)
							with gr.Column(min_width=100):
								gr.Markdown('')
						with gr.Row():
							elem += gr.Checkbox(label='Use this checkpoint for detailing(Face, Person, Hand)', value=True)
						with gr.Row():
							elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
						with gr.Row():
							elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
						with gr.Row():
							with gr.Column(min_width=100):
								asamplers = [constants.sampler_default]
								asamplers.extend([x.name for x in shared.list_samplers()])
								elem += gr.Dropdown(label='Sampling method', visible=True, value=asamplers[0], choices=asamplers)
							with gr.Column(min_width=100):
								upscalers = [constants.fast_upscaler]
								upscalers.extend([x.name for x in shared.sd_upscalers])
								elem += gr.Dropdown(label='Upscaler', visible=True, value=upscalers[0], choices=upscalers)
						with gr.Row():
							with gr.Column(min_width=100):
								elem += gr.Slider(minimum=1, maximum=150, value=20, step=1, label='Refiner Sampling Steps', elem_id='bmab_refiner_steps')
								elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='Refiner CFG Scale', elem_id='bmab_refiner_cfg_scale')
								elem += gr.Slider(minimum=0, maximum=1, value=0.75, step=0.01, label='Refiner Denoising Strength', elem_id='bmab_refiner_denoising')
						with gr.Row():
							with gr.Column(min_width=100):
								elem += gr.Slider(minimum=0, maximum=4, value=1, step=0.1, label='Refiner Scale', elem_id='bmab_refiner_scale')
								elem += gr.Slider(minimum=0, maximum=2048, value=0, step=1, label='Refiner Width', elem_id='bmab_refiner_width')
								elem += gr.Slider(minimum=0, maximum=2048, value=0, step=1, label='Refiner Height', elem_id='bmab_refiner_height')
			with gr.Accordion(f'BMAB', open=False):
				with gr.Row():
					with gr.Tabs(elem_id='bmab_tabs'):
						with gr.Tab('Basic', elem_id='bmab_basic_tabs'):
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
						with gr.Tab('Edge', elem_id='bmab_edge_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable edge enhancement', value=False)
							with gr.Row():
								elem += gr.Slider(minimum=1, maximum=255, value=50, step=1, label='Edge low threshold')
								elem += gr.Slider(minimum=1, maximum=255, value=200, step=1, label='Edge high threshold')
							with gr.Row():
								elem += gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label='Edge strength')
								gr.Markdown('')
						with gr.Tab('Imaging', elem_id='bmab_imaging_tabs'):
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
						with gr.Tab('Person', elem_id='bmab_person_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable person detailing for landscape', value=False)
							with gr.Row():
								elem += gr.Checkbox(label='Enable best quality (EXPERIMENTAL, Use more GPU)', value=False)
								elem += gr.Checkbox(label='Force upscale ratio 1:1 without area limit', value=False)
							with gr.Row():
								elem += gr.Checkbox(label='Block over-scaled image', value=True)
								elem += gr.Checkbox(label='Auto Upscale if Block over-scaled image enabled', value=True)
							with gr.Row():
								with gr.Column(min_width=100):
									elem += gr.Slider(minimum=1, maximum=8, value=4, step=0.01, label='Upscale Ratio')
									elem += gr.Slider(minimum=0, maximum=20, value=3, step=1, label='Dilation mask')
									elem += gr.Slider(minimum=0.01, maximum=1, value=0.1, step=0.01, label='Large person area limit')
									elem += gr.Slider(minimum=0, maximum=20, value=1, step=1, label='Limit')
									elem += gr.Slider(minimum=0, maximum=2, value=1, step=0.01, visible=shared.opts.data.get('bmab_test_function', False), label='Background color (HIDDEN)')
									elem += gr.Slider(minimum=0, maximum=30, value=0, step=1, visible=shared.opts.data.get('bmab_test_function', False), label='Background blur (HIDDEN)')
								with gr.Column(min_width=100):
									elem += gr.Slider(minimum=0, maximum=1, value=0.4, step=0.01, label='Denoising Strength')
									elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='CFG Scale')
									gr.Markdown('')
						with gr.Tab('Face', elem_id='bmab_face_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable face detailing', value=False)
							with gr.Row():
								elem += gr.Checkbox(label='Enable face detailing before hires.fix (EXPERIMENTAL)', value=False)
							with gr.Row():
								elem += gr.Checkbox(label='Enable best quality (EXPERIMENTAL, Use more GPU)', value=False)
							with gr.Row():
								with gr.Column(min_width=100):
									elem += gr.Dropdown(label='Face detailing sort by', choices=['Score', 'Size', 'Left', 'Right', 'Center'], type='value', value='Score')
								with gr.Column(min_width=100):
									elem += gr.Slider(minimum=0, maximum=20, value=1, step=1, label='Limit')
							with gr.Tab('Face1', elem_id='bmab_face1_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Tab('Face2', elem_id='bmab_face2_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Tab('Face3', elem_id='bmab_face3_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Tab('Face4', elem_id='bmab_face4_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Tab('Face5', elem_id='bmab_face5_tabs'):
								with gr.Row():
									elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
								with gr.Row():
									elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
							with gr.Row():
								with gr.Tab('Parameters', elem_id='bmab_parameter_tabs'):
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
							with gr.Row():
								with gr.Column(min_width=100):
									choices = detectors.list_face_detectors()
									elem += gr.Dropdown(label='Detection Model', choices=choices, type='value', value=choices[0])
								with gr.Column():
									gr.Markdown('')
						with gr.Tab('Hand', elem_id='bmab_hand_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable hand detailing (EXPERIMENTAL)', value=False)
								elem += gr.Checkbox(label='Block over-scaled image', value=True)
							with gr.Row():
								elem += gr.Checkbox(label='Enable best quality (EXPERIMENTAL, Use more GPU)', value=False)
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
						with gr.Tab('ControlNet', elem_id='bmab_controlnet_tabs'):
							with gr.Row():
								elem += gr.Checkbox(label='Enable ControlNet access', value=False)
							with gr.Row():
								elem += gr.Checkbox(label='Process with BMAB refiner', value=False)
							with gr.Row():
								with gr.Tab('Noise', elem_id='bmab_cn_noise_tabs'):
									with gr.Row():
										elem += gr.Checkbox(label='Enable noise', value=False)
									with gr.Row():
										with gr.Column():
											elem += gr.Slider(minimum=0.0, maximum=2, value=0.4, step=0.05, elem_id='bmab_cn_noise', label='Noise strength')
											elem += gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, elem_id='bmab_cn_noise_begin', label='Noise begin')
											elem += gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.01, elem_id='bmab_cn_noise_end', label='Noise end')
										with gr.Column():
											gr.Markdown('')
			with gr.Accordion(f'BMAB Postprocessor', open=False):
				with gr.Row():
					with gr.Tab('Resize by person', elem_id='bmab_postprocess_resize_tab'):
						with gr.Row():
							elem += gr.Checkbox(label='Enable resize by person', value=False)
							mode = [constants.resize_mode_default, 'Inpaint', 'ControlNet inpaint+lama']
							elem += gr.Dropdown(label='Mode', visible=True, value=mode[0], choices=mode)
						with gr.Row():
							with gr.Column():
								elem += gr.Slider(minimum=0.80, maximum=0.95, value=0.85, step=0.01, label='Resize by person')
							with gr.Column():
								elem += gr.Slider(minimum=0, maximum=1, value=0.6, step=0.01, label='Denoising Strength for Inpaint, ControlNet')
						with gr.Row():
							with gr.Column():
								gr.Markdown('')
							with gr.Column():
								elem += gr.Slider(minimum=4, maximum=128, value=30, step=1, label='Mask Dilation')
					with gr.Tab('Upscale', elem_id='bmab_postprocess_upscale_tab'):
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
			with gr.Accordion(f'BMAB Config & Preset', open=False):
				with gr.Row():
					configs = parameters.Parameters().list_config()
					config = '' if not configs else configs[0]
					with gr.Tab('Configuration', elem_id='bmab_configuration_tabs'):
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
					with gr.Tab('Preset', elem_id='bmab_configuration_tabs'):
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
	shared.opts.add_option('bmab_debug_print', shared.OptionInfo(False, 'Print debug message.', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_show_extends', shared.OptionInfo(False, 'Show before processing image. (DO NOT ENABLE IN CLOUD)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_test_function', shared.OptionInfo(False, 'Show Test Function', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_keep_original_setting', shared.OptionInfo(False, 'Keep original setting', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_save_image_before_process', shared.OptionInfo(False, 'Save image that before processing', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_save_image_after_process', shared.OptionInfo(False, 'Save image that after processing (some bugs)', section=('bmab', 'BMAB')))
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


script_callbacks.on_ui_settings(on_ui_settings)
