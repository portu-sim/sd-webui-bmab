import random
import gradio as gr

from modules import sd_models
from modules import sd_vae
from modules import ui_components
from modules import shared
from modules import extras
from modules import images

from sd_bmab import constants
from sd_bmab import util
from sd_bmab import detectors
from sd_bmab import parameters
from sd_bmab.base import context
from sd_bmab.base import filter
from sd_bmab.base import installer
from sd_bmab import pipeline
from sd_bmab import masking
from sd_bmab.util import debug_print
from sd_bmab.processors.controlnet import Openpose
from sd_bmab.processors.postprocess import Watermark


bmab_version = 'v24.04.21.1'

final_images = []
last_process = None
bmab_script = None
gallery_select_index = 0


def create_ui(bscript, is_img2img):
	class ListOv(list):
		def __iadd__(self, x):
			self.append(x)
			return self

	ui_checkpoints = [constants.checkpoint_default]
	ui_checkpoints.extend([str(x) for x in sd_models.checkpoints_list.keys()])
	ui_vaes = [constants.vae_default]
	ui_vaes.extend([str(x) for x in sd_vae.vae_dict.keys()])

	elem = ListOv()
	with gr.Group():
		with gr.Row():
			with gr.Column():
				elem += gr.Checkbox(label=f'Enable BMAB', value=False)
			with gr.Column():
				btn_stop = ui_components.ToolButton('⏹️', visible=True, interactive=True, tooltip='stop generation', elem_id='bmab_stop_generation')
		with gr.Accordion(f'BMAB Preprocessor', open=False):
			with gr.Row():
				with gr.Tab('Context', id='bmab_context', elem_id='bmab_context_tabs'):
					with gr.Tab('Generic'):
						with gr.Row():
							with gr.Column():
								with gr.Row():
									checkpoint_models = gr.Dropdown(label='CheckPoint', visible=True, value=ui_checkpoints[0], choices=ui_checkpoints)
									elem += checkpoint_models
							with gr.Column():
								with gr.Row():
									vaes_models = gr.Dropdown(label='SD VAE', visible=True, value=ui_vaes[0], choices=ui_vaes)
									elem += vaes_models
						with gr.Row():
							gr.Markdown(constants.checkpoint_description)
						with gr.Row():
							elem += gr.Slider(minimum=0, maximum=1.5, value=1, step=0.001, label='txt2img noise multiplier for hires.fix (EXPERIMENTAL)', elem_id='bmab_txt2img_noise_multiplier')
						with gr.Row():
							elem += gr.Slider(minimum=0, maximum=1, value=0, step=0.01, label='txt2img extra noise multiplier for hires.fix (EXPERIMENTAL)', elem_id='bmab_txt2img_extra_noise_multiplier')
						with gr.Row():
							with gr.Column():
								with gr.Row():
									dd_hiresfix_filter1 = gr.Dropdown(label='Hires.fix filter before upscale', visible=True, value=filter.filters[0], choices=filter.filters)
									elem += dd_hiresfix_filter1
							with gr.Column():
								with gr.Row():
									dd_hiresfix_filter2 = gr.Dropdown(label='Hires.fix filter after upscale', visible=True, value=filter.filters[0], choices=filter.filters)
									elem += dd_hiresfix_filter2
					with gr.Tab('Kohya Hires.fix'):
						with gr.Row():
							with gr.Column():
								elem += gr.Checkbox(label='Enable Kohya hires.fix', value=False)
						with gr.Row():
							gr.HTML(constants.kohya_hiresfix_description)
						with gr.Row():
							elem += gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Stop at, first", value=0.15)
							elem += gr.Slider(minimum=1, maximum=10, step=1, label="Depth, first", value=3)
						with gr.Row():
							elem += gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Stop at, second", value=0.4)
							elem += gr.Slider(minimum=1, maximum=10, step=1, label="Depth, second", value=4)
						with gr.Row():
							elem += gr.Dropdown(['bicubic', 'bilinear', 'nearest', 'nearest-exact'], label='Layer scaler', value='bicubic')
							elem += gr.Slider(minimum=0.1, maximum=1.0, step=0.05, label="Downsampling scale", value=0.5)
							elem += gr.Slider(minimum=1.0, maximum=4.0, step=0.1, label="Upsampling scale", value=2.0)
						with gr.Row():
							elem += gr.Checkbox(label="Smooth scaling", value=True)
							elem += gr.Checkbox(label="Early upsampling", value=False)
							elem += gr.Checkbox(label='Disable for additional passes', value=True)
				with gr.Tab('Resample', id='bmab_resample', elem_id='bmab_resample_tabs'):
					with gr.Row():
						with gr.Column():
							elem += gr.Checkbox(label='Enable self resample', value=False)
						with gr.Column():
							elem += gr.Checkbox(label='Save image before processing', value=False)
					with gr.Row():
						elem += gr.Checkbox(label='Enable resample before hires.fix', value=False)
					with gr.Row():
						with gr.Column():
							with gr.Row():
								resample_models = gr.Dropdown(label='CheckPoint', visible=True, value=ui_checkpoints[0], choices=ui_checkpoints)
								elem += resample_models
						with gr.Column():
							with gr.Row():
								resample_vaes = gr.Dropdown(label='SD VAE', visible=True, value=ui_vaes[0], choices=ui_vaes)
								elem += resample_vaes
					with gr.Row():
						with gr.Column(min_width=100):
							methods = ['txt2img-1pass', 'txt2img-2pass', 'img2img-1pass']
							elem += gr.Dropdown(label='Resample method', visible=True, value=methods[0], choices=methods)
						with gr.Column():
							dd_resample_filter = gr.Dropdown(label='Resample filter', visible=True, value=filter.filters[0], choices=filter.filters)
							elem += dd_resample_filter
					with gr.Row():
						elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Resample prompt')
					with gr.Row():
						elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Resample negative prompt')
					with gr.Row():
						with gr.Column(min_width=100):
							with gr.Row():
								with gr.Column(min_width=50):
									asamplers = [constants.sampler_default]
									asamplers.extend([x.name for x in shared.list_samplers()])
									elem += gr.Dropdown(label='Sampling method', visible=True, value=asamplers[0], choices=asamplers)
								with gr.Column(min_width=50):
									ascheduler = util.get_scueduler_list()
									elem += gr.Dropdown(label='Scheduler', elem_id="resample_scheduler", choices=ascheduler, value=ascheduler[0])
						with gr.Column(min_width=100):
							upscalers = [constants.fast_upscaler]
							upscalers.extend([x.name for x in shared.sd_upscalers])
							elem += gr.Dropdown(label='Upscaler', visible=True, value=upscalers[0], choices=upscalers)
					with gr.Row():
						with gr.Column(min_width=100):
							elem += gr.Slider(minimum=1, maximum=150, value=20, step=1, label='Resample Sampling Steps', elem_id='bmab_resample_steps')
							elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='Resample CFG Scale', elem_id='bmab_resample_cfg_scale')
							elem += gr.Slider(minimum=0, maximum=1, value=0.75, step=0.01, label='Resample Denoising Strength', elem_id='bmab_resample_denoising')
							elem += gr.Slider(minimum=0.0, maximum=2, value=0.5, step=0.05, label='Resample strength', elem_id='bmab_resample_cn_strength')
							elem += gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label='Resample begin', elem_id='bmab_resample_cn_begin')
							elem += gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.01, label='Resample end', elem_id='bmab_resample_cn_end')
				with gr.Tab('Pretraining', id='bmab_pretraining', elem_id='bmab_pretraining_tabs'):
					with gr.Row():
						elem += gr.Checkbox(label='Enable pretraining detailer', value=False)
					with gr.Row():
						elem += gr.Checkbox(label='Enable pretraining before hires.fix', value=False)
					with gr.Row():
						with gr.Column():
							with gr.Row():
								pretraining_checkpoint_models = gr.Dropdown(label='CheckPoint', visible=True, value=ui_checkpoints[0], choices=ui_checkpoints)
								elem += pretraining_checkpoint_models
						with gr.Column():
							with gr.Row():
								pretraining_vaes_models = gr.Dropdown(label='SD VAE', visible=True, value=ui_vaes[0], choices=ui_vaes)
								elem += pretraining_vaes_models
					with gr.Row():
						with gr.Column(min_width=100):
							with gr.Row():
								models = ['Select Model']
								models.extend(util.list_pretraining_models())
								pretraining_models = gr.Dropdown(label='Pretraining Model', visible=True, value=models[0], choices=models, elem_id='bmab_pretraining_models')
								elem += pretraining_models
						with gr.Column(min_width=100):
							dd_pretraining_filter = gr.Dropdown(label='Pretraining filter', visible=True, value=filter.filters[0], choices=filter.filters)
							elem += dd_pretraining_filter
					with gr.Row():
						elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Pretraining prompt')
					with gr.Row():
						elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Pretraining negative prompt')
					with gr.Row():
						with gr.Column(min_width=100):
							with gr.Row():
								with gr.Column(min_width=50):
									asamplers = [constants.sampler_default]
									asamplers.extend([x.name for x in shared.list_samplers()])
									elem += gr.Dropdown(label='Sampling method', visible=True, value=asamplers[0], choices=asamplers)
								with gr.Column(min_width=50):
									ascheduler = util.get_scueduler_list()
									elem += gr.Dropdown(label='Scheduler', elem_id="pretraining_scheduler", choices=ascheduler, value=ascheduler[0])
					with gr.Row():
						with gr.Column(min_width=100):
							elem += gr.Slider(minimum=1, maximum=150, value=20, step=1, label='Pretraining sampling steps', elem_id='bmab_pretraining_steps')
							elem += gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label='Pretraining CFG scale', elem_id='bmab_pretraining_cfg_scale')
							elem += gr.Slider(minimum=0, maximum=1, value=0.75, step=0.01, label='Pretraining denoising Strength', elem_id='bmab_pretraining_denoising')
							elem += gr.Slider(minimum=0, maximum=128, value=4, step=1, label='Pretraining dilation', elem_id='bmab_pretraining_dilation')
							elem += gr.Slider(minimum=0.1, maximum=1, value=0.35, step=0.01, label='Pretraining box threshold', elem_id='bmab_pretraining_box_threshold')
				with gr.Tab('Edge', elem_id='bmab_edge_tabs'):
					with gr.Row():
						elem += gr.Checkbox(label='Enable edge enhancement', value=False)
					with gr.Row():
						elem += gr.Slider(minimum=1, maximum=255, value=50, step=1, label='Edge low threshold')
						elem += gr.Slider(minimum=1, maximum=255, value=200, step=1, label='Edge high threshold')
					with gr.Row():
						elem += gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label='Edge strength')
						gr.Markdown('')
				with gr.Tab('Resize', elem_id='bmab_preprocess_resize_tab'):
					with gr.Row():
						elem += gr.Checkbox(label='Enable resize (intermediate)', value=False)
					with gr.Row():
						elem += gr.Checkbox(label='Resized by person', value=True)
					with gr.Row():
						gr.HTML(constants.resize_description)
					with gr.Row():
						with gr.Column():
							methods = ['stretching', 'inpaint', 'inpaint+lama', 'inpaint_only', 'inpaint_only+lama']
							elem += gr.Dropdown(label='Method', visible=True, value=methods[0], choices=methods)
						with gr.Column():
							align = [x for x in util.alignment.keys()]
							elem += gr.Dropdown(label='Alignment', visible=True, value=align[4], choices=align)
					with gr.Row():
						with gr.Column():
							dd_resize_filter = gr.Dropdown(label='Resize filter', visible=True, value=filter.filters[0], choices=filter.filters)
							elem += dd_resize_filter
						with gr.Column():
							gr.Markdown('')
					with gr.Row():
						elem += gr.Slider(minimum=0.50, maximum=0.95, value=0.85, step=0.01, label='Resize by person intermediate')
					with gr.Row():
						elem += gr.Slider(minimum=0, maximum=1, value=0.75, step=0.01, label='Denoising Strength for inpaint and inpaint+lama', elem_id='bmab_resize_intermediate_denoising')
				with gr.Tab('Refiner', id='bmab_refiner', elem_id='bmab_refiner_tabs'):
					with gr.Row():
						elem += gr.Checkbox(label='Enable refiner', value=False)
					with gr.Row():
						with gr.Column():
							with gr.Row():
								refiner_models = gr.Dropdown(label='CheckPoint for refiner', visible=True, value=ui_checkpoints[0], choices=ui_checkpoints)
								elem += refiner_models
						with gr.Column():
							with gr.Row():
								vaes = [constants.vae_default]
								vaes.extend([str(x) for x in sd_vae.vae_dict.keys()])
								refiner_vaes = gr.Dropdown(label='SD VAE', visible=True, value=ui_vaes[0], choices=ui_vaes)
								elem += refiner_vaes
					with gr.Row():
						elem += gr.Checkbox(label='Use this checkpoint for detailing(Face, Person, Hand)', value=True)
					with gr.Row():
						elem += gr.Textbox(placeholder='prompt. if empty, use main prompt', lines=3, visible=True, value='', label='Prompt')
					with gr.Row():
						elem += gr.Textbox(placeholder='negative prompt. if empty, use main negative prompt', lines=3, visible=True, value='', label='Negative Prompt')
					with gr.Row():
						with gr.Column(min_width=100):
							with gr.Row():
								with gr.Column(min_width=50):
									asamplers = [constants.sampler_default]
									asamplers.extend([x.name for x in shared.list_samplers()])
									elem += gr.Dropdown(label='Sampling method', visible=True, value=asamplers[0], choices=asamplers)
								with gr.Column(min_width=50):
									ascheduler = util.get_scueduler_list()
									elem += gr.Dropdown(label='Scheduler', elem_id="refiner_scheduler", choices=ascheduler, value=ascheduler[0])
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
							elem += gr.Checkbox(label='Enable detect', value=False)
						with gr.Row():
							elem += gr.Textbox(placeholder='1girl', visible=True, value='', label='Prompt')
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
							elem += gr.Checkbox(label='Enable face detailing before upscale', value=False)
						with gr.Row():
							elem += gr.Checkbox(label='Disable extra networks in prompt (LORA, Hypernetwork, ...)', value=False)
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
								with gr.Row():
									with gr.Column(min_width=50):
										face_models = gr.Dropdown(label='CheckPoint for face', visible=True, value=ui_checkpoints[0], choices=ui_checkpoints)
										elem += face_models
									with gr.Column(min_width=50):
										face_vaes = gr.Dropdown(label='SD VAE for face', visible=True, value=ui_vaes[0], choices=ui_vaes)
										elem += face_vaes
								with gr.Row():
									with gr.Column(min_width=50):
										asamplers = [constants.sampler_default]
										asamplers.extend([x.name for x in shared.list_samplers()])
										elem += gr.Dropdown(label='Sampler', elem_id="face_sampler", visible=True, value=asamplers[0], choices=asamplers)
									with gr.Column(min_width=50):
										ascheduler = util.get_scueduler_list()
										elem += gr.Dropdown(label='Scheduler', elem_id="face_scheduler", choices=ascheduler, value=ascheduler[0])
								with gr.Row():
									inpaint_area = gr.Radio(label='Inpaint area', choices=['Whole picture', 'Only masked'], type='value', value='Only masked')
									elem += inpaint_area
								with gr.Row():
									elem += gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32)
								with gr.Row():
									choices = detectors.list_face_detectors()
									elem += gr.Dropdown(label='Detection Model', choices=choices, type='value', value=choices[0])
							with gr.Column():
								elem += gr.Slider(minimum=0, maximum=1, value=0.4, step=0.01, label='Face Denoising Strength', elem_id='bmab_face_denoising_strength')
								elem += gr.Slider(minimum=0, maximum=64, value=4, step=1, label='Face Dilation', elem_id='bmab_face_dilation')
								elem += gr.Slider(minimum=0.1, maximum=1, value=0.35, step=0.01, label='Face Box threshold')
								elem += gr.Checkbox(label='Skip face detailing by area', value=False)
								elem += gr.Slider(minimum=0.0, maximum=3.0, value=0.26, step=0.01, label='Face area (MegaPixel)')
					with gr.Tab('Hand', elem_id='bmab_hand_tabs'):
						with gr.Row():
							elem += gr.Checkbox(label='Enable hand detailing (EXPERIMENTAL)', value=False)
							elem += gr.Checkbox(label='Block over-scaled image', value=True)
						with gr.Row():
							elem += gr.Checkbox(label='Enable best quality (EXPERIMENTAL, Use more GPU)', value=False)
						with gr.Row():
							elem += gr.Dropdown(label='Method', visible=True, interactive=True, value='subframe', choices=['subframe', 'each hand', 'inpaint each hand', 'at once', 'depth hand refiner'])
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
										elem += gr.Radio(label='Hire-fix option for noise', choices=['Both', 'Low res only', 'High res only'], type='value', value='Both')
									with gr.Column():
										gr.Markdown('')
							with gr.Tab('Pose', elem_id='bmab_cn_pose_tabs'):
								with gr.Row():
									elem += gr.Checkbox(label='Enable pose', value=False)
								with gr.Row():
									with gr.Column():
										elem += gr.Slider(minimum=0.0, maximum=2, value=0.3, step=0.05, elem_id='bmab_cn_pose', label='Pose strength')
										elem += gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, elem_id='bmab_cn_pose_begin', label='Pose begin')
										elem += gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, elem_id='bmab_cn_pose_end', label='Pose end')
										elem += gr.Checkbox(label='Face only', value=False)
										poses = ['Random']
										poses.extend(Openpose.list_pose())
										dd_pose = gr.Dropdown(label='Pose Selection', interactive=True, visible=True, value=poses[0], choices=poses)
										elem += dd_pose
									with gr.Column():
										pose_image = gr.Image(elem_id='bmab_pose_image')
		with gr.Accordion(f'BMAB Postprocessor', open=False):
			with gr.Row():
				with gr.Tab('Resize by person', elem_id='bmab_postprocess_resize_tab'):
					with gr.Row():
						elem += gr.Checkbox(label='Enable resize by person', value=False)
						mode = ['Inpaint', 'ControlNet inpaint+lama']
						elem += gr.Dropdown(label='Mode', visible=True, value=mode[0], choices=mode)
					with gr.Row():
						with gr.Column():
							elem += gr.Slider(minimum=0.70, maximum=0.95, value=0.85, step=0.01, label='Resize by person')
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
				with gr.Tab('Filter', id='bmab_final_filter', elem_id='bmab_final_filter_tab'):
					with gr.Row():
						dd_final_filter = gr.Dropdown(label='Final filter', visible=True, value=filter.filters[0], choices=filter.filters)
						elem += dd_final_filter
				with gr.Tab('Watermark', id='bmab_watermark', elem_id='bmab_watermark'):
					elem += gr.Checkbox(label='Watermark enabled', value=False)
					with gr.Row():
						with gr.Column(min_width=100):
							fonts = Watermark.list_fonts()
							elem += gr.Dropdown(label='Watermark Font', visible=True, value=fonts[0], choices=fonts)
							align = [x for x in Watermark.alignment.keys()]
							elem += gr.Dropdown(label='Watermark Alignment', visible=True, value=align[5], choices=align)
							elem += gr.Dropdown(label='Watermark Text Alignment', visible=True, value='left', choices=['left', 'right', 'center'])
							elem += gr.Dropdown(label='Watermark Text Rotate', visible=True, value='0', choices=['0', '90', '180', '270'])
							elem += gr.Textbox(label='Watermark Text Color', visible=True, value='#000000')
							elem += gr.Textbox(label='Watermark Background Color', visible=True, value='#000000')
						with gr.Column(min_width=100):
							elem += gr.Slider(minimum=4, maximum=128, value=12, step=1, label='Font Size')
							elem += gr.Slider(minimum=0, maximum=100, value=100, step=1, label='Transparency')
							elem += gr.Slider(minimum=0, maximum=100, value=0, step=1, label='Background Transparency')
							elem += gr.Slider(minimum=0, maximum=100, value=5, step=1, label='Margin')
					with gr.Row():
						elem += gr.Textbox(placeholder='watermark text here', lines=1, max_lines=10, visible=True, value='', label='Watermark or Image path')
		with gr.Accordion(f'BMAB Refresh, Config, Preset, Installer', open=False):
			with gr.Row():
				configs = parameters.Parameters().list_config()
				config = '' if not configs else configs[0]
				with gr.Tab('Configuration', elem_id='bmab_configuration_tabs'):
					with gr.Row():
						with gr.Column(scale=2):
							with gr.Row():
								config_dd = gr.Dropdown(label='Configuration', visible=True, interactive=True, allow_custom_value=True, value=config, choices=configs)
								elem += config_dd
								load_btn = ui_components.ToolButton('⬇️', visible=True, interactive=True, tooltip='load configuration', elem_id='bmab_load_configuration')
								save_btn = ui_components.ToolButton('⬆️', visible=True, interactive=True, tooltip='save configuration', elem_id='bmab_save_configuration')
								reset_btn = ui_components.ToolButton('🔃', visible=True, interactive=True, tooltip='reset to default', elem_id='bmab_reset_configuration')
						with gr.Column(scale=1):
							gr.Markdown('')
					with gr.Row():
						with gr.Column(scale=1):
							btn_refresh_all = gr.Button('Refresh ALL', visible=True, interactive=True, elem_id='bmab_refresh_all')
						with gr.Column(scale=1):
							gr.Markdown('')
						with gr.Column(scale=1):
							gr.Markdown('')
						with gr.Column(scale=1):
							gr.Markdown('')
				with gr.Tab('Preset', elem_id='bmab_configuration_tabs'):
					with gr.Row():
						with gr.Column(min_width=100):
							gr.Markdown('Preset Loader : preset override UI configuration.')
					with gr.Row():
						presets = parameters.Parameters().list_preset()
						with gr.Column(min_width=100):
							with gr.Row():
								preset_dd = gr.Dropdown(label='Preset', visible=True, interactive=True, allow_custom_value=True, value=presets[0], choices=presets)
								elem += preset_dd
								refresh_btn = ui_components.ToolButton('🔄', visible=True, interactive=True, tooltip='refresh preset', elem_id='bmab_preset_refresh')
				with gr.Tab('Toy', elem_id='bmab_toy_tabs'):
					with gr.Row():
						merge_result = gr.Markdown('Result here')
					with gr.Row():
						random_checkpoint = gr.Button('Merge Random Checkpoint', visible=True, interactive=True, elem_id='bmab_merge_random_checkpoint')
				with gr.Tab('Installer', elem_id='bmab_install_tabs'):
					with gr.Row():
						pkgs = ['GroundingDINO']
						dd_pkg = gr.Dropdown(label='Package', visible=True, value=pkgs[0], choices=pkgs)
						btn_install = ui_components.ToolButton('🔄', visible=True, interactive=True, tooltip='Install package', elem_id='bmab_btn_install')
					with gr.Row():
						markdown_install = gr.Markdown('')
		with gr.Accordion(f'BMAB Testroom', open=False, visible=shared.opts.data.get('bmab_for_developer', False)):
			with gr.Row():
				gallery = gr.Gallery(label='Images', value=[], elem_id='bmab_testroom_gallery')
				result_image = gr.Image(elem_id='bmab_result_image')
			with gr.Row():
				btn_fetch_images = ui_components.ToolButton('🔄', visible=True, interactive=True, tooltip='fetch images', elem_id='bmab_fetch_images')
				btn_process_pipeline = ui_components.ToolButton('▶️', visible=True, interactive=True, tooltip='fetch images', elem_id='bmab_fetch_images')

		gr.Markdown(f'<div style="text-align: right; vertical-align: bottom"><span style="color: green">{bmab_version}</span></div>')

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

		def merge_random_checkpoint(*args):
			def find_random(k, f):
				for v in k:
					if v.startswith(f):
						return v

			result = ''
			checkpoints = [str(x) for x in sd_models.checkpoints_list.keys()]
			target = random.choices(checkpoints, k=3)
			multiplier = random.randrange(10, 90, 1) / 100
			index = random.randrange(0x10000000, 0xFFFFFFFF, 1)
			output = f'bmab_random_{format(index, "08X")}'
			extras.run_modelmerger(None, target[0], target[1], target[2], 'Weighted sum', multiplier, False, output, 'safetensors', 0, None, '', True, True, True, '{}')
			result += f'{output}.safetensors generated<br>'
			for x in range(1, random.randrange(0, 5, 1)):
				checkpoints = [str(x) for x in sd_models.checkpoints_list.keys()]
				br = find_random(checkpoints, f'{output}.safetensors')
				if br is None:
					return
				index = random.randrange(0x10000000, 0xFFFFFFFF, 1)
				output = f'bmab_random_{format(index, "08X")}'
				target = random.choices(checkpoints, k=2)
				multiplier = random.randrange(10, 90, 1) / 100
				extras.run_modelmerger(None, br, target[0], target[1], 'Weighted sum', multiplier, False, output, 'safetensors', 0, None, '', True, True, True, '{}')
				result += f'{output}.safetensors generated<br>'
			debug_print('done')
			return {
				merge_result: {
					'value': result,
					'__type__': 'update'
				}
			}

		def fetch_images(*args):
			global gallery_select_index
			gallery_select_index = 0
			return {
				gallery: {
					'value': final_images,
					'__type__': 'update'
				}
			}

		def process_pipeline(*args):
			config, a = parameters.parse_args(args)
			preview = final_images[gallery_select_index]
			p = last_process
			ctx = context.Context.newContext(bmab_script, p, a, gallery_select_index)
			preview = pipeline.process(ctx, preview)
			images.save_image(
				preview, p.outpath_samples, '',
				p.all_seeds[gallery_select_index], p.all_prompts[gallery_select_index],
				shared.opts.samples_format, p=p, suffix="-testroom")
			return {
				result_image: {
					'value': preview,
					'__type__': 'update'
				}
			}

		refresh_targets = [dd_hiresfix_filter1, dd_hiresfix_filter2, dd_resample_filter, dd_resize_filter, dd_final_filter, dd_pretraining_filter]
		refresh_targets.extend([checkpoint_models, vaes_models, refiner_models, refiner_vaes, face_models, face_vaes, resample_models, resample_vaes])
		refresh_targets.extend([pretraining_checkpoint_models, pretraining_vaes_models, pretraining_models, dd_pose])

		def reload_filter(*args):
			filter.reload_filters()
			inputs = list(args)

			_checkpoints = [constants.checkpoint_default]
			_checkpoints.extend([str(x) for x in sd_models.checkpoints_list.keys()])

			_vaes = [constants.vae_default]
			_vaes.extend([str(x) for x in sd_vae.vae_dict.keys()])

			_pretraining_models = ['Select Model']
			_pretraining_models.extend(util.list_pretraining_models())

			_poses = ['Random']
			_poses.extend(Openpose.list_pose())

			values = [
				filter.filters, filter.filters, filter.filters, filter.filters, filter.filters, filter.filters,
				_checkpoints, _vaes, _checkpoints, _vaes, _checkpoints, _vaes, _checkpoints, _vaes,
				_checkpoints, _vaes, _pretraining_models, _poses
			]

			ret = {
				t: {
					'choices': v,
					'value': v[0] if i not in v else i,
					'__type__': 'update'
				}
				for t, i, v in zip(refresh_targets, inputs, values)
			}

			return ret

		def image_selected(data: gr.SelectData, *args):
			debug_print(data.index)
			global gallery_select_index
			gallery_select_index = data.index

		def hit_install(*args):
			pkg_name = args[0]
			if pkg_name == 'GroundingDINO':
				installer.install_groudingdino()
				msg = f'{pkg_name} installed'
			else:
				msg = 'Nothing installed.'
			return {
				markdown_install: {
					'value': msg,
					'__type__': 'update'
				}
			}

		def stop_process(*args):
			bscript.stop_generation = True
			gr.Info('Waiting for processing done.')

		def pose_selected(*args):
			print(args)
			return Openpose.get_pose(args[0])

		load_btn.click(load_config, inputs=[config_dd], outputs=elem)
		save_btn.click(save_config, inputs=elem, outputs=[config_dd])
		reset_btn.click(reset_config, outputs=elem)
		refresh_btn.click(refresh_preset, outputs=elem)

		random_checkpoint.click(merge_random_checkpoint, outputs=[merge_result])
		btn_fetch_images.click(fetch_images, outputs=[gallery])

		btn_refresh_all.click(
			reload_filter,
			inputs=refresh_targets,
			outputs=refresh_targets,
		)

		btn_process_pipeline.click(process_pipeline, inputs=elem, outputs=[result_image])
		gallery.select(image_selected, inputs=[gallery])

		btn_install.click(hit_install, inputs=[dd_pkg], outputs=[markdown_install])
		btn_stop.click(stop_process)
		dd_pose.select(pose_selected, inputs=[dd_pose], outputs=[pose_image])

	return elem


def on_ui_settings():
	shared.opts.add_option('bmab_debug_print', shared.OptionInfo(False, 'Print debug message.', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_debug_logging', shared.OptionInfo(False, 'Enable developer logging.', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_show_extends', shared.OptionInfo(False, 'Show before processing image. (DO NOT ENABLE IN CLOUD)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_test_function', shared.OptionInfo(False, 'Show Test Function', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_keep_original_setting', shared.OptionInfo(False, 'Keep original setting', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_save_image_before_process', shared.OptionInfo(False, 'Save image that before processing', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_save_image_after_process', shared.OptionInfo(False, 'Save image that after processing (some bugs)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_for_developer', shared.OptionInfo(False, 'Show developer hidden function.', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_use_dino_predict', shared.OptionInfo(False, 'Use GroudingDINO for detecting hand. GroudingDINO should be installed manually.', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_max_detailing_element', shared.OptionInfo(
		default=0, label='Max Detailing Element', component=gr.Slider, component_args={'minimum': 0, 'maximum': 10, 'step': 1}, section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_detail_full', shared.OptionInfo(True, 'Allways use FULL, VAE type for encode when detail anything. (v1.6.0)', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_optimize_vram', shared.OptionInfo(default='None', label='Checkpoint for Person, Face, Hand', component=gr.Radio, component_args={'choices': ['None', 'low vram', 'med vram']}, section=('bmab', 'BMAB')))
	mask_names = masking.list_mask_names()
	shared.opts.add_option('bmab_mask_model', shared.OptionInfo(default=mask_names[0], label='Masking model', component=gr.Radio, component_args={'choices': mask_names}, section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_use_specific_model', shared.OptionInfo(False, 'Use specific model', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_model', shared.OptionInfo(default='', label='Checkpoint for Person, Face, Hand', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_openpose', shared.OptionInfo(default='control_v11p_sd15_openpose_fp16 [73c2b67d]', label='ControlNet openpose model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_lineart', shared.OptionInfo(default='control_v11p_sd15_lineart_fp16 [5c23b17d]', label='ControlNet lineart model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_inpaint', shared.OptionInfo(default='control_v11p_sd15_inpaint_fp16 [be8bc0ed]', label='ControlNet inpaint model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_tile_resample', shared.OptionInfo(default='control_v11f1e_sd15_tile_fp16 [3b860298]', label='ControlNet tile model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))
	shared.opts.add_option('bmab_cn_inpaint_depth_hand', shared.OptionInfo(default='control_sd15_inpaint_depth_hand_fp16 [09456e54]', label='ControlNet tile model', component=gr.Textbox, component_args='', section=('bmab', 'BMAB')))

