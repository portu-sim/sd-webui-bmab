import gradio as gr

from modules import scripts
from modules import shared
from modules import script_callbacks
from modules.processing import StableDiffusionProcessingImg2Img
from modules.processing import StableDiffusionProcessingTxt2Img

from sd_bmab import samplers, util, process, detailing

bmab_version = 'v23.08.20.2'
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
			('input_image', None),
			('contrast', 1),
			('brightness', 1),
			('sharpeness', 1),
			('color_temperature', 0),
			('noise_alpha', 0),
			('noise_alpha_final', 0),
			('blend_enabled', False),
			('blend_alpha', 1),
			('dino_detect_enabled', False),
			('dino_prompt', ''),
			('edge_flavor_enabled', False),
			('edge_low_threadhold', 50),
			('edge_high_threadhold', 200),
			('edge_strength', 0.5),
			('face_detailing_enabled', False),
			('face_detailing_before_hresfix_enabled', False),
			('face_lighting', 0.0),
			('resize_by_person_enabled', False),
			('resize_by_person', 0.85),
		]

		ext_params = [
			('hand_detailing_enabled', False)
		]

		if len(args) != len(params):
			print('Refresh webui first.')
			raise Exception('Refresh webui first.')

		if args[0]:
			ar = {arg: args[idx] for idx, (arg, d) in enumerate(params)}
			ext = {arg: d for arg, d in ext_params}
			ar.update(ext)
			if self.config:
				ar.update(self.config)
		else:
			ar = {arg: d for arg, d in params}
			ext = {arg: d for arg, d in ext_params}
			ar.update(ext)
			if self.config:
				ar['enabled'] = True
				ar.update(self.config)
		return ar

	def before_process(self, p, *args):
		self.extra_image = []

		prompt, self.config = util.get_config(p.prompt)
		a = self.parse_args(args)
		if not a['enabled']:
			return

		p.prompt = prompt
		p.setup_prompts()

		if a['hand_detailing_enabled'] and isinstance(p, StableDiffusionProcessingImg2Img):
			p.batch_size = 1
			p.width = p.init_images[0].width
			p.height = p.init_images[0].height

		if a['face_detailing_before_hresfix_enabled'] and isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
			process.process_detailing_before_hires_fix(self, p, a)

		if isinstance(p, StableDiffusionProcessingImg2Img):
			process.process_dino_detect(p, self, a)

	def before_process_batch(self, p, *args, **kwargs):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if a['hand_detailing_enabled'] and isinstance(p, StableDiffusionProcessingImg2Img):
			process.process_img2img_break_sampling(self, p, a)

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
		with gr.Group():
			with gr.Accordion(f'BMAB', open=False):
				with gr.Row():
					with gr.Column():
						enabled = gr.Checkbox(label=f'Enabled {bmab_version}', value=False)
				with gr.Row():
					with gr.Tabs(elem_id='tabs'):
						with gr.Tab('Basic', elem_id='basic_tabs'):
							with gr.Row():
								contrast = gr.Slider(minimum=0, maximum=2, value=1, step=0.05, label='Contrast')
							with gr.Row():
								brightness = gr.Slider(minimum=0, maximum=2, value=1, step=0.05, label='Brightness')
							with gr.Row():
								sharpeness = gr.Slider(minimum=-5, maximum=5, value=1, step=0.1, label='Sharpeness')
							with gr.Row():
								color_temperature = gr.Slider(
									minimum=-2000, maximum=+2000, value=0, step=1, label='Color temperature')
							with gr.Row():
								noise_alpha = gr.Slider(minimum=0, maximum=1, value=0, step=0.05, label='Noise alpha')
							with gr.Row():
								noise_alpha_final = gr.Slider(minimum=0, maximum=1, value=0, step=0.05, label='Noise alpha at final stage')
						with gr.Tab('Edge', elem_id='edge_tabs'):
							with gr.Row():
								edge_flavor_enabled = gr.Checkbox(label='Enable edge enhancement', value=False)
							with gr.Row():
								edge_low_threadhold = gr.Slider(minimum=1, maximum=255, value=50, step=1, label='Edge low threshold')
								edge_high_threadhold = gr.Slider(minimum=1, maximum=255, value=200, step=1, label='Edge high threshold')
							with gr.Row():
								edge_strength = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label='Edge strength')
						with gr.Tab('Imaging', elem_id='imaging_tabs'):
							with gr.Row():
								input_image = gr.Image(source='upload')
							with gr.Row():
								blend_enabled = gr.Checkbox(label='Blend enabled', value=False)
							with gr.Row():
								blend_alpha = gr.Slider(minimum=0, maximum=1, value=1, step=0.05, label='Blend alpha')
							with gr.Row():
								dino_detect_enabled = gr.Checkbox(label='Enable dino detect', value=False)
							with gr.Row():
								dino_prompt = gr.Textbox(placeholder='1girl:0:0.4:0', visible=True, value='',  label='Prompt')
						with gr.Tab('Face', elem_id='face_tabs'):
							with gr.Row():
								face_detailing_enabled = gr.Checkbox(label='Enable face detailing', value=False)
							with gr.Row():
								face_detailing_before_hresfix_enabled = gr.Checkbox(label='Enable face detailing before hires.fix (EXPERIMENTAL)', value=False)
							with gr.Row():
								face_lighting = gr.Slider(minimum=-1, maximum=1, value=0, step=0.05, label='Face lighting (EXPERIMENTAL)')
						with gr.Tab('Resize', elem_id='resize_tabs'):
							with gr.Row():
								resize_by_person_enabled = gr.Checkbox(label='Enable resize by person', value=False)
							with gr.Row():
								resize_by_person = gr.Slider(minimum=0.80, maximum=0.95, value=0.85, step=0.01, label='Resize by person')

				return (
					enabled, input_image, contrast, brightness, sharpeness, color_temperature, noise_alpha, noise_alpha_final, blend_enabled, blend_alpha,
					dino_detect_enabled, dino_prompt, edge_flavor_enabled, edge_low_threadhold, edge_high_threadhold, edge_strength,
					face_detailing_enabled, face_detailing_before_hresfix_enabled, face_lighting, resize_by_person_enabled, resize_by_person)


def on_ui_settings():
	shared.opts.add_option('bmab_show_extends', shared.OptionInfo(False, 'Show before processing image', section=('bmab', 'BMAB')))


script_callbacks.on_ui_settings(on_ui_settings)


