import gradio as gr
from PIL import Image

from modules import scripts
from modules import shared
from modules import script_callbacks
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img

from sd_bmab import samplers, util, process, face

bmab_version = 'v23.08.16.1'
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

	def before_component(self, component, **kwargs):
		super().before_component(component, **kwargs)

	def before_process(self, p, *args):
		self.extra_image = []

		prompt, self.config = util.get_config(p.prompt)
		a = self.parse_args(args)
		if not a['enabled']:
			return

		p.prompt = prompt
		p.setup_prompts()

		if isinstance(p, StableDiffusionProcessingImg2Img):
			if a['dino_detect_enabled']:
				if p.image_mask is not None:
					self.extra_image.append(p.init_images[0])
					self.extra_image.append(p.image_mask)
					p.image_mask = util.sam(a['dino_prompt'], p.init_images[0])
					self.extra_image.append(p.image_mask)
				if p.image_mask is None and a['input_image'] is not None:
					mask = util.sam(a['dino_prompt'], p.init_images[0])
					inputimg = Image.fromarray(a['input_image'])
					newpil = Image.new('RGB', p.init_images[0].size)
					newdata = [bdata if mdata == 0 else ndata for mdata, ndata, bdata in zip(mask.getdata(), p.init_images[0].getdata(), inputimg.getdata())]
					newpil.putdata(newdata)
					p.init_images[0] = newpil
					self.extra_image.append(newpil)

	def process_img2img_process_all(self, a, p):
		if isinstance(p, StableDiffusionProcessingImg2Img):
			if p.resize_mode == 2 and len(p.init_images) == 1:
				im = p.init_images[0]
				p.extra_generation_params['BMAB resize image'] = '%s %s' % (p.width, p.height)
				img = util.resize_image(p.resize_mode, im, p.width, p.height)
				self.extra_image.append(img)
				for idx in range(0, len(p.init_latent)):
					p.init_latent[idx] = util.image_to_latent(p, img)

			if process.check_process(a, p):
				if len(p.init_images) == 1:
					img = util.latent_to_image(p.init_latent, 0)
					img = process.process_all(a, p, img)
					self.extra_image.append(img)
					for idx in range(0, len(p.init_latent)):
						p.init_latent[idx] = util.image_to_latent(p, img)
				else:
					for idx in range(0, len(p.init_latent)):
						img = util.latent_to_image(p.init_latent, idx)
						img = process.process_all(a, p, img)
						self.extra_image.append(img)
						p.init_latent[idx] = util.image_to_latent(p, img)

	def process_batch(self, p, *args, **kwargs):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		prompts = kwargs['prompts']
		if p.prompt.find('#') >= 0:
			for idx in range(0, len(prompts)):
				prompts[idx] = process.process_prompt(prompts[idx])
				p.extra_generation_params['BMAB random prompt'] = prompts[idx]
			if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
				p.hr_prompts = prompts

		if a['execute_before_img2img']:
			self.process_img2img_process_all(a, p)

	def parse_args(self, args):
		if len(args) != 20:
			print('Refresh webui first.')
			raise Exception('Refresh webui first.')

		params = (
			('enabled', False),
			('execute_before_img2img', False),
			('input_image', None),
			('contrast', 1),
			('brightness', 1),
			('sharpeness', 1),
			('color_temperature', 0),
			('noise_alpha', 0),
			('blend_enabled', False),
			('blend_alpha', 1),
			('dino_detect_enabled', False),
			('dino_prompt', ''),
			('edge_flavor_enabled', False),
			('edge_low_threadhold', 50),
			('edge_high_threadhold', 200),
			('edge_strength', 0.5),
			('face_lighting_enabled', False),
			('face_lighting', 0.0),
			('resize_by_person_enabled', False),
			('resize_by_person', 0.85)
		)

		if args[0]:
			ar = {arg: args[idx] for idx, (arg, d) in enumerate(params)}
			if self.config:
				ar.update(self.config)
		else:
			ar = {arg: d for arg, d in params}
			if self.config:
				ar['enabled'] = True
				ar.update(self.config)
		return ar

	def postprocess_batch(self, p, *args, **kwargs):
		super().postprocess_batch(p, *args, **kwargs)
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if not a['execute_before_img2img']:
			self.process_img2img_process_all(a, p)

		face.process_face_lighting(a, p, kwargs['images'])

	def postprocess_image(self, p, pp, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if not a['execute_before_img2img']:
			pp.image = process.process_all(a, p, pp.image)
		pp.image = process.after_process(a, p, pp.image)

	def postprocess(self, p, processed, *args):
		if shared.opts.bmab_show_extends:
			processed.images.extend(self.extra_image)

	def before_hr(self, p, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if not process.check_process(a, p):
			return

		if isinstance(p.sampler, samplers.KDiffusionSamplerOv):
			class CallBack(samplers.SamplerCallBack):
				def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning):
					for idx in range(0, len(x)):
						img = util.latent_to_image(x, idx)
						img = process.process_all(self.args, p, img)
						self.script.extra_image.append(img)
						x[idx] = util.image_to_latent(p, img)

			p.sampler.register_callback(CallBack(self, a))

	def describe(self):
		return 'This stuff is worth it, you can buy me a beer in return.'

	def _create_ui(self):
		with gr.Group():
			with gr.Accordion('BMAB', open=False):
				with gr.Row():
					with gr.Column():
						enabled = gr.Checkbox(label='Enabled', value=False)
					with gr.Column():
						execute_before_img2img = gr.Checkbox(label='Process before img2img', value=False)
					with gr.Column():
						gr.Markdown(bmab_version)
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
						with gr.Tab('Edge', elem_id='edge_tabs'):
							with gr.Row():
								edge_flavor_enabled = gr.Checkbox(label='Edge enhancement enabled', value=False)
							with gr.Row():
								edge_low_threadhold = gr.Slider(minimum=1, maximum=255, value=50, step=1,
																label='Edge low threshold')
								edge_high_threadhold = gr.Slider(minimum=1, maximum=255, value=200, step=1,
																 label='Edge high threshold')
							with gr.Row():
								edge_strength = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05,
														  label='Edge strength')
						with gr.Tab('Imaging', elem_id='imaging_tabs'):
							with gr.Row():
								input_image = gr.Image(source='upload')
							with gr.Row():
								blend_enabled = gr.Checkbox(label='Blend enabled', value=False)
							with gr.Row():
								blend_alpha = gr.Slider(minimum=0, maximum=1, value=1, step=0.05, label='Blend alpha')
							with gr.Row():
								dino_detect_enabled = gr.Checkbox(label='Dino detect enabled', value=False)
							with gr.Row():
								dino_prompt = gr.Textbox(placeholder='1girl:0:0.4:0', visible=True, value='',
														 label='Prompt')
						with gr.Tab('Face', elem_id='face_tabs'):
							with gr.Row():
								face_lighting_enabled = gr.Checkbox(label='Enable face lighting', value=False)
							with gr.Row():
								face_lighting = gr.Slider(minimum=-1, maximum=1, value=-0.05, step=0.05,
														  label='Face lighting')
						with gr.Tab('Resize', elem_id='resize_tabs'):
							with gr.Row():
								resize_by_person_enabled = gr.Checkbox(label='Enable resize by person', value=False)
							with gr.Row():
								resize_by_person = gr.Slider(minimum=0.80, maximum=0.95, value=0.85, step=0.01,
															 label='Resize by person')

				return (
					enabled, execute_before_img2img, input_image, contrast, brightness, sharpeness, color_temperature,
					noise_alpha, blend_enabled, blend_alpha,
					dino_detect_enabled, dino_prompt, edge_flavor_enabled, edge_low_threadhold, edge_high_threadhold,
					edge_strength, face_lighting_enabled, face_lighting, resize_by_person_enabled, resize_by_person)


def on_ui_settings():
	shared.opts.add_option('bmab_show_extends', shared.OptionInfo(False, 'Show before processing image', section=('bmab', 'BMAB')))


script_callbacks.on_ui_settings(on_ui_settings)


