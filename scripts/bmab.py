import gradio as gr
from PIL import Image
from functools import partial

from modules import scripts
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.sd_samplers_kdiffusion import KDiffusionSampler

from scripts.util import sam, image_to_latent, latent_to_image, tensor_to_image, image_to_tensor, resize_image
from scripts.process import process_prompt, process_all, check_process, after_process
from scripts.process import process_resize_by_person
from scripts.face import process_face_lighting


class BmabExtScript(scripts.Script):

	def __init__(self) -> None:
		super().__init__()
		self.extra_image = []

	def title(self):
		return 'BMAB Extension'

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		enabled, execute_before_img2img, input_image, contrast, brightness, sharpeness, color_temperature, noise_alpha, blend_enabled, blend_alpha, dino_detect_enabled, dino_prompt, edge_flavor_enabled, edge_low_threadhold, edge_high_threadhold, edge_strength, face_lighting, resize_by_person = self._create_ui()

		self.infotext_fields = (
			(enabled, lambda x: gr.Checkbox.update(value='enabled' in x)),
			(execute_before_img2img, lambda x: gr.Checkbox.update(value='execute_before_img2img' in x)),
			(input_image, 'input image'),
			(contrast, 'contrast value'),
			(brightness, 'brightness value'),
			(sharpeness, 'sharpeness value'),
			(color_temperature, 'color temperature value'),
			(noise_alpha, 'noise alpha value'),
			(blend_enabled, lambda x: gr.Checkbox.update(value='blend_enabled' in x)),
			(blend_alpha, 'blend transparency value'),
			(dino_detect_enabled, lambda x: gr.Checkbox.update(value='dino_detect_enabled' in x)),
			(dino_prompt, 'dino prompt value'),
			(edge_flavor_enabled, lambda x: gr.Checkbox.update(value='edge_flavor_enabled' in x)),
			(edge_low_threadhold, 'edge low threshold value'),
			(edge_high_threadhold, 'edge high threshold value'),
			(edge_strength, 'edge strength value'),
			(face_lighting, 'face lighting value'),
			(resize_by_person, 'resize by person'),
		)

		return [
			enabled, execute_before_img2img, input_image, contrast, brightness, sharpeness, color_temperature,
			noise_alpha, blend_enabled, blend_alpha,
			dino_detect_enabled, dino_prompt,
			edge_flavor_enabled, edge_low_threadhold, edge_high_threadhold, edge_strength,
			face_lighting, resize_by_person
		]

	def before_component(self, component, **kwargs):
		super().before_component(component, **kwargs)

	def before_process(self, p, *args):
		self.extra_image = []
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if isinstance(p, StableDiffusionProcessingImg2Img):
			if a['dino_detect_enabled']:
				if p.image_mask is not None:
					self.extra_image.append(p.init_images[0])
					self.extra_image.append(p.image_mask)
					p.image_mask = sam(a['dino_prompt'], p.init_images[0])
					self.extra_image.append(p.image_mask)
				if p.image_mask is None and a['input_image'] is not None:
					mask = sam(a['dino_prompt'], p.init_images[0])

					mdata = mask.getdata()
					ndata = p.init_images[0].getdata()
					inputimg = Image.fromarray(a['input_image'])
					bdata = inputimg.getdata()

					newdata = []
					for idx in range(0, len(mdata)):
						if mdata[idx] == 0:
							newdata.append(bdata[idx])
						else:
							newdata.append(ndata[idx])

					newpil = Image.new('RGB', p.init_images[0].size)
					newpil.putdata(newdata)
					p.init_images[0] = newpil
					self.extra_image.append(newpil)

	def process_batch(self, p, *args, **kwargs):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		prompts = kwargs['prompts']
		if p.prompt.find('#') >= 0:
			for idx in range(0, len(prompts)):
				prompts[idx] = process_prompt(prompts[idx])
				p.extra_generation_params['BMAB random prompt'] = prompts[idx]
			if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
				p.hr_prompts = prompts

		if not a['execute_before_img2img']:
			return

		if isinstance(p, StableDiffusionProcessingImg2Img):
			if p.resize_mode == 2 and len(p.init_images) == 1:
				im = p.init_images[0]
				p.extra_generation_params['BMAB resize image'] = '%s %s' % (p.width, p.height)
				img = resize_image(p.resize_mode, im, p.width, p.height)
				self.extra_image.append(img)
				for idx in range(0, len(p.init_latent)):
					p.init_latent[idx] = image_to_latent(p, img)

			if check_process(a, p):
				if len(p.init_images) == 1:
					img = latent_to_image(p.init_latent, 0)
					img = process_resize_by_person(a, p, img)
					img = process_all(a, p, img)
					self.extra_image.append(img)
					for idx in range(0, len(p.init_latent)):
						p.init_latent[idx] = image_to_latent(p, img)
				else:
					for idx in range(0, len(p.init_latent)):
						img = latent_to_image(p.init_latent, 0)
						img = process_resize_by_person(a, p, img)
						img = process_all(a, p, img)
						self.extra_image.append(img)
						p.init_latent[idx] = image_to_latent(p, img)

	def parse_args(self, args):
		return {
			'enabled': args[0],
			'execute_before_img2img': args[1],
			'input_image': args[2],
			'contrast': args[3],
			'brightness': args[4],
			'sharpeness': args[5],
			'color_temperature': args[6],
			'noise_alpha': args[7],
			'blend_enabled': args[8],
			'blend_alpha': args[9],
			'dino_detect_enabled': args[10],
			'dino_prompt': args[11],
			'edge_flavor_enabled': args[12],
			'edge_low_threadhold': args[13],
			'edge_high_threadhold': args[14],
			'edge_strength': args[15],
			'face_lighting': args[16],
			'resize_by_person': args[17],
		}

	def postprocess_batch(self, p, *args, **kwargs):
		super().postprocess_batch(p, *args, **kwargs)
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if a['face_lighting'] != 0:
			images = kwargs['images']
			for idx in range(0, len(images)):
				img = tensor_to_image(images[idx])
				img = process_face_lighting(a, p, img)
				images[idx] = image_to_tensor(img)

	def postprocess_image(self, p, pp, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if not a['execute_before_img2img']:
			pp.image = process_all(a, p, pp.image)
		pp.image = after_process(a, p, pp.image)

	def postprocess(self, p, processed, *args):
		# processed.images.extend(self.extra_image)
		pass

	def before_hr(self, p, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if not check_process(a, p):
			return

		if isinstance(p, StableDiffusionProcessingTxt2Img):
			if isinstance(p.sampler, KDiffusionSampler):

				def sample_img2img(self, s, ar, p, x, noise, conditioning, unconditional_conditioning, steps=None,
				                   image_conditioning=None):
					for idx in range(0, len(x)):
						img = latent_to_image(x, 0)
						img = process_resize_by_person(ar, p, img)
						s.extra_image.append(img)
						img = process_all(ar, p, img)
						s.extra_image.append(img)
						x[idx] = image_to_latent(p, img)
					return self.temp(p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning)

				p.sampler.temp = p.sampler.sample_img2img
				p.sampler.sample_img2img = partial(sample_img2img.__get__(p.sampler, KDiffusionSampler), self, a)

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
								input_image = gr.Image(source="upload")
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
								face_lighting = gr.Slider(minimum=-1, maximum=1, value=0, step=0.05,
								                          label='Face lighting')
						with gr.Tab('Resize', elem_id='resize_tabs'):
							with gr.Row():
								resize_by_person = gr.Slider(minimum=0.79, maximum=0.95, value=0.79, step=0.01,
								                             label='Resize by person')

				return (
				enabled, execute_before_img2img, input_image, contrast, brightness, sharpeness, color_temperature,
				noise_alpha, blend_enabled, blend_alpha,
				dino_detect_enabled, dino_prompt, edge_flavor_enabled, edge_low_threadhold, edge_high_threadhold,
				edge_strength, face_lighting, resize_by_person)



