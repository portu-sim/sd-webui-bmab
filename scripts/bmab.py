import cv2
import torch
import numpy as np

import gradio as gr
from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance
from functools import partial

from modules import scripts
from modules import shared
from modules import devices
from modules import images
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.sd_samplers import sample_to_image


def image_to_latent(p, img):
	image = np.array(img).astype(np.float32) / 255.0
	image = np.moveaxis(image, 2, 0)
	batch_images = np.expand_dims(image, axis=0).repeat(1, axis=0)
	image = torch.from_numpy(batch_images)
	image = 2. * image - 1.
	image = image.to(shared.device, dtype=devices.dtype_vae)
	return p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(image))


def latent_to_image(x, index=0):
	img = sample_to_image(x, index, approximation=0)
	return img


def resize_image(resize_mode, im, width, height, upscaler_name=None):
	print(resize_mode, im, width, height, upscaler_name)

	if resize_mode == 2:
		vwidth = im.width
		vheight = height
		res = Image.new("RGB", (vwidth, vheight))
		dw = (vwidth - im.width) // 2
		dh = (vheight - im.height)
		res.paste(im, (dw, dh))
		if dh > 0:
			res.paste(im.resize((vwidth, dh), box=(0, 0, vwidth, 0)), box=(0, 0))

		im = res
		vwidth = width
		vheight = height
		res = Image.new("RGB", (vwidth, vheight))
		dw = (vwidth - im.width) // 2
		dh = (vheight - im.height)
		res.paste(im, (dw, dh))

		if dw > 0:
			res.paste(im.resize((dw, height), box=(0, 0, 0, height)), box=(0, 0))
			res.paste(im.resize((dw, height), box=(im.width, 0, im.width, height)),
					  box=(im.width + dw, 0))

		return res

	return images.resize_image(resize_mode, im, width, height, upscaler_name)


def generate_noise(width, height):
	img_1 = np.zeros([height, width, 3], dtype=np.uint8)
	# Generate random Gaussian noise
	mean = 0
	stddev = 180
	r, g, b = cv2.split(img_1)
	cv2.randn(r, mean, stddev)
	cv2.randn(g, mean, stddev)
	cv2.randn(b, mean, stddev)
	img = cv2.merge([r, g, b])
	pil_image = Image.fromarray(img, mode='RGB')
	return pil_image


def edge_flavor(pil, canny_th1: int, canny_th2: int, strength: float):
	numpy_image = np.array(pil)
	base = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
	arcanny = cv2.Canny(base, canny_th1, canny_th2)
	canny = Image.fromarray(arcanny)
	canny = ImageOps.invert(canny)

	mdata = canny.getdata()
	ndata = pil.getdata()

	newdata = []
	for idx in range(0, len(mdata)):
		if mdata[idx] == 0:
			newdata.append((0, 0, 0))
		else:
			newdata.append(ndata[idx])

	newbase = Image.new('RGB', pil.size)
	newbase.putdata(newdata)
	return Image.blend(pil, newbase, alpha=strength).convert("RGB")


def load_ddsd():
	for sc in scripts.scripts_data:
		if sc.script_class.__module__ == 'ddsd.py':
			return sc.module
	return None


def sam(prompt, input_image):
	mod = load_ddsd()
	if mod is None:
		print('Not found ddsd module.')
		return None

	detailer_sam_model = 'sam_vit_b_01ec64.pth'
	detailer_dino_model = 'groundingdino_swint_ogc.pth'
	mask = mod.dino_detect_from_prompt(
		prompt, detailer_sam_model,
		detailer_dino_model, input_image,
		True,
		'Inner', None
	)
	devices.torch_gc()
	return Image.fromarray(mask)


def check_process(args, p):
	return args['edge_flavor_enabed'] or args['noise_alpha'] or \
		(args['blend_enabed'] and args['input_image'] is not None and 0 <= args['blend_alpha'] <= 1)


def process_all(args, p, bgimg):
	if args['edge_flavor_enabed']:
		print('edge flavor', args['edge_low_threadhold'], args['edge_high_threadhold'], args['edge_strength'])
		bgimg = edge_flavor(bgimg, args['edge_low_threadhold'], args['edge_high_threadhold'], args['edge_strength'])

	if args['noise_alpha'] != 0:
		img_noise = generate_noise(bgimg.size[0], bgimg.size[1])
		bgimg = Image.blend(bgimg, img_noise, alpha=args['noise_alpha'])

	if args['blend_enabed'] and args['input_image'] is not None and 0 <= args['blend_alpha'] <= 1:
		blend = Image.fromarray(args['input_image'], mode='RGB')
		img = Image.new(mode='RGB', size=bgimg.size)
		img.paste(bgimg, (0, 0))
		img.paste(blend)
		bgimg = Image.blend(bgimg, img, alpha=args['blend_alpha'])

	return bgimg


def after_process(args, p, bgimg):
	if args['contrast']:
		enhancer = ImageEnhance.Contrast(bgimg)
		print('contrast', args['contrast'])
		bgimg = enhancer.enhance(args['contrast'])

	if args['brightness']:
		enhancer = ImageEnhance.Brightness(bgimg)
		print('brightness', args['brightness'])
		bgimg = enhancer.enhance(args['brightness'])

	if args['sharpeness']:
		enhancer = ImageEnhance.Sharpness(bgimg)
		print('sharpeness', args['sharpeness'])
		bgimg = enhancer.enhance(args['sharpeness'])

	return bgimg


class BmabExtScript(scripts.Script):

	def __init__(self) -> None:
		super().__init__()
		self.extra_image = []

	def title(self):
		return 'BMAB Extension'

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		enabled, execute_before_img2img, input_image, contrast, brightness, sharpeness, noise_alpha, blend_enabed, blend_alpha, dino_detect_enabed, dino_prompt, edge_flavor_enabed, edge_low_threadhold, edge_high_threadhold, edge_strength = self._create_ui()

		self.infotext_fields = (
			(enabled, lambda x: gr.Checkbox.update(value='enabled' in x)),
			(execute_before_img2img, lambda x: gr.Checkbox.update(value='execute_before_img2img' in x)),
			(input_image, 'input image'),
			(contrast, 'contrast value'),
			(brightness, 'brightness value'),
			(sharpeness, 'sharpeness value'),
			(noise_alpha, 'noise alpha value'),
			(blend_enabed, lambda x: gr.Checkbox.update(value='blend_enabed' in x)),
			(blend_alpha, 'blend transparency value'),
			(dino_detect_enabed, lambda x: gr.Checkbox.update(value='dino_detect_enabed' in x)),
			(dino_prompt, 'dino prompt value'),
			(edge_flavor_enabed, lambda x: gr.Checkbox.update(value='edge_flavor_enabed' in x)),
			(edge_low_threadhold, 'edge low threshold value'),
			(edge_high_threadhold, 'edge high threshold value'),
			(edge_strength, 'edge strength value'),
		)

		return [
			enabled, execute_before_img2img, input_image, contrast, brightness, sharpeness, noise_alpha,
			blend_enabed, blend_alpha,
			dino_detect_enabed, dino_prompt,
			edge_flavor_enabed, edge_low_threadhold, edge_high_threadhold, edge_strength
		]

	def run(self, p, *args):
		print('run', args)
		super().run(p, *args)

	def before_component(self, component, **kwargs):
		super().before_component(component, **kwargs)

	def before_process(self, p, *args):
		self.extra_image = []
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if isinstance(p, StableDiffusionProcessingImg2Img):
			if a['dino_detect_enabed']:
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

		if isinstance(p, StableDiffusionProcessingImg2Img):
			if p.resize_mode == 2 and len(p.init_images) == 1:
				print('img2img.resize')
				im = p.init_images[0]
				img = resize_image(p.resize_mode, im, p.width, p.height)
				self.extra_image.append(img)
				for idx in range(0, len(p.init_latent)):
					p.init_latent[idx] = image_to_latent(p, img)

			if check_process(a, p):
				if len(p.init_images) == 1:
					img = latent_to_image(p.init_latent, 0)
					print('img2img.process_all')
					img = process_all(a, p, img)
					self.extra_image.append(img)
					for idx in range(0, len(p.init_latent)):
						p.init_latent[idx] = image_to_latent(p, img)
				else:
					for idx in range(0, len(p.init_latent)):
						img = latent_to_image(p.init_latent, idx)
						print('img2img.process_all')
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
			'noise_alpha': args[6],
			'blend_enabed': args[7],
			'blend_alpha': args[8],
			'dino_detect_enabed': args[9],
			'dino_prompt': args[10],
			'edge_flavor_enabed': args[11],
			'edge_low_threadhold': args[12],
			'edge_high_threadhold': args[13],
			'edge_strength': args[14],
		}

	def postprocess_image(self, p, pp, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if not a['execute_before_img2img']:
			pp.image = process_all(a, p, pp.image)
		pp.image = after_process(a, p, pp.image)

	def postprocess(self, p, processed, *args):
		processed.images.extend(self.extra_image)

	def callback_state(self, d):
		print('test')

	def before_hr(self, p, *args):
		a = self.parse_args(args)
		if not a['enabled']:
			return

		if not check_process(a, p):
			return

		print('before_hr', p)
		print('sampler', p.sampler)

		if isinstance(p, StableDiffusionProcessingTxt2Img):
			if isinstance(p.sampler, KDiffusionSampler):

				def sample_img2img(self, s, ar, p, x, noise, conditioning, unconditional_conditioning, steps=None,
								   image_conditioning=None):
					for idx in range(0, len(x)):
						img = latent_to_image(x, idx)
						print('process_all', ar)
						img = process_all(ar, p, img)
						s.extra_image.append(img)
						x[idx] = image_to_latent(p, img)
					return self.temp(p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning)

				p.sampler.temp = p.sampler.sample_img2img
				p.sampler.sample_img2img = partial(sample_img2img.__get__(p.sampler, KDiffusionSampler), self, a)

	def describe(self):
		return 'This stuff is worth it, you can bye me a beer in return.'

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
								contrast = gr.Slider(minimum=0, maximum=2, valu=1, step=0.05, label='Contrast')
							with gr.Row():
								brightness = gr.Slider(minimum=0, maximum=2, value=1, step=0.05, label='Brightness')
							with gr.Row():
								sharpeness = gr.Slider(minimum=-5, maximum=5, value=1, step=0.1, label='Sharpeness')
							with gr.Row():
								noise_alpha = gr.Slider(minimum=0, maximum=1, value=0, step=0.05, label='Noise alpha')
						with gr.Tab('Edge', elem_id='edge_tabs'):
							with gr.Row():
								edge_flavor_enabed = gr.Checkbox(label='Edge enhancement enabled', value=False)
							with gr.Row():
								edge_low_threadhold = gr.Slider(minimum=1, maximum=255, value=50, step=1, label='Edge low threshold')
								edge_high_threadhold = gr.Slider(minimum=1, maximum=255, value=200, step=1, label='Edge high threshold')
							with gr.Row():
								edge_strength = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label='Edge strength')
						with gr.Tab('Imaging', elem_id='imaging_tabs'):
							with gr.Row():
								input_image = gr.Image(source="upload")
							with gr.Row():
								blend_enabed = gr.Checkbox(label='Blend enabled', value=False)
							with gr.Row():
								blend_alpha = gr.Slider(minimum=0, maximum=1, value=1, step=0.05, label='Blend alpha')
							with gr.Row():
								dino_detect_enabed = gr.Checkbox(label='Dino detect enabled', value=False)
							with gr.Row():
								dino_prompt = gr.Textbox(placeholder='1girl:0:0.4:0', visible=True, value='')

				return (enabled, execute_before_img2img, input_image, contrast, brightness, sharpeness, noise_alpha, blend_enabed, blend_alpha,
						dino_detect_enabed, dino_prompt, edge_flavor_enabed, edge_low_threadhold, edge_high_threadhold, edge_strength)


