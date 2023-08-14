import json
import os

import numpy as np
import torch
from PIL import Image

from modules import devices
from modules import images
from modules import shared
from modules.processing import process_images
from modules.sd_samplers import sample_to_image
from sd_bmab import dinosam, sdprocessing


def get_config(prompt):
	config_file = None
	newprompt = []
	for line in prompt.split('\n'):
		if line.startswith('##'):
			config_file = line[2:]
			continue
		newprompt.append(line)
	if config_file is None:
		return prompt, {}

	cfg_dir = os.path.join(os.path.dirname(__file__), "../config")
	json_file = os.path.join(cfg_dir, f'{config_file}.json')
	if not os.path.isfile(json_file):
		return '\n'.join(newprompt), {}
	with open(json_file) as f:
		config = json.load(f)
	print('Loading config', json.dumps(config, indent=2))
	return '\n'.join(newprompt), config


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


def tensor_to_image(xx):
	x_sample = 255. * np.moveaxis(xx.cpu().numpy(), 0, 2)
	x_sample = x_sample.astype(np.uint8)
	return Image.fromarray(x_sample)


def image_to_tensor(xx):
	image = np.array(xx).astype(np.float32) / 255
	image = np.moveaxis(image, 2, 0)
	image = torch.from_numpy(image)
	return image


def resize_image(resize_mode, im, width, height, upscaler_name=None):
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


def sam(prompt, input_image):
	boxes, logits, phrases = dinosam.dino_predict(input_image, prompt, 0.35, 0.25)
	mask = dinosam.sam_predict(input_image, boxes)
	return mask


def process_img2img(p, img, options=None):
	if shared.state.skipped or shared.state.interrupted:
		return img

	i2i_param = dict(
		init_images=[img],
		resize_mode=0,
		denoising_strength=0.4,
		mask=None,
		mask_blur=4,
		inpainting_fill=1,
		inpaint_full_res=True,
		inpaint_full_res_padding=32,
		inpainting_mask_invert=0,
		initial_noise_multiplier=1.0,
		sd_model=p.sd_model,
		outpath_samples=p.outpath_samples,
		outpath_grids=p.outpath_grids,
		prompt=p.prompt,
		negative_prompt=p.negative_prompt,
		styles=p.styles,
		seed=p.seed,
		subseed=p.subseed,
		subseed_strength=p.subseed_strength,
		seed_resize_from_h=p.seed_resize_from_h,
		seed_resize_from_w=p.seed_resize_from_w,
		sampler_name=p.sampler_name,
		batch_size=1,
		n_iter=1,
		steps=20,  # p.steps,
		cfg_scale=7,
		width=img.width,
		height=img.height,
		restore_faces=False,
		tiling=p.tiling,
		extra_generation_params=p.extra_generation_params,
		do_not_save_samples=True,
		do_not_save_grid=True,
		override_settings={},
	)
	if options is not None:
		i2i_param.update(options)

	img2img = sdprocessing.StableDiffusionProcessingImg2ImgOv(**i2i_param)
	img2img.scripts = None
	img2img.script_args = None
	img2img.block_tqdm = True
	shared.state.job_count += 1

	processed = process_images(img2img)

	return processed.images[0]
