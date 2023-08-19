import os
import json
import torch
import numpy as np
from PIL import Image

from modules import shared
from modules import devices
from modules import images
from modules.sd_samplers import sample_to_image


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
		print(f'Not found configuration file {config_file}.json')
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
			res.paste(im.resize((dw, height), box=(im.width, 0, im.width, height)), box=(im.width + dw, 0))

		return res

	return images.resize_image(resize_mode, im, width, height, upscaler_name)


def box_dilation(box, dil):
	x1, y1, x2, y2 = box
	dx = int((x2 - x1) * dil)
	dy = int((y2 - y1) * dil)
	return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def fix_box_size(box):
	x1, y1, x2, y2 = box
	w = x2 - x1
	h = y2 - y1
	w = (w // 8) * 8
	h = (h // 8) * 8
	return x1, y1, x1 + w, y1 + h


def fix_size_by_scale(w, h, scale):
	w = ((w * scale) // 8) * 8
	h = ((h * scale) // 8) * 8
	return w, h
