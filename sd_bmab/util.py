import torch
import numpy as np
from PIL import Image

from modules import shared
from modules import devices
from modules import images
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
	x1, y1, x2, y2 = tuple(int(x) for x in box)
	dx = int((x2 - x1) * dil)
	dy = int((y2 - y1) * dil)
	return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def fix_box_size(box):
	x1, y1, x2, y2 = tuple(int(x) for x in box)
	w = x2 - x1
	h = y2 - y1
	w = (w // 8) * 8
	h = (h // 8) * 8
	return x1, y1, x1 + w, y1 + h


def fix_size_by_scale(w, h, scale):
	w = int(((w * scale) // 8) * 8)
	h = int(((h * scale) // 8) * 8)
	return w, h


def fix_box_by_scale(box, scale):
	x1, y1, x2, y2 = tuple(int(x) for x in box)
	w = x2 - x1
	h = y2 - y1
	dx = int(w * scale / 2)
	dy = int(h * scale / 2)
	return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def fix_box_limit(box, size):
	x1, y1, x2, y2 = tuple(int(x) for x in box)
	w = size[0]
	h = size[1]
	if x1 < 0:
		x1 = 0
	if y1 < 0:
		y1 = 0
	if x2 >= w:
		x2 = w-1
	if y2 >= h:
		y2 = h-1
	return x1, y1, x2, y2
