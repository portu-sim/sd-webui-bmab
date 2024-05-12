import torch
import numpy as np

from ultralytics import YOLO

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

import modules
from modules import devices

from sd_bmab import util
from sd_bmab.base import sam
from sd_bmab.external import load_external_module


def process_iclight(context, image, bg_image, prompt, blending, bg_source, arg1, arg2):
	np_image = np.array(image.convert('RGB')).astype("uint8")

	if bg_image is None:
		mod = load_external_module('iclight', 'iclightnm')
		input_fg, matting = mod.run_rmbg(np_image)
		seed, subseed = context.get_seeds()
		result = mod.process_relight(input_fg, prompt, image.width, image.height, 1, seed, 25,
			'best quality', 'lowres, bad anatomy, bad hands, cropped, worst quality',
			arg1[0], arg1[1], arg1[2], arg1[3], bg_source)
		mod.clean_up()
		context.add_extra_image(image)
		context.add_extra_image(result)
	else:
		mod = load_external_module('iclight', 'iclightbg')
		input_fg, matting = mod.run_rmbg(np_image)
		seed, subseed = context.get_seeds()
		result = mod.process_relight(input_fg, None, prompt, image.width, image.height, 1, seed, 20,
			'best quality', 'lowres, bad anatomy, bad hands, cropped, worst quality',
			arg2[0], arg2[1], arg2[2], bg_source)
		mod.clean_up()
		context.add_extra_image(image)
		context.add_extra_image(bg_image)
		context.add_extra_image(result)
	return result


def process_bmab_relight(context, image, bg_image, prompt, blending, bg_source, arg1):
	mod = load_external_module('iclight', 'iclightbg')
	seed, subseed = context.get_seeds()
	img1 = image.convert('RGBA')
	if bg_image is None:
		print('BG Source', bg_source)
		if bg_source == 'Face' or bg_source == 'Person':
			img2 = generate_detection_gradient(image, bg_source)
			context.add_extra_image(img2)
		else:
			img2 = generate_gradient((32, 32, 32), (224, 224, 224), image.width, image.height, bg_source)
		img2 = img2.convert('RGBA')
	else:
		img2 = bg_image.resize(img1.size, Image.LANCZOS).convert('RGBA')

	blended = Image.blend(img1, img2, alpha=blending)
	np_image = np.array(image.convert('RGB')).astype("uint8")
	input_bg = np.array(blended.convert('RGB')).astype("uint8")
	input_fg, matting = mod.run_rmbg(np_image)
	result = mod.process_relight(input_fg, input_bg, prompt, image.width, image.height, 1, seed, 20,
		'best quality', 'lowres, bad anatomy, bad hands, cropped, worst quality',
		arg1[0], arg1[1], arg1[2], 'Use Background Image')
	mod.clean_up()
	return result


def generate_gradient(
		colour1, colour2, width: int, height: int, d) -> Image:
	"""Generate a vertical gradient."""
	base = Image.new('RGB', (width, height), colour1)
	top = Image.new('RGB', (width, height), colour2)
	mask = Image.new('L', (width, height))
	mask_data = []
	if d == 'Left':
		for y in range(height):
			mask_data.extend([255 - int(255 * (x / width)) for x in range(width)])
	if d == 'Right':
		for y in range(height):
			mask_data.extend([int(255 * (x / width)) for x in range(width)])
	if d == 'Bottom':
		for y in range(height):
			mask_data.extend([int(255 * (y / height))] * width)
	if d == 'Top':
		for y in range(height):
			mask_data.extend([255 - int(255 * (y / height))] * width)
	mask.putdata(mask_data)
	base.paste(top, (0, 0), mask)
	return base


def predict(image: Image, model, confidence):
	yolo = util.load_pretraining_model(model)
	boxes = []
	confs = []
	load = torch.load
	torch.load = modules.safe.unsafe_torch_load
	try:
		model = YOLO(yolo)
		pred = model(image, conf=confidence, device='')
		boxes = pred[0].boxes.xyxy.cpu().numpy()
		boxes = boxes.tolist()
		confs = pred[0].boxes.conf.tolist()
	except:
		pass
	torch.load = load
	devices.torch_gc()

	return boxes, confs


def generate_detection_gradient(image, model):
	mask = Image.new('L', (512, 768), 32)
	dr = ImageDraw.Draw(mask, 'L')

	if model == 'Face':
		boxes, confs = predict(image, 'face_yolov8n.pt', 0.35)
		for box, conf in zip(boxes, confs):
			x1, y1, x2, y2 = tuple(int(x) for x in box)
			dx = int((x2-x1))
			dy = int((y2-y1))
			dr.ellipse((x1 - dx, y1 - dy, x2 + dx, y2 + dy), fill=225)
		blur = ImageFilter.GaussianBlur(10)
	elif model == 'Person':
		boxes, confs = predict(image, 'person_yolov8n-seg.pt', 0.35)
		for box, conf in zip(boxes, confs):
			x1, y1, x2, y2 = tuple(int(x) for x in box)
			m = sam.sam_predict_box(image, (x1, y1, x2, y2))
			mask.paste(m, mask=m)
		blur = ImageFilter.GaussianBlur(30)
	else:
		return mask
	return mask.filter(blur)


def bmab_relight(context, process_type, image, bg_image, prompt, blending, bg_source):
	if process_type == 'intensive':
		if bg_source == 'Face' or bg_source == 'Person':
			bg_source = 'None'
		return process_iclight(context, image, bg_image, prompt, blending, bg_source, (2, 1.0, 0.5, 0.9), (7, 1.0, 0.5))
	elif process_type == 'less intensive':
		if bg_source == 'Face' or bg_source == 'Person':
			bg_source = 'None'
		return process_iclight(context, image, bg_image, prompt, blending, bg_source, (2, 1.0, 0.45, 0.85), (7, 1.0, 0.45))
	elif process_type == 'normal':
		return process_bmab_relight(context, image, bg_image, prompt, blending, bg_source, (7, 1.0, 0.45))
	elif process_type == 'soft':
		return process_bmab_relight(context, image, bg_image, prompt, blending, bg_source, (7, 1.0, 0.4))


