import os
import cv2
import torch
import numpy as np
from pathlib import Path
import glob
from basicsr.utils.download_util import load_file_from_url

from PIL import Image
import modules
from modules import shared
from modules import devices
from modules import images
from modules.sd_samplers import sample_to_image
from modules.paths import models_path

from ultralytics import YOLO


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


def debug_print(*args):
	if shared.opts.bmab_debug_print:
		print(*args)


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


def fix_sqare_box(box):
	x1, y1, x2, y2 = tuple(int(x) for x in box)
	w = int((x2 - x1) / 2)
	h = int((y2 - y1) / 2)
	x = x1 + w
	y = y1 + h
	l = max(w, h)
	x1 = x - l
	x2 = x + l
	y1 = y - l
	y2 = y + l
	return x1, y1, x2, y2


def change_vae(name='auto'):
	modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file=modules.sd_vae.vae_dict[name])


def get_seeds(s, p, a):
	return p.all_seeds[s.index], p.all_subseeds[s.index]


def ultralytics_predict(image, confidence):
	bmab_model_path = os.path.join(models_path, "bmab")
	yolo = f'{bmab_model_path}/face_yolov8n.pt'
	boxes = []
	load = torch.load
	torch.load = modules.safe.unsafe_torch_load
	try:
		model = YOLO(yolo)
		pred = model(image, conf=confidence, device='')
		boxes = pred[0].boxes.xyxy.cpu().numpy()
		boxes = boxes.tolist()
	except:
		pass
	torch.load = load
	return boxes


def dict_to_str(d):
	return ','.join([f'{k}={v}' for k, v in d.items()])


def dilate_mask(mask, dilation):
	if dilation < 4:
		return mask
	arr = np.array(mask)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
	arr = cv2.dilate(arr, kernel, iterations=1)
	return Image.fromarray(arr)


def erode_mask(mask, erosion):
	if erosion < 4:
		return mask
	arr = np.array(mask)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion, erosion))
	arr = cv2.erode(arr, kernel, iterations=1)
	return Image.fromarray(arr)


def get_cn_args(p):
	for script_object in p.scripts.alwayson_scripts:
		filename = Path(script_object.filename).stem
		if filename == 'controlnet':
			return (script_object.args_from, script_object.args_to)
	return None


def b64_encoding(image):
	from io import BytesIO
	import base64

	buffered = BytesIO()
	image.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode("utf-8")


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


def lazy_loader(filename):
	bmab_model_path = os.path.join(models_path, "bmab")
	files = glob.glob(bmab_model_path)
	
	targets = {
		'sam_vit_b_01ec64.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
		'sam_vit_l_0b3195.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
		'sam_vit_h_4b8939.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
		'groundingdino_swint_ogc.pth': 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth',
		'GroundingDINO_SwinT_OGC.py': 'https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py',
		'face_yolov8n.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt',
		'face_yolov8n_v2.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt',
		'face_yolov8m.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt',
		'face_yolov8s.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt',
		'hand_yolov8n.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt',
		'hand_yolov8s.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt',
		'person_yolov8m-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt',
		'person_yolov8n-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt',
		'person_yolov8s-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt',
		'sam_hq_vit_b.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth',
		'sam_hq_vit_h.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth',
		'sam_hq_vit_l.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth',
		'sam_hq_vit_tiny.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth',
	}

	if filename in targets and filename not in files:
		load_file_from_url(targets[filename], bmab_model_path)
	return os.path.join(bmab_model_path, filename)
