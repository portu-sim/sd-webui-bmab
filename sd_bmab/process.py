import cv2
import numpy as np
import math
import random

from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance

from sd_bmab import util, dinosam

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


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

	newdata = [(0, 0, 0) if mdata == 0 else ndata for mdata, ndata in zip(canny.getdata(), pil.getdata())]
	newbase = Image.new('RGB', pil.size)
	newbase.putdata(newdata)
	return Image.blend(pil, newbase, alpha=strength).convert("RGB")


def check_process(args, p):
	return args['edge_flavor_enabled'] or args['noise_alpha'] or args['face_detailing_enabled'] or \
		   (args['blend_enabled'] and args['input_image'] is not None and 0 <= args['blend_alpha'] <= 1) or \
		   args['resize_by_person_enabled']


def process_all(args, p, bgimg):
	if args['resize_by_person_enabled']:
		bgimg = process_resize_by_person(args, p, bgimg)

	if args['noise_alpha'] != 0:
		p.extra_generation_params['BMAB noise alpha'] = args['noise_alpha']
		img_noise = generate_noise(bgimg.size[0], bgimg.size[1])
		bgimg = Image.blend(bgimg, img_noise, alpha=args['noise_alpha'])

	if args['edge_flavor_enabled']:
		p.extra_generation_params['BMAB edge flavor low threadhold'] = args['edge_low_threadhold']
		p.extra_generation_params['BMAB edge flavor high threadhold'] = args['edge_high_threadhold']
		p.extra_generation_params['BMAB edge flavor strength'] = args['edge_strength']
		bgimg = edge_flavor(bgimg, args['edge_low_threadhold'], args['edge_high_threadhold'], args['edge_strength'])

	if args['blend_enabled'] and args['input_image'] is not None and 0 <= args['blend_alpha'] <= 1:
		p.extra_generation_params['BMAB blend alpha'] = args['blend_alpha']
		blend = Image.fromarray(args['input_image'], mode='RGB')
		img = Image.new(mode='RGB', size=bgimg.size)
		img.paste(bgimg, (0, 0))
		img.paste(blend)
		bgimg = Image.blend(bgimg, img, alpha=args['blend_alpha'])

	return bgimg


def calc_color_temperature(temp):
	white = (255.0, 254.11008387561782, 250.0419083427406)

	temperature = temp / 100

	if temperature <= 66:
		red = 255.0
	else:
		red = float(temperature - 60)
		red = 329.698727446 * math.pow(red, -0.1332047592)
		if red < 0:
			red = 0
		if red > 255:
			red = 255

	if temperature <= 66:
		green = temperature
		green = 99.4708025861 * math.log(green) - 161.1195681661
	else:
		green = float(temperature - 60)
		green = 288.1221695283 * math.pow(green, -0.0755148492)
	if green < 0:
		green = 0
	if green > 255:
		green = 255

	if temperature >= 66:
		blue = 255.0
	else:
		if temperature <= 19:
			blue = 0.0
		else:
			blue = float(temperature - 10)
			blue = 138.5177312231 * math.log(blue) - 305.0447927307
			if blue < 0:
				blue = 0
			if blue > 255:
				blue = 255

	return red / white[0], green / white[1], blue / white[2]


def after_process(args, p, bgimg):
	if args['contrast'] != 1:
		p.extra_generation_params['BMAB contrast'] = args['contrast']
		enhancer = ImageEnhance.Contrast(bgimg)
		bgimg = enhancer.enhance(args['contrast'])

	if args['brightness'] != 1:
		p.extra_generation_params['BMAB brightness'] = args['brightness']
		enhancer = ImageEnhance.Brightness(bgimg)
		bgimg = enhancer.enhance(args['brightness'])

	if args['sharpeness'] != 1:
		p.extra_generation_params['BMAB sharpeness'] = args['sharpeness']
		enhancer = ImageEnhance.Sharpness(bgimg)
		bgimg = enhancer.enhance(args['sharpeness'])

	if args['color_temperature'] != 0:
		p.extra_generation_params['BMAB color temperature'] = args['color_temperature']
		temp = calc_color_temperature(6500 + args['color_temperature'])
		az = []
		for d in bgimg.getdata():
			az.append((int(d[0] * temp[0]), int(d[1] * temp[1]), int(d[2] * temp[2])))
		bgimg = Image.new('RGB', bgimg.size)
		bgimg.putdata(az)

	return bgimg


def process_prompt(prompt):
	lines = prompt.split('\n')

	read_line = 0
	base_prompt = ''
	for line in lines:
		if line.startswith('#random'):
			candidates = lines[read_line + 1:]
			base_prompt += random.choice(candidates) + '\n'
			return base_prompt
		base_prompt += line + '\n'
		read_line += 1
	return base_prompt


def process_resize_by_person(arg, p, img):
	print('prepare dino')

	enabled = arg.get('resize_by_person_enabled', False)
	if not enabled:
		return img

	value = arg.get('resize_by_person', 0)
	if 0.79 > value >= 1.0:
		return img

	p.extra_generation_params['BMAB process_resize_by_person'] = value

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(img, 'person')

	org_size = img.size
	print('size', org_size)

	largest = (0, None)
	for box in boxes:
		x1, y1, x2, y2 = box
		size = (x2 - x1) * (y2 - y1)
		if size > largest[0]:
			largest = (size, box)

	if largest[0] == 0:
		return img

	x1, y1, x2, y2 = largest[1]
	ratio = (y2 - y1) / img.height
	print('ratio', ratio)
	print('org_size', org_size)

	if ratio > value:
		image_ratio = ratio / value
		if image_ratio < 1.0:
			return img
		print('image resize ratio', image_ratio)
		img = util.resize_image(2, img, int(img.width * image_ratio), int(img.height * image_ratio))
		img = img.resize(org_size, resample=LANCZOS)
		p.extra_generation_params['BMAB process_resize_by_person_ratio'] = '%.3s' % image_ratio

	return img
