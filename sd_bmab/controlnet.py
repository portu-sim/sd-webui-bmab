from pathlib import Path
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from modules import shared
from modules import processing

from sd_bmab import util, dinosam, process


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


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


def get_openpose_args(image):
	cn_args = {
		'input_image': b64_encoding(image),
		'module': 'openpose',
		'model': shared.opts.bmab_cn_openpose,
		'weight': 1,
		"guidance_start": 0,
		"guidance_end": 1,
		'resize mode': 'Just Resize',
		'allow preview': False,
		'pixel perfect': False,
		'control mode': 'My prompt is more important',
		'processor_res': 512,
		'threshold_a': 64,
		'threshold_b': 64,
	}
	return cn_args


def get_inpaint_lama_args(image, mask):
	cn_args = {
		'input_image': b64_encoding(image),
		'mask': b64_encoding(mask),
		'module': 'inpaint_only+lama',
		'model': shared.opts.bmab_cn_inpaint,
		'weight': 1,
		"guidance_start": 0,
		"guidance_end": 1,
		'resize mode': 'Resize and Fill',
		'allow preview': False,
		'pixel perfect': False,
		'control mode': 'ControlNet is more important',
		'processor_res': 512,
		'threshold_a': 64,
		'threshold_b': 64,
	}
	return cn_args


def get_noise_args(image, weight):
	cn_args = {
		'input_image': b64_encoding(image),
		'model': shared.opts.bmab_cn_lineart,
		'weight': weight,
		"guidance_start": 0.1,
		"guidance_end": 0.9,
		'resize mode': 'Just Resize',
		'allow preview': False,
		'pixel perfect': False,
		'control mode': 'ControlNet is more important',
		'processor_res': 512,
		'threshold_a': 64,
		'threshold_b': 64,
	}
	return cn_args


def get_ratio(img, s, p, args, value):
	p.extra_generation_params['BMAB process_resize_by_person'] = value

	final_ratio = 1
	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(img, 'person')

	largest = (0, None)
	for box in boxes:
		x1, y1, x2, y2 = box
		size = (x2 - x1) * (y2 - y1)
		if size > largest[0]:
			largest = (size, box)

	if largest[0] == 0:
		return final_ratio

	x1, y1, x2, y2 = largest[1]
	ratio = (y2 - y1) / img.height
	print('ratio', ratio)
	dinosam.release()

	if ratio > value:
		image_ratio = ratio / value
		if image_ratio < 1.0:
			return final_ratio
		final_ratio = image_ratio
	return final_ratio


def resize_by_person_using_controlnet(s, p, a, cn_num, value, dilation):
	if not isinstance(p, processing.StableDiffusionProcessingImg2Img):
		return False

	cn_args = get_cn_args(p)

	print('resize_by_person_enabled_inpaint', value)
	img = p.init_images[0]
	s.extra_image.append(img)

	ratio = get_ratio(img, s, p, a, value)
	print('image resize ratio', ratio)
	org_size = img.size
	dw, dh = org_size

	p.extra_generation_params['BMAB controlnet mode'] = 'inpaint'
	p.extra_generation_params['BMAB resize by person ratio'] = '%.3s' % ratio

	resized_width = int(dw / ratio)
	resized_height = int(dh / ratio)
	resized = img.resize((resized_width, resized_height), resample=LANCZOS)
	p.resize_mode = 2
	input_image = util.resize_image(2, resized, dw, dh)
	p.init_images[0] = input_image

	offset_x = int((dw - resized_width) / 2)
	offset_y = dh - resized_height

	mask = Image.new('L', (dw, dh), 255)
	dr = ImageDraw.Draw(mask, 'L')
	dr.rectangle((offset_x, offset_y, offset_x + resized_width, offset_y + resized_height), fill=0)
	mask = mask.resize(org_size, resample=LANCZOS)
	mask = util.dilate_mask(mask, dilation)

	cn_op_arg = get_inpaint_lama_args(input_image, mask)
	idx = cn_args[0] + cn_num
	sc_args = list(p.script_args)
	sc_args[idx] = cn_op_arg
	p.script_args = tuple(sc_args)
	return True


def process_controlnet(s, p, a):
	controlnet_opt = a.get('module_config', {}).get('controlnet', {})

	if not controlnet_opt.get('enabled', False):
		return

	p.extra_generation_params['BMAB_controlnet_option'] = util.dict_to_str(controlnet_opt)
	noise_enabled = controlnet_opt.get('noise', False)
	if not noise_enabled:
		return

	print('Seed', p.seed)
	print('AllSeeds', p.all_seeds)

	cn_args = get_cn_args(p)
	print('ControlNet', cn_args)

	count = 0

	if noise_enabled:
		noise_strength = controlnet_opt.get('noise_strength', 0.4)
		print('noise enabled.', noise_strength)
		p.extra_generation_params['BMAB controlnet mode'] = 'lineart'
		p.extra_generation_params['BMAB noise strength'] = noise_strength

		img = process.generate_noise(p.width, p.height)
		cn_op_arg = get_noise_args(img, noise_strength)
		idx = cn_args[0] + count
		count += 1
		sc_args = list(p.script_args)
		sc_args[idx] = cn_op_arg
		p.script_args = tuple(sc_args)
