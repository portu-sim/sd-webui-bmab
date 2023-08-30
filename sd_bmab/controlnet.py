from pathlib import Path
from PIL import Image

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
		'model': 'control_v11p_sd15_openpose_fp16 [73c2b67d]',
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


def get_noise_args(image, weight):
	cn_args = {
		'input_image': b64_encoding(image),
		'model': 'control_v11p_sd15_lineart [43d4be0d]',
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


def process_resize_by_person(img, s, p, args):
	controlnet_opt = args.get('module_config', {}).get('controlnet', {})
	value = controlnet_opt.get('resize_by_person', 0.5)
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

	dinosam.release()

	return img


def process_controlnet(s, p, a):
	controlnet_opt = a.get('module_config', {}).get('controlnet', {})

	if not controlnet_opt.get('enabled', False):
		return

	p.extra_generation_params['BMAB_controlnet_option'] = util.dict_to_str(controlnet_opt)

	resize_by_person_enabled = controlnet_opt.get('resize_by_person_enabled', False)
	noise_enabled = controlnet_opt.get('noise', False)
	if not resize_by_person_enabled and not noise_enabled:
		return

	print('Seed', p.seed)
	print('AllSeeds', p.all_seeds)

	cn_args = get_cn_args(p)
	print('ControlNet', cn_args)

	count = 0

	if resize_by_person_enabled:
		img, seed = process.process_txt2img(s, p, a, {})
		s.extra_image.append(img)
		img = process_resize_by_person(img, s, p, a)
		# img = util.resize_image(2, img, int(img.width * 1.2), int(img.height * 1.2))
		p.seed = seed
		cn_op_arg = get_openpose_args(img)
		idx = cn_args[0] + count
		count += 1
		sc_args = list(p.script_args)
		sc_args[idx] = cn_op_arg
		p.script_args = tuple(sc_args)

	if noise_enabled:
		noise_strength = controlnet_opt.get('noise_strength', 0.4)
		print('noise enabled.', noise_strength)

		img = process.generate_noise(p.width, p.height)
		cn_op_arg = get_noise_args(img, noise_strength)
		idx = cn_args[0] + count
		count += 1
		sc_args = list(p.script_args)
		sc_args[idx] = cn_op_arg
		p.script_args = tuple(sc_args)
