import cv2
import numpy as np
import math
import random

from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageEnhance

from copy import copy, deepcopy
from pathlib import Path

from functools import partial

from modules import shared
from modules import devices
from modules import images
from modules.processing import process_images
from modules.processing import StableDiffusionProcessingImg2Img
from modules.processing import StableDiffusionProcessingTxt2Img

from sd_bmab import dinosam, constants, util, detailing, controlnet
from sd_bmab.util import debug_print


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
	return args['edge_flavor_enabled'] or args['noise_alpha'] or args['face_detailing_enabled'] or args['hand_detailing_enabled'] or \
		   (args['blend_enabled'] and args['input_image'] is not None and 0 <= args['blend_alpha'] <= 1) or \
		   args['resize_by_person_enabled']


def check_hires_fix_process(args, p):
	return args['edge_flavor_enabled'] or args['noise_alpha'] or args['hand_detailing_before_hiresfix_enabled'] or \
		   args['hand_detailing_before_hiresfix_enabled'] or \
		   (args['blend_enabled'] and args['input_image'] is not None and 0 <= args['blend_alpha'] <= 1) or \
		   args['resize_by_person_enabled']


def process_all(s, p, args, bgimg, caller='before_img2img'):
	if args['resize_by_person_enabled']:
		bgimg = process_resize_by_person(bgimg, s, p, args, caller=caller)

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


def after_process(bgimg, s, p, args):

	if args['noise_alpha_final'] != 0:
		p.extra_generation_params['BMAB noise alpha final'] = args['noise_alpha_final']
		img_noise = generate_noise(bgimg.size[0], bgimg.size[1])
		bgimg = Image.blend(bgimg, img_noise, alpha=args['noise_alpha_final'])

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

	if args['color_saturation'] != 1:
		p.extra_generation_params['BMAB color'] = args['color_saturation']
		enhancer = ImageEnhance.Color(bgimg)
		bgimg = enhancer.enhance(args['color_saturation'])

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


def process_resize_by_person(img, s, p, arg, caller='before_img2img'):
	if shared.state.skipped or shared.state.interrupted:
		return img

	enabled = arg.get('resize_by_person_enabled', False)
	resize_by_person_opt = arg.get('module_config', {}).get('resize_by_person_opt', {})
	mode = resize_by_person_opt.get('mode', constants.resize_mode_default)
	value = resize_by_person_opt.get('scale', 0)

	if not enabled:
		return img
	if 0.79 > value >= 1.0:
		return img
	if caller == 'before_img2img' and mode != 'Intermediate':
		return img

	if mode == 'Intermediate':
		return process_resize_by_person_intermedate(img, s, p, arg)
	if mode == 'ControlNet inpaint+lama':
		return process_resize_by_person_using_controlnet(img, s, p, arg)
	if mode == 'Inpaint':
		return process_resize_by_person_using_inpaint(img, s, p, arg)

	return img


@detailing.timecalc
def process_resize_by_person_intermedate(img, s, p, a):
	resize_by_person_opt = a.get('module_config', {}).get('resize_by_person_opt', {})
	value = resize_by_person_opt.get('scale', 0)

	p.extra_generation_params['BMAB process_resize_by_person'] = value

	debug_print('prepare dino')
	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(img, 'person')

	org_size = img.size
	debug_print('size', org_size)

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
	debug_print('ratio', ratio)
	debug_print('org_size', org_size)

	if ratio > value:
		image_ratio = ratio / value
		if image_ratio < 1.0:
			return img
		debug_print('image resize ratio', image_ratio)
		img = util.resize_image(2, img, int(img.width * image_ratio), int(img.height * image_ratio))
		img = img.resize(org_size, resample=LANCZOS)
		p.extra_generation_params['BMAB process_resize_by_person_ratio'] = '%.3s' % image_ratio

	return img


@detailing.timecalc
def process_resize_by_person_using_controlnet(img, s, p, a):
	resize_by_person_opt = a.get('module_config', {}).get('resize_by_person_opt', {})
	value = resize_by_person_opt.get('scale', 0)
	denoising_strength = resize_by_person_opt.get('denoising_strength', 0.4)
	dilation = resize_by_person_opt.get('dilation', 0.4)

	opt = dict(denoising_strength=denoising_strength)
	i2i_param = build_img2img(p, img, opt)

	img2img = StableDiffusionProcessingImg2Img(**i2i_param)
	img2img.cached_c = [None, None]
	img2img.cached_uc = [None, None]
	img2img.scripts, img2img.script_args = apply_extensions(p, cn_enabled=True)

	if controlnet.resize_by_person_using_controlnet(s, img2img, a, 0, value, dilation):
		processed = process_images(img2img)
		img = processed.images[0]
		img2img.close()
		devices.torch_gc()
	return img


@detailing.timecalc
def process_resize_by_person_using_inpaint(img, s, p, a):
	resize_by_person_opt = a.get('module_config', {}).get('resize_by_person_opt', {})
	value = resize_by_person_opt.get('scale', 0)
	denoising_strength = resize_by_person_opt.get('denoising_strength', 0.4)
	dilation = resize_by_person_opt.get('dilation', 0.4)

	debug_print('prepare dino')
	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(img, 'person')

	org_size = img.size
	debug_print('size', org_size)

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
	debug_print('ratio', ratio)
	debug_print('org_size', org_size)
	if ratio <= value:
		return img
	image_ratio = ratio / value
	if image_ratio < 1.0:
		return img
	debug_print('scale', image_ratio)
	ratio = image_ratio

	org_size = img.size
	dw, dh = org_size

	p.extra_generation_params['BMAB controlnet mode'] = 'inpaint'
	p.extra_generation_params['BMAB resize by person ratio'] = '%.3s' % ratio

	resized_width = int(dw / ratio)
	resized_height = int(dh / ratio)
	resized = img.resize((resized_width, resized_height), resample=LANCZOS)
	p.resize_mode = 2
	input_image = util.resize_image(2, resized, dw, dh)

	offset_x = int((dw - resized_width) / 2)
	offset_y = dh - resized_height

	mask = Image.new('L', (dw, dh), 255)
	dr = ImageDraw.Draw(mask, 'L')
	dr.rectangle((offset_x, offset_y, offset_x + resized_width, offset_y + resized_height), fill=0)
	mask = mask.resize(org_size, resample=LANCZOS)
	mask = util.dilate_mask(mask, dilation)

	opt = dict(mask=mask, denoising_strength=denoising_strength)
	i2i_param = build_img2img(p, input_image, opt)

	img2img = StableDiffusionProcessingImg2Img(**i2i_param)
	img2img.cached_c = [None, None]
	img2img.cached_uc = [None, None]
	img2img.scripts, img2img.script_args = apply_extensions(p, cn_enabled=False)

	processed = process_images(img2img)
	img = processed.images[0]

	img2img.close()

	devices.torch_gc()
	return img


def sam(prompt, input_image):
	boxes, logits, phrases = dinosam.dino_predict(input_image, prompt, 0.35, 0.25)
	mask = dinosam.sam_predict(input_image, boxes)
	return mask


def apply_extensions(p, cn_enabled=False):
	script_runner = copy(p.scripts)
	script_args = deepcopy(p.script_args)
	active_script = ['dynamic_thresholding']

	if cn_enabled:
		active_script.append('controlnet')
		for idx, obj in enumerate(script_args):
			if 'controlnet' in obj.__class__.__name__.lower():
				if hasattr(obj, 'enabled'):
					obj.enabled = False
				if hasattr(obj, 'input_mode'):
					obj.input_mode = getattr(obj.input_mode, 'SIMPLE', 'simple')
			elif isinstance(obj, dict) and 'module' in obj:
				obj['enabled'] = False

	filtered_alwayson = []
	for script_object in script_runner.alwayson_scripts:
		filepath = script_object.filename
		filename = Path(filepath).stem
		if filename in active_script:
			filtered_alwayson.append(script_object)

	script_runner.alwayson_scripts = filtered_alwayson
	return script_runner, script_args


def build_img2img(p, img, options):

	img = img.convert('RGB')

	if 'inpaint_full_res' in options:
		res = options['inpaint_full_res']
		if res == 'Whole picture':
			options['inpaint_full_res'] = 0
		if res == 'Only masked':
			options['inpaint_full_res'] = 1

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
		steps=p.steps,
		cfg_scale=p.cfg_scale,
		width=img.width,
		height=img.height,
		restore_faces=False,
		tiling=p.tiling,
		extra_generation_params=p.extra_generation_params,
		do_not_save_samples=True,
		do_not_save_grid=True,
		override_settings=p.override_settings,
	)
	if options is not None:
		i2i_param.update(options)

	return i2i_param


def process_img2img(p, img, options=None):
	if shared.state.skipped or shared.state.interrupted:
		return img

	i2i_param = build_img2img(p, img, options)

	img2img = StableDiffusionProcessingImg2Img(**i2i_param)
	img2img.cached_c = [None, None]
	img2img.cached_uc = [None, None]
	img2img.scripts, img2img.script_args = apply_extensions(p)

	processed = process_images(img2img)
	img = processed.images[0]

	img2img.close()

	devices.torch_gc()
	return img


def process_txt2img(s, p, a, options: dict):

	t2i_param = dict(
		denoising_strength=0.4,
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
		steps=p.steps,
		cfg_scale=p.cfg_scale,
		width=p.width,
		height=p.height,
		restore_faces=False,
		tiling=p.tiling,
		extra_generation_params=p.extra_generation_params,
		do_not_save_samples=True,
		do_not_save_grid=True,
		override_settings={},
	)
	if options is not None:
		t2i_param.update(options)

	txt2img = StableDiffusionProcessingTxt2Img(**t2i_param)
	txt2img.scripts = None
	txt2img.script_args = None

	processed = process_images(txt2img)
	debug_print('seeds', txt2img.seed)
	debug_print('all seeds', txt2img.all_seeds)
	img = processed.images[0]
	devices.torch_gc()
	return img, txt2img.all_seeds[0]


def masked_image(img, xyxy):
	x1, y1, x2, y2 = xyxy
	check = img.convert('RGBA')
	dd = Image.new('RGBA', img.size, (0, 0, 0, 0))
	dr = ImageDraw.Draw(dd, 'RGBA')
	dr.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 255))
	check = Image.blend(check, dd, alpha=0.5)
	check.convert('RGB').save('check.png')


def process_dino_detect(p, s, a):
	if a['dino_detect_enabled']:
		if p.image_mask is not None:
			s.extra_image.append(p.init_images[0])
			s.extra_image.append(p.image_mask)
			p.image_mask = sam(a['dino_prompt'], p.init_images[0])
			s.extra_image.append(p.image_mask)
			devices.torch_gc()
		if p.image_mask is None and a['input_image'] is not None:
			mask = sam(a['dino_prompt'], p.init_images[0])
			inputimg = a['input_image']
			newpil = Image.new('RGB', p.init_images[0].size)
			newdata = [bdata if mdata == 0 else ndata for mdata, ndata, bdata in zip(mask.getdata(), p.init_images[0].getdata(), inputimg.getdata())]
			newpil.putdata(newdata)
			p.init_images[0] = newpil
			s.extra_image.append(newpil)


def process_img2img_process_all(s, p, a):
	if isinstance(p, StableDiffusionProcessingImg2Img):
		if p.resize_mode == 2 and len(p.init_images) == 1:
			im = p.init_images[0]
			p.extra_generation_params['BMAB resize image'] = '%s %s' % (p.width, p.height)
			img = util.resize_image(p.resize_mode, im, p.width, p.height)
			s.extra_image.append(img)
			for idx in range(0, len(p.init_latent)):
				p.init_latent[idx] = util.image_to_latent(p, img)
				devices.torch_gc()

		if check_process(a, p):
			if len(p.init_images) == 1:
				img = util.latent_to_image(p.init_latent, 0)
				img = process_all(s, p, a, img)
				s.extra_image.append(img)
				for idx in range(0, len(p.init_latent)):
					p.init_latent[idx] = util.image_to_latent(p, img)
					devices.torch_gc()
			else:
				for idx in range(0, len(p.init_latent)):
					img = util.latent_to_image(p.init_latent, idx)
					img = process_all(s, p, a, img)
					s.extra_image.append(img)
					p.init_latent[idx] = util.image_to_latent(p, img)
					devices.torch_gc()


def override_sample(s, p, a):
	if hasattr(p, '__sample'):
		return
	p.__sample = p.sample

	def resize(_p, _s, arg, resize_mode, img, width, height, upscaler_name=None):
		images.resize_image = p.resize_hook
		pidx = _p.iteration * _p.batch_size
		_p.__idx += 1
		arg['current_prompt'] = _p.all_prompts[pidx]
		if arg['face_detailing_before_hiresfix_enabled']:
			img = detailing.process_face_detailing_inner(img, _s, _p, arg)
		if arg['hand_detailing_before_hiresfix_enabled']:
			img = detailing.process_hand_detailing(img, _s, _p, arg)
		# s.extra_image.append(img)
		im = _p.resize_hook(resize_mode, img, width, height, upscaler_name)
		im = process_all(_s, _p, arg, im)
		images.resize_image = partial(resize, p, s, a)
		return im

	def _sample(self, s, a, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
		self.resize_hook = images.resize_image
		p.__idx = 0
		images.resize_image = partial(resize, self, s, a)
		try:
			ret = self.__sample(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts)
		except Exception as e:
			raise e
		finally:
			images.resize_image = self.resize_hook
		return ret

	p.sample = partial(_sample, p, s, a)


def process_upscale_before_detailing(image, s, p, a):
	if not a['upscale_enabled'] or not a['detailing_after_upscale']:
		return image
	return process_upscale_inner(image, s, p, a)


def process_upscale_after_detailing(image, s, p, a):
	if not a['upscale_enabled'] or a['detailing_after_upscale']:
		return image
	return process_upscale_inner(image, s, p, a)


@detailing.timecalc
def process_upscale_inner(image, s, p, args):
	ratio = args['upscale_ratio']
	upscaler = args['upscaler_name']
	debug_print(f'Upscale ratio {ratio} Upscaler {upscaler}')
	p.extra_generation_params['BMAB_upscale_option'] = f'Upscale ratio {ratio} Upscaler {upscaler}'

	if ratio < 1.0 or ratio > 4.0:
		debug_print('upscale out of range')
		return image
	image = image.convert('RGB')
	p.extra_generation_params['BMAB process upscale'] = ratio
	args['max_area'] = image.width * image.height
	args['upscale_limit'] = True

	w = image.width
	h = image.height
	img = images.resize_image(0, image, int(w * ratio), int(h * ratio), upscaler)
	return img.convert('RGB')
