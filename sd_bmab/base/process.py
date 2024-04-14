from PIL import Image
from PIL import ImageDraw

from copy import copy, deepcopy
from pathlib import Path

from modules import shared, sd_models
from modules.shared import opts, state, sd_model
from modules import devices
from modules.processing import process_images, StableDiffusionProcessingImg2Img

from sd_bmab import util
from sd_bmab.base.common import StopGeneration
from sd_bmab.base.context import Context
from sd_bmab.sd_override import StableDiffusionProcessingTxt2ImgOv, StableDiffusionProcessingImg2ImgOv


def apply_extensions(p, cn_enabled=False):
	script_runner = copy(p.scripts)
	script_args = deepcopy(p.script_args)
	active_script = ['dynamic_thresholding', 'wildcards']

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
		#override_settings=p.override_settings,
		override_settings={
			'sd_model_checkpoint': shared.opts.data['sd_model_checkpoint']
		},
	)

	if hasattr(p, 'scheduler'):
		i2i_param['scheduler'] = p.scheduler,
	else:
		del options['scheduler']

	if options is not None:
		i2i_param.update(options)

	return i2i_param


def process_img2img(p, img, options=None):
	if shared.state.skipped or shared.state.interrupted:
		return img

	i2i_param = build_img2img(p, img, options)

	img2img = StableDiffusionProcessingImg2ImgOv(**i2i_param)
	#img2img = StableDiffusionProcessingImg2Img(**i2i_param)
	img2img.cached_c = [None, None]
	img2img.cached_uc = [None, None]
	img2img.scripts, img2img.script_args = apply_extensions(p)

	with StopGeneration():
		processed = process_images(img2img)
	img = processed.images[0]

	img2img.close()

	devices.torch_gc()
	return img


def process_img2img_with_controlnet(context: Context, image, options, controlnet):
	i2i_param = build_img2img(context.sdprocessing, image, options)

	img2img = StableDiffusionProcessingImg2ImgOv(**i2i_param)
	#img2img = StableDiffusionProcessingImg2Img(**i2i_param)
	img2img.cached_c = [None, None]
	img2img.cached_uc = [None, None]
	img2img.scripts, img2img.script_args = apply_extensions(context.sdprocessing, cn_enabled=True)

	cn_args = util.get_cn_args(img2img)
	idx = cn_args[0]
	sc_args = list(img2img.script_args)
	sc_args[idx] = controlnet
	img2img.script_args = sc_args

	processed = process_images(img2img)
	image = processed.images[0]
	img2img.close()
	devices.torch_gc()

	return image


def process_txt2img(p, options=None, controlnet=None):
	t2i_param = dict(
		denoising_strength=0.4,
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
		#override_settings=p.override_settings,
		override_settings={
			'sd_model_checkpoint': shared.opts.data['sd_model_checkpoint']
		},
	)

	if hasattr(p, 'scheduler'):
		t2i_param['scheduler'] = p.scheduler,
	else:
		del options['scheduler']

	if options is not None:
		t2i_param.update(options)

	txt2img = StableDiffusionProcessingTxt2ImgOv(**t2i_param)
	txt2img.cached_c = [None, None]
	txt2img.cached_uc = [None, None]

	if controlnet is None:
		txt2img.scripts, txt2img.script_args = apply_extensions(p, False)
	else:
		txt2img.scripts, txt2img.script_args = apply_extensions(p, True)
		cn_args = util.get_cn_args(txt2img)
		idx = cn_args[0]
		sc_args = list(txt2img.script_args)
		sc_args[idx] = controlnet
		txt2img.script_args = sc_args

	with StopGeneration():
		processed = process_images(txt2img)
	img = processed.images[0]
	devices.torch_gc()
	return img


def masked_image(img, xyxy):
	x1, y1, x2, y2 = xyxy
	check = img.convert('RGBA')
	dd = Image.new('RGBA', img.size, (0, 0, 0, 0))
	dr = ImageDraw.Draw(dd, 'RGBA')
	dr.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 255))
	check = Image.blend(check, dd, alpha=0.5)
	check.convert('RGB').save('check.png')
