from functools import partial
from modules import images
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img

from sd_bmab.base import Context
from sd_bmab.processors.detailer import FaceDetailer

from sd_bmab.processors.basic import EdgeEnhancement, NoiseAlpha, Img2imgMasking
from sd_bmab.processors.preprocess import ResizeIntermidiate
from sd_bmab.processors.preprocess import ResamplePreprocessor


def process_intermediate_step1(context, image):
	all_processors = [
		ResamplePreprocessor(step=1),
		FaceDetailer(step=1),
		ResizeIntermidiate(step=1),
	]

	processed = image.copy()

	for proc in all_processors:
		result = proc.preprocess(context, processed)
		if result is None or not result:
			continue
		ret = proc.process(context, processed)
		proc.postprocess(context, processed)
		processed = ret

	return processed


def process_intermediate_step2(context, image):
	all_processors = [
		EdgeEnhancement(),
		ResizeIntermidiate(),
		Img2imgMasking(),
		NoiseAlpha(),
	]

	processed = image.copy()

	for proc in all_processors:
		result = proc.preprocess(context, processed)
		if result is None or not result:
			continue
		ret = proc.process(context, processed)
		proc.postprocess(context, processed)
		processed = ret

	return processed


def process_img2img(ctx):
	if not isinstance(ctx.sdprocessing, StableDiffusionProcessingImg2Img):
		return

	image = ctx.sdprocessing.init_images[0]
	ctx.sdprocessing.init_images[0] = process_intermediate_step2(ctx, image)


def process_hiresfix(ctx):
	if not isinstance(ctx.sdprocessing, StableDiffusionProcessingTxt2Img):
		return

	if hasattr(ctx.sdprocessing, '__sample'):
		return

	all_processors = [
		FaceDetailer(),
		EdgeEnhancement(),
		ResizeIntermidiate(step=1),
		ResizeIntermidiate(),
		NoiseAlpha()
	]

	if True not in [proc.preprocess(ctx, None) for proc in all_processors]:
		return

	ctx.sdprocessing.__sample = ctx.sdprocessing.sample

	def resize(ctx: Context, resize_mode, img, width, height, upscaler_name=None):
		images.resize_image = ctx.sdprocessing.resize_hook
		pidx = ctx.sdprocessing.iteration * ctx.sdprocessing.batch_size
		ctx.index += 1
		ctx.args['current_prompt'] = ctx.sdprocessing.all_prompts[pidx]
		img = process_intermediate_step1(ctx, img)
		im = ctx.sdprocessing.resize_hook(resize_mode, img, width, height, upscaler_name)
		im = process_intermediate_step2(ctx, im)
		images.resize_image = partial(resize, ctx)
		return im

	def _sample(ctx: Context, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
		ctx.sdprocessing.resize_hook = images.resize_image
		images.resize_image = partial(resize, ctx)
		try:
			ret = ctx.sdprocessing.__sample(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts)
		except Exception as e:
			raise e
		finally:
			images.resize_image = ctx.sdprocessing.resize_hook
		return ret

	ctx.sdprocessing.sample = partial(_sample, ctx)
