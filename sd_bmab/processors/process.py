from functools import partial
from modules import images
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img

from sd_bmab.base import dino, sam, Context
from sd_bmab.processors.upscaler import AfterProcessUpscaler, BeforeProcessUpscaler
from sd_bmab.processors.resize import InpaintResize, InpaintLamaResize, IntermidiateResize
from sd_bmab.processors.detailer import FaceDetailer, PersonDetailer, HandDetailer
from sd_bmab.processors.utils import BeforeProcessFileSaver, AfterProcessFileSaver
from sd_bmab.processors.basic import FinalProcessorBasic, EdgeEnhancement, NoiseAlpha, Img2imgMasking, BlendImage
from sd_bmab.processors.controlnet import LineartNoise


def process(context, image):
	all_processors = [
		BeforeProcessFileSaver(),
		InpaintResize(),
		InpaintLamaResize(),
		BeforeProcessUpscaler(),
		PersonDetailer(),
		FaceDetailer(),
		HandDetailer(),
		AfterProcessUpscaler(),
		FinalProcessorBasic(),
		BlendImage(),
		AfterProcessFileSaver()
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


def process_intermediate(context ,image):
	all_processors = [
		FaceDetailer(),
		EdgeEnhancement(),
		IntermidiateResize(),
		NoiseAlpha()
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


def process_intermediate_step1(context, image):
	all_processors = [
		FaceDetailer(),
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
		IntermidiateResize(),
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


def process_hiresfix(ctx):
	if not isinstance(ctx.sdprocessing, StableDiffusionProcessingTxt2Img):
		return

	if hasattr(ctx.sdprocessing, '__sample'):
		return

	all_processors = [
		FaceDetailer(),
		EdgeEnhancement(),
		IntermidiateResize(),
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


def process_img2img(ctx):
	if not isinstance(ctx.sdprocessing, StableDiffusionProcessingImg2Img):
		return

	image = ctx.sdprocessing.init_images[0]
	ctx.sdprocessing.init_images[0] = process_intermediate_step2(ctx, image)


def process_controlnet(context):
	all_processors = [
		LineartNoise(),
	]

	for proc in all_processors:
		result = proc.preprocess(context, None)
		if result is None or not result:
			continue
		proc.process(context, None)
		proc.postprocess(context, None)


def release():
	dino.release()
	sam.release()
