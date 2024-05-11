import traceback
from functools import partial
from modules import images
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules import shared

from sd_bmab.util import debug_print
from sd_bmab.base import Context
from sd_bmab.processors.detailer import FaceDetailer, FaceDetailerBeforeUpsacle

from sd_bmab.processors.basic import EdgeEnhancement, NoiseAlpha, Img2imgMasking, ICLightBeforeUpsacle
from sd_bmab.processors.preprocess import ResizeIntermidiateBeforeUpscale
from sd_bmab.processors.preprocess import ResamplePreprocessorBeforeUpscale, ResizeIntermidiateAfterUpsacle
from sd_bmab.processors.preprocess import PretrainingDetailerBeforeUpscale


def is_controlnet_required(context):
	pipeline_modules = [
		ResamplePreprocessorBeforeUpscale(),
		ResizeIntermidiateBeforeUpscale()
	]
	for mod in pipeline_modules:
		if mod.use_controlnet(context):
			return True
	return False


def process_intermediate_before_upscale(context, image):
	pipeline_before_upscale = [
		ResamplePreprocessorBeforeUpscale(),
		PretrainingDetailerBeforeUpscale(),
		FaceDetailerBeforeUpsacle(),
		ICLightBeforeUpsacle(),
		ResizeIntermidiateBeforeUpscale(),
	]
	
	processed = image.copy()
	for proc in pipeline_before_upscale:
		try:
			result = proc.preprocess(context, processed)
			if result is None or not result:
				continue
			if shared.state.interrupted or shared.state.skipped:
				break
			ret = proc.process(context, processed)
			if shared.state.interrupted or shared.state.skipped:
				break
			proc.postprocess(context, processed)
			processed = ret
		except Exception:
			traceback.print_exc()
	return processed


def process_intermediate_after_upscale(context, image):
	pipeline_before_upscale = [
		EdgeEnhancement(),
		ResizeIntermidiateAfterUpsacle(),
		Img2imgMasking(),
		NoiseAlpha(),
	]
	
	processed = image.copy()
	for proc in pipeline_before_upscale:
		result = proc.preprocess(context, processed)
		if result is None or not result:
			continue
		if shared.state.interrupted or shared.state.skipped:
			break
		ret = proc.process(context, processed)
		proc.postprocess(context, processed)
		processed = ret
	return processed


def process_img2img(ctx):
	if not ctx.is_img2img():
		return

	image = ctx.sdprocessing.init_images[0]
	debug_print('process img2img ', image.size)
	image = process_intermediate_before_upscale(ctx, image)
	image = process_intermediate_after_upscale(ctx, image)
	debug_print('process img2img ', image.size)
	ctx.sdprocessing.init_images[0] = image


'''
def process_hiresfix(ctx):
	if not isinstance(ctx.sdprocessing, StableDiffusionProcessingTxt2Img):
		return

	if hasattr(ctx.sdprocessing, '__sample'):
		return

	all_processors = [
		FaceDetailer(),
		EdgeEnhancement(),
		ResizeIntermidiateBeforeUpscale(),
		ResizeIntermidiateAfterUpsacle(),
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
'''
