from sd_bmab.base import dino
from sd_bmab.processors.upscaler import AfterProcessUpscaler, BeforeProcessUpscaler
from sd_bmab.processors.resize import InpaintResize, InpaintLamaResize
from sd_bmab.processors.detailer import FaceDetailer, PersonDetailer, HandDetailer
from sd_bmab.processors.utils import BeforeProcessFileSaver, AfterProcessFileSaver
from sd_bmab.processors.utils import ApplyModel, RollbackModel
from sd_bmab.processors.basic import FinalProcessorBasic, EdgeEnhancement, NoiseAlpha, BlendImage
from sd_bmab.processors.controlnet import LineartNoise
from sd_bmab.processors.preprocess import Refiner, RefinerRollbackModel, PretrainingDetailer, ResizeIntermidiate, Preprocess


def process(context, image):
	all_processors = [
		BeforeProcessFileSaver(),
		PretrainingDetailer(),
		Preprocess(),
		Refiner(),
		InpaintResize(),
		InpaintLamaResize(),
		BeforeProcessUpscaler(),
		ApplyModel(),
		PersonDetailer(),
		FaceDetailer(),
		HandDetailer(),
		RollbackModel(),
		RefinerRollbackModel(),
		AfterProcessUpscaler(),
		FinalProcessorBasic(),
		BlendImage(),
		AfterProcessFileSaver()
	]

	processed = image.copy()

	for proc in all_processors:
		try:
			result = proc.preprocess(context, processed)
			if result is None or not result:
				continue
			ret = proc.process(context, processed)
			proc.postprocess(context, processed)
			processed = ret
		except:
			raise
		finally:
			RollbackModel().process(context, processed)

	return processed


def process_intermediate(context, image):
	all_processors = [
		FaceDetailer(),
		EdgeEnhancement(),
		ResizeIntermidiate(),
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
