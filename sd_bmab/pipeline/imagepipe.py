import traceback

from modules import shared

from sd_bmab.processors.upscaler import AfterProcessUpscaler, BeforeProcessUpscaler
from sd_bmab.processors.resize import InpaintResize, InpaintLamaResize
from sd_bmab.processors.detailer import FaceDetailer, PersonDetailer, HandDetailer
from sd_bmab.processors.utils import BeforeProcessFileSaver, AfterProcessFileSaver
from sd_bmab.processors.utils import ApplyModel, RollbackModel, CheckPointChanger, CheckPointRestore
from sd_bmab.processors.basic import FinalProcessorBasic, EdgeEnhancement, NoiseAlpha, BlendImage
from sd_bmab.processors.controlnet import LineartNoise
from sd_bmab.processors.preprocess import RefinerPreprocessor, PretrainingDetailer, ResizeIntermidiate
from sd_bmab.processors.preprocess import ResamplePreprocessor
from sd_bmab.internalpipeline import Preprocess
from sd_bmab.util import debug_print


def is_controlnet_required(context):
	pipeline_modules = [
		ResamplePreprocessor(),
		InpaintLamaResize(),
	]
	for mod in pipeline_modules:
		if mod.use_controlnet(context):
			return True
	return False


def process(context, image):
	pipeline_modules = [
		BeforeProcessFileSaver(),
		CheckPointChanger(),
		ResamplePreprocessor(),
		PretrainingDetailer(),
		Preprocess(),
		RefinerPreprocessor(),
		InpaintResize(),
		InpaintLamaResize(),
		BeforeProcessUpscaler(),
		ApplyModel(),
		PersonDetailer(),
		FaceDetailer(),
		HandDetailer(),
		RollbackModel(),
		CheckPointRestore(),
		AfterProcessUpscaler(),
		FinalProcessorBasic(),
		BlendImage(),
		AfterProcessFileSaver()
	]
	
	processed = image.copy()

	try:
		for proc in pipeline_modules:
			try:
				if shared.state.interrupted or shared.state.skipped:
					return

				result = proc.preprocess(context, processed)
				if result is None or not result:
					continue
				ret = proc.process(context, processed)
				proc.postprocess(context, processed)
				processed = ret
			except:
				traceback.print_exc()
	finally:
		debug_print('Restore Checkpoint at final')
		RollbackModel().process(context, processed)
		CheckPointRestore().process(context, processed)

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
	pass
