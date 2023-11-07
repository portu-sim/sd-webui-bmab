from sd_bmab.sd_override.samper import override_samplers
from sd_bmab.sd_override.img2img import StableDiffusionProcessingImg2ImgOv
from sd_bmab.sd_override.txt2img import StableDiffusionProcessingTxt2ImgOv

from modules import processing


def override_sd_webui():
	override_samplers()
	processing.StableDiffusionProcessingTxt2Img = StableDiffusionProcessingTxt2ImgOv
	processing.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2ImgOv
