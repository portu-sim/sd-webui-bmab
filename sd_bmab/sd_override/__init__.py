from sd_bmab.sd_override.samper import override_samplers
from sd_bmab.sd_override.img2img import StableDiffusionProcessingImg2ImgOv
from sd_bmab.sd_override.txt2img import StableDiffusionProcessingTxt2ImgOv
from modules import processing


#from modules import script_callbacks
#from . import script_callbacks_extranoise

# Merge callback_map from script_callbacks_extranoise into script_callbacks
#if 'callbacks_extra_noise' not in script_callbacks.callback_map:
    #script_callbacks.callback_map['callbacks_extra_noise'] = []

# Add callbacks to the 'callbacks_extra_noise' section
#def on_extra_noise(callback):
    #script_callbacks.add_callback(script_callbacks.callback_map['callbacks_extra_noise'], callback)
	
# Now 'on_extra_noise' can be used to add callbacks for 'callbacks_extra_noise'


processing.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2ImgOv
processing.StableDiffusionProcessingTxt2Img = StableDiffusionProcessingTxt2ImgOv


def override_sd_webui():
	# override_samplers()
	pass
