


# API

Stable diffusion webui의 API를 이용하여 이미지를 생성하는 경우 아래와 같이 BMAB를 사용하여 API Call을 할 수 있습니다.


```python
import requests
import json
import base64


prompt = '''
1girl
'''
negative_prompt = '(worst quality, low quality:1.4),'

txt2img = {
	'prompt': prompt,
	'negative_prompt': negative_prompt,
	'steps': 20,
	'width': 512,
	'height': 768,
	'cfg_scale': 7,
	'seed': -1,
	'sampler_index': 'DPM++ SDE Karras',
	'script_name': None,
	'alwayson_scripts': {
		'BMAB': {
			'args': [
				{
					'enabled': True,
					'face_detailing_enabled': True,
				}
			]
		}
	}
}

response = requests.post('http://localhost:7860/sdapi/v1/txt2img', data=json.dumps(txt2img))
print(response)
j = response.json()
b64_image = j['images'][0]


with open('test.png', 'wb') as image_file:
	image_file.write(base64.b64decode(b64_image))

```

BAMB의 Argument는 저장된 설정 파일과 동일하며, 
이것을 기반으로 모든 설정을 사용할 수 있습니다. 설정에 없는 경우 기본값을 사용합니다.

아래는 json 형태의 기본 설정 값입니다.

```json
{
  "enabled": false,
  "preprocess_checkpoint": "Use same checkpoint",
  "preprocess_vae": "Use same vae",
  "txt2img_noise_multiplier": 1,
  "txt2img_extra_noise_multiplier": 0,
  "txt2img_filter_hresfix_before_upscale": "None",
  "txt2img_filter_hresfix_after_upscale": "None",
  "resample_enabled": false,
  "module_config": {
    "resample_opt": {
      "save_image": false,
      "hiresfix_enabled": false,
      "checkpoint": "Use same checkpoint",
      "vae": "Use same vae",
      "method": "txt2img-1pass",
      "filter": "None",
      "prompt": "",
      "negative_prompt": "",
      "sampler": "Use same sampler",
      "upscaler": "BMAB fast",
      "steps": 20,
      "cfg_scale": 7,
      "denoising_strength": 0.75,
      "strength": 0.5,
      "begin": 0.1,
      "end": 0.9
    },
    "pretraining_opt": {
      "hiresfix_enabled": false,
      "pretraining_model": "Select Model",
      "prompt": "",
      "negative_prompt": "",
      "sampler": "Use same sampler",
      "steps": 20,
      "cfg_scale": 7,
      "denoising_strength": 0.75,
      "dilation": 4,
      "box_threshold": 0.35
    },
    "resize_intermediate_opt": {
      "resize_by_person": true,
      "method": "stretching",
      "alignment": "bottom",
      "filter": "None",
      "scale": 0.85,
      "denoising_strength": 0.75
    },
    "refiner_opt": {
      "checkpoint": "Use same checkpoint",
      "keep_checkpoint": true,
      "prompt": "",
      "negative_prompt": "",
      "sampler": "Use same sampler",
      "upscaler": "BMAB fast",
      "steps": 20,
      "cfg_scale": 7,
      "denoising_strength": 0.75,
      "scale": 1,
      "width": 0,
      "height": 0
    },
    "person_detailing_opt": {
      "best_quality": false,
      "force_1:1": false,
      "block_overscaled_image": true,
      "auto_upscale": true,
      "scale": 4,
      "dilation": 3,
      "area_ratio": 0.1,
      "limit": 1,
      "background_color": 1,
      "background_blur": 0
    },
    "person_detailing": {
      "denoising_strength": 0.4,
      "cfg_scale": 7
    },
    "face_detailing_opt": {
      "best_quality": false,
      "sort_by": "Score",
      "limit": 1,
      "prompt0": "",
      "negative_prompt0": "",
      "prompt1": "",
      "negative_prompt1": "",
      "prompt2": "",
      "negative_prompt2": "",
      "prompt3": "",
      "negative_prompt3": "",
      "prompt4": "",
      "negative_prompt4": "",
      "override_parameter": false,
      "sampler": "Use same sampler",
      "detection_model": "BMAB Face(Normal)",
      "dilation": 4,
      "box_threshold": 0.35,
      "skip_large_face": false,
      "large_face_pixels": 0.26
    },
    "face_detailing": {
      "width": 512,
      "height": 512,
      "cfg_scale": 7,
      "steps": 20,
      "mask_blur": 4,
      "inpaint_full_res": "Only masked",
      "inpaint_full_res_padding": 32,
      "denoising_strength": 0.4
    },
    "hand_detailing_opt": {
      "block_overscaled_image": true,
      "best_quality": false,
      "detailing_method": "subframe",
      "auto_upscale": true,
      "scale": 4,
      "box_threshold": 0.3,
      "dilation": 0.1,
      "additional_parameter": ""
    },
    "hand_detailing": {
      "prompt": "",
      "negative_prompt": "",
      "denoising_strength": 0.4,
      "cfg_scale": 7,
      "inpaint_full_res": "Only masked",
      "inpaint_full_res_padding": 32
    },
    "controlnet": {
      "enabled": false,
      "with_refiner": false,
      "noise": false,
      "noise_strength": 0.4,
      "noise_begin": 0.1,
      "noise_end": 0.9
    },
    "resize_by_person_opt": {
      "mode": "Inpaint",
      "scale": 0.85,
      "denoising_strength": 0.6,
      "dilation": 30
    }
  },
  "pretraining_enabled": false,
  "edge_flavor_enabled": false,
  "edge_low_threadhold": 50,
  "edge_high_threadhold": 200,
  "edge_strength": 0.5,
  "resize_intermediate_enabled": false,
  "refiner_enabled": false,
  "contrast": 1,
  "brightness": 1,
  "sharpeness": 1,
  "color_saturation": 1,
  "color_temperature": 0,
  "noise_alpha": 0,
  "noise_alpha_final": 0,
  "input_image": null,
  "blend_enabled": false,
  "blend_alpha": 1,
  "detect_enabled": false,
  "masking_prompt": "",
  "person_detailing_enabled": false,
  "face_detailing_enabled": false,
  "face_detailing_before_hiresfix_enabled": false,
  "hand_detailing_enabled": false,
  "resize_by_person_enabled": false,
  "upscale_enabled": false,
  "detailing_after_upscale": true,
  "upscaler_name": "None",
  "upscale_ratio": 1.5,
  "config_file": "test",
  "preset": "None"
}
```

