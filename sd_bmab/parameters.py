import os
import json
from sd_bmab import constants


class Parameters(object):
	def __init__(self) -> None:
		super().__init__()

		self.params = [
			('enabled', False),
			('contrast', 1),
			('brightness', 1),
			('sharpeness', 1),
			('color_temperature', 0),
			('noise_alpha', 0),
			('noise_alpha_final', 0),
			('edge_flavor_enabled', False),
			('edge_low_threadhold', 50),
			('edge_high_threadhold', 200),
			('edge_strength', 0.5),
			('input_image', None),
			('blend_enabled', False),
			('blend_alpha', 1),
			('dino_detect_enabled', False),
			('dino_prompt', ''),
			('person_detailing_enabled', False),
			('module_config.person_detailing_opt.block_overscaled_image', True),
			('module_config.person_detailing_opt.auto_upscale', True),
			('module_config.person_detailing_opt.scale', 4),
			('module_config.person_detailing_opt.dilation', 2),
			('module_config.person_detailing_opt.area_ratio', 0.1),
			('module_config.person_detailing_opt.limit', 1),
			('module_config.person_detailing.denoising_strength', 0.4),
			('module_config.person_detailing.cfg_scale', 7),
			('face_detailing_enabled', False),
			('face_detailing_before_hiresfix_enabled', False),
			('module_config.face_detailing_opt.sort_by', 'Score'),
			('module_config.face_detailing_opt.limit', 1),
			('module_config.face_detailing_opt.prompt0', ''),
			('module_config.face_detailing_opt.negative_prompt0', ''),
			('module_config.face_detailing_opt.prompt1', ''),
			('module_config.face_detailing_opt.negative_prompt1', ''),
			('module_config.face_detailing_opt.prompt2', ''),
			('module_config.face_detailing_opt.negative_prompt2', ''),
			('module_config.face_detailing_opt.prompt3', ''),
			('module_config.face_detailing_opt.negative_prompt3', ''),
			('module_config.face_detailing_opt.prompt4', ''),
			('module_config.face_detailing_opt.negative_prompt4', ''),
			('module_config.face_detailing_opt.override_parameter', False),
			('module_config.face_detailing.width', 512),
			('module_config.face_detailing.height', 512),
			('module_config.face_detailing.cfg_scale', 7),
			('module_config.face_detailing.steps', 20),
			('module_config.face_detailing.mask_blur', 4),
			('module_config.face_detailing_opt.sampler', constants.sampler_default),
			('module_config.face_detailing.inpaint_full_res', 'Only masked'),
			('module_config.face_detailing.inpaint_full_res_padding', 32),
			('module_config.face_detailing.denoising_strength', 0.4),
			('module_config.face_detailing_opt.dilation', 4),
			('module_config.face_detailing_opt.box_threshold', 0.3),
			('hand_detailing_enabled', False),
			('module_config.hand_detailing_opt.block_overscaled_image', True),
			('module_config.hand_detailing_opt.detailing_method', 'subframe'),
			('module_config.hand_detailing.prompt', ''),
			('module_config.hand_detailing.negative_prompt', ''),
			('module_config.hand_detailing.denoising_strength', 0.4),
			('module_config.hand_detailing.cfg_scale', 7),
			('module_config.hand_detailing_opt.auto_upscale', True),
			('module_config.hand_detailing_opt.scale', 2),
			('module_config.hand_detailing_opt.box_threshold', 0.3),
			('module_config.hand_detailing_opt.dilation', 0.1),
			('module_config.hand_detailing.inpaint_full_res', 'Whole picture'),
			('module_config.hand_detailing.inpaint_full_res_padding', 32),
			('module_config.hand_detailing_opt.additional_parameter', ''),
			('resize_by_person_enabled', False),
			('resize_by_person', 0.85),
			('upscale_enabled', False),
			('detailing_after_upscale', True),
			('upscaler_name', 'None'),
			('upscale_ratio', 1.5),
			('module_config.controlnet.enabled', False),
			('module_config.controlnet.resize_by_person_enabled', False),
			('module_config.controlnet.resize_by_person', 0.4),
			('module_config.controlnet.noise', False),
			('module_config.controlnet.noise_strength', 0.7),
			('config_file', ''),
			('preset', 'None'),
		]

		self.ext_params = [
			('hand_detailing_before_hiresfix_enabled', False),
		]

	@staticmethod
	def get_dict_from_args(args, d):
		ar = {}
		if d is not None:
			ar = d
		for p in args:
			key = p[0]
			value = p[1]
			keys = key.split('.')
			cur = ar
			if len(keys) > 1:
				key = keys[-1]
				for k in keys[:-1]:
					if k not in cur:
						cur[k] = {}
					cur = cur[k]
			cur[key] = value
		return ar

	@staticmethod
	def get_param_from_dict(prefix, d):
		arr = []
		for key, value in d.items():
			if isinstance(value, dict):
				prefixz = prefix + key + '.'
				sub = Parameters.get_param_from_dict(prefixz, value)
				arr.extend(sub)
			else:
				arr.append((prefix + key, value))
		return arr

	def get_dict(self, args, external_config):
		if len(args) != len(self.params):
			print('Refresh webui first.')
			raise Exception('Refresh webui first.')

		if args[0]:
			args_list = [(self.params[idx][0], v) for idx, v in enumerate(args)]
			args_list.extend(self.ext_params)
			ar = Parameters.get_dict_from_args(args_list, None)
		else:
			self.params.extend(self.ext_params)
			ar = Parameters.get_dict_from_args(self.params, None)

		if external_config:
			cfgarg = Parameters.get_param_from_dict('', external_config)
			ar = Parameters.get_dict_from_args(cfgarg, ar)
			ar['enabled'] = True

		return ar

	def get_default(self):
		return [x[1] for x in self.params]

	def get_preset(self, prompt):
		config_file = None
		newprompt = []
		for line in prompt.split('\n'):
			if line.startswith('##'):
				config_file = line[2:]
				continue
			newprompt.append(line)
		if config_file is None:
			return prompt, {}

		cfg_dir = os.path.join(os.path.dirname(__file__), "../preset")
		json_file = os.path.join(cfg_dir, f'{config_file}.json')
		if not os.path.isfile(json_file):
			print(f'Not found configuration file {config_file}.json')
			return '\n'.join(newprompt), {}
		with open(json_file) as f:
			config = json.load(f)
		print('Loading config', json.dumps(config, indent=2))
		return '\n'.join(newprompt), config

	def load_preset(self, args):
		name = 'None'
		for (key, value), a in zip(self.params, args):
			if key == 'preset':
				name = a
		if name == 'None':
			return {}

		cfg_dir = os.path.join(os.path.dirname(__file__), "../preset")
		json_file = os.path.join(cfg_dir, f'{name}.json')
		if not os.path.isfile(json_file):
			print(f'Not found configuration file {name}.json')
			return {}
		with open(json_file) as f:
			config = json.load(f)
		print('Loading config', json.dumps(config, indent=2))
		return config

	def get_save_config_name(self, args):
		name = None
		for (key, value), a in zip(self.params, args):
			if key == 'config_file':
				name = a
		if name is None:
			return 'noname'
		return name

	def load_config(self, name):
		save_dir = os.path.join(os.path.dirname(__file__), "../saved")
		with open(os.path.join(save_dir, f'{name}.json'), 'r') as f:
			loaded_dict = json.load(f)
		default_args = Parameters.get_dict_from_args(self.params, None)
		loaded_args = Parameters.get_param_from_dict('', loaded_dict)
		final_dict = Parameters.get_dict_from_args(loaded_args, default_args)
		final_args = Parameters.get_param_from_dict('', final_dict)
		sort_dict = {a[0]: a[1] for a in final_args}
		ret = [sort_dict[key] for key, value in self.params]
		return ret

	def save_config(self, args):
		name = 'noname'
		for (key, value), a in zip(self.params, args):
			if key == 'config_file':
				name = a

		save_dir = os.path.join(os.path.dirname(__file__), "../saved")
		args_list = [(self.params[idx][0], v) for idx, v in enumerate(args)]
		conf = Parameters.get_dict_from_args(args_list, None)
		with open(os.path.join(save_dir, f'{name}.json'), 'w') as f:
			json.dump(conf, f, indent=2)

	def list_config(self):
		save_dir = os.path.join(os.path.dirname(__file__), "../saved")
		if not os.path.isdir(save_dir):
			os.mkdir(save_dir)

		configs = [x for x in os.listdir(save_dir) if x.endswith('.json')]

		return [x[:-5] for x in configs]

	def list_preset(self):
		presets = ['None']
		preset_dir = os.path.join(os.path.dirname(__file__), "../preset")
		configs = [x for x in os.listdir(preset_dir) if x.endswith('.json')]
		presets.extend([x[:-5] for x in configs])
		return presets

