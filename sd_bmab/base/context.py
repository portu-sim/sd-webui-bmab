from modules import shared
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img


class Context(object):
	def __init__(self, s, p, a, idx, hiresfix=False) -> None:
		super().__init__()

		self.script = s
		self.sdprocessing = p
		self.args = a
		self.index = idx
		self.hiresfix = hiresfix
		self.controlnet_count = 0
		self.refiner = None

	@staticmethod
	def newContext(s, p, a, idx, **kwargs):
		return Context(s, p, a, idx, **kwargs)

	def get_current_prompt(self):
		return self.sdprocessing.prompt

	def get_prompt_by_index(self):
		if self.sdprocessing.all_prompts is None or len(self.sdprocessing.all_prompts) <= self.index:
			return self.sdprocessing.prompt
		return self.sdprocessing.all_prompts[self.index]

	def get_negative_prompt_by_index(self):
		if self.sdprocessing.all_negative_prompts is None or len(self.sdprocessing.all_negative_prompts) <= self.index:
			return self.sdprocessing.negative_prompt
		return self.sdprocessing.all_negative_prompts[self.index]

	def get_seeds(self):
		if self.sdprocessing.all_seeds is None or self.sdprocessing.all_subseeds is None:
			return self.sdprocessing.seed, self.sdprocessing.subseed
		if len(self.sdprocessing.all_seeds) <= self.index or len(self.sdprocessing.all_subseeds) <= self.index:
			return self.sdprocessing.seed, self.sdprocessing.subseed
		return self.sdprocessing.all_seeds[self.index], self.sdprocessing.all_subseeds[self.index]

	def get_max_area(self):
		if shared.opts.bmab_optimize_vram == 'low vram':
			return 512 * 768
		elif shared.opts.bmab_optimize_vram == 'med vram':
			return self.sdprocessing.width * self.sdprocessing.height
		if isinstance(self.sdprocessing, StableDiffusionProcessingTxt2Img) and self.sdprocessing.enable_hr:
			return self.sdprocessing.hr_upscale_to_x * self.sdprocessing.hr_upscale_to_y
		return self.sdprocessing.width * self.sdprocessing.height

	def add_generation_param(self, key, value):
		self.sdprocessing.extra_generation_params[key] = value

	def with_refiner(self):
		return self.args.get('refiner_enabled', False)

	def is_refiner_context(self):
		return self.refiner is not None

	def is_hires_fix(self):
		if isinstance(self.sdprocessing, StableDiffusionProcessingTxt2Img) and self.sdprocessing.enable_hr:
			return True
		return False

	def add_job(self):
		shared.state.job_count += 1
		shared.state.sampling_step = 0
		shared.state.current_image_sampling_step = 0

	def is_img2img(self):
		return isinstance(self.sdprocessing, StableDiffusionProcessingImg2Img)
