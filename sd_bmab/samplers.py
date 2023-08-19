import modules
import k_diffusion.sampling
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.sd_samplers import set_samplers
from modules.shared import opts, state
import modules.shared as shared
from modules import sd_samplers_common
from sd_bmab import sdprocessing


class SamplerCallBack(object):
	def __init__(self, s, ar) -> None:
		super().__init__()
		self.script = s
		self.args = ar
		self.is_break = False

	def initialize(self, p):
		pass

	def callback_state(self, d):
		pass

	def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning):
		pass

	def sample(self, p, x, conditioning, unconditional_conditioning, steps, image_conditioning):
		pass


class KDiffusionSamplerOv(KDiffusionSampler):

	def __init__(self, funcname, sd_model):
		super().__init__(funcname, sd_model)
		self.callback = None
		self.block_tqdm = False
		self.p = None

	def initialize(self, p):
		self.p = p
		if isinstance(p, sdprocessing.StableDiffusionProcessingImg2ImgOv):
			self.block_tqdm = p.block_tqdm
		if self.callback:
			self.callback.initialize(p)
		return super().initialize(p)

	def callback_state(self, d):
		if self.callback:
			self.callback.callback_state(d)

		step = d['i']
		latent = d["denoised"]
		if opts.live_preview_content == "Combined":
			sd_samplers_common.store_latent(latent)

		self.last_latent = latent

		if self.stop_at is not None and step > self.stop_at:
			raise sd_samplers_common.InterruptedException

		state.sampling_step = step
		if not self.block_tqdm:
			shared.total_tqdm.update()

	def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
		if self.callback:
			self.callback.sample_img2img(p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning)
			if self.callback.is_break:
				return x
		return super().sample_img2img(p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning)

	def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
		if self.callback:
			self.callback.sample(p, x, conditioning, unconditional_conditioning, steps, image_conditioning)
		samples = super().sample(p, x, conditioning, unconditional_conditioning, steps, image_conditioning)

		if hasattr(p, 'end_sample'):
			p.end_sample(samples)

		return samples

	def register_callback(self, cb):
		self.callback = cb


def override_samplers():
	modules.sd_samplers_kdiffusion.samplers_data_k_diffusion = [
		modules.sd_samplers_common.SamplerData(label,
											   lambda model, funcname=funcname: KDiffusionSamplerOv(funcname, model),
											   aliases, options)
		for label, funcname, aliases, options in modules.sd_samplers_kdiffusion.samplers_k_diffusion
		if hasattr(k_diffusion.sampling, funcname)
	]
	modules.sd_samplers.all_samplers = [
		*modules.sd_samplers_kdiffusion.samplers_data_k_diffusion,
		*modules.sd_samplers_compvis.samplers_data_compvis,
	]
	modules.sd_samplers.all_samplers_map = {x.name: x for x in modules.sd_samplers.all_samplers}
