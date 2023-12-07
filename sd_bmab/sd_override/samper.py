import modules
import k_diffusion.sampling
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.sd_samplers import set_samplers
from modules.shared import opts, state, shared
import inspect
from modules import sd_samplers_common
#from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
from .script_callbacks_extranoise import ExtraNoiseParams, extra_noise_callback, on_extra_noise


class KDiffusionSamplerBMAB(KDiffusionSampler):
	def __init__(self, funcname, sd_model=shared.sd_model):
		
		def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
			steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
	
			sigmas = self.get_sigmas(p, steps)
			sigma_sched = sigmas[steps - t_enc - 1:]
	
			xi = x + noise * sigma_sched[0]
	
			if opts.img2img_extra_noise > 0:
				p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
				extra_noise_params = ExtraNoiseParams(noise, x, xi)
				extra_noise_callback(extra_noise_params)
				noise = extra_noise_params.noise
				xi += noise * opts.img2img_extra_noise
	
			if hasattr(p, 'extra_noise') and p.extra_noise > 0:
				p.extra_generation_params["Extra noise"] = p.extra_noise
				extra_noise_params = ExtraNoiseParams(noise, x, xi)
				extra_noise_callback(extra_noise_params)
				noise = extra_noise_params.noise
				xi += noise * p.extra_noise
	
			extra_params_kwargs = self.initialize(p)
			parameters = inspect.signature(self.func).parameters
	
			if 'sigma_min' in parameters:
				## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
				extra_params_kwargs['sigma_min'] = sigma_sched[-2]
			if 'sigma_max' in parameters:
				extra_params_kwargs['sigma_max'] = sigma_sched[0]
			if 'n' in parameters:
				extra_params_kwargs['n'] = len(sigma_sched) - 1
			if 'sigma_sched' in parameters:
				extra_params_kwargs['sigma_sched'] = sigma_sched
			if 'sigmas' in parameters:
				extra_params_kwargs['sigmas'] = sigma_sched
	
			if self.config.options.get('brownian_noise', False):
				noise_sampler = self.create_noise_sampler(x, sigmas, p)
				extra_params_kwargs['noise_sampler'] = noise_sampler
	
			if self.config.options.get('solver_type', None) == 'heun':
				extra_params_kwargs['solver_type'] = 'heun'
	
			self.model_wrap_cfg.init_latent = x
			self.last_latent = x
			self.sampler_extra_args = {
				'cond': conditioning,
				'image_cond': image_conditioning,
				'uncond': unconditional_conditioning,
				'cond_scale': p.cfg_scale,
				's_min_uncond': self.s_min_uncond
			}
	
			samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))
	
			if self.model_wrap_cfg.padded_cond_uncond:
				p.extra_generation_params["Pad conds"] = True
	
			return samples


def override_samplers():
	modules.sd_samplers_kdiffusion.samplers_data_k_diffusion = [
		modules.sd_samplers_common.SamplerData(label,
											   lambda model, funcname=funcname: KDiffusionSamplerBMAB(funcname, model),
											   aliases, options)
		for label, funcname, aliases, options in modules.sd_samplers_kdiffusion.samplers_k_diffusion
		if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
	]
	if hasattr(modules, 'sd_samplers_timesteps'):
		modules.sd_samplers.all_samplers = [
			*modules.sd_samplers_kdiffusion.samplers_data_k_diffusion,
			*modules.sd_samplers_timesteps.samplers_data_timesteps,
		]
	else:
		modules.sd_samplers.all_samplers = [
			*modules.sd_samplers_kdiffusion.samplers_data_k_diffusion,
			*modules.sd_samplers_compvis.samplers_data_compvis,
		]
	modules.sd_samplers.all_samplers_map = {x.name: x for x in modules.sd_samplers.all_samplers}
