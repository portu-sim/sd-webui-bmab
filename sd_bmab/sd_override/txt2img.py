import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass

import modules
from modules.processing_class import StableDiffusionProcessingTxt2Img
from modules.processing_helpers import txt2img_image_conditioning
from modules import processing
from modules import sd_samplers
from modules import images
from modules import devices
from modules import sd_models
from modules import shared
from modules.shared import opts
from modules.processing import decode_first_stage, create_random_tensors
from modules.sd_hijack_hypertile import hypertile_set

from sd_bmab.base import filter


class SkipWritingToConfig:
    """This context manager prevents load_model_weights from writing checkpoint name to the config when it loads weight."""

    skip = False
    previous = None

    def __enter__(self):
        self.previous = SkipWritingToConfig.skip
        SkipWritingToConfig.skip = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        SkipWritingToConfig.skip = self.previous


@dataclass(repr=False)
class StableDiffusionProcessingTxt2ImgOv(StableDiffusionProcessingTxt2Img):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.shape=[4, self.height // 8, self.width // 8]
        
        self.bscript = None
        self.bscript_args = None
        self.extra_noise = 0
        self.initial_noise_multiplier = opts.initial_noise_multiplier

    def init(self, all_prompts, all_seeds, all_subseeds):
        ret = super().init(all_prompts, all_seeds, all_subseeds)
        self.extra_generation_params['Hires prompt'] = ''
        self.extra_generation_params['Hires negative prompt'] = ''
        return ret
    
    def txt2img_image_conditioning(p, x, width=None, height=None):
        width = width or p.width
        height = height or p.height
        if p.sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models
            image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
            image_conditioning = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(image_conditioning))
            image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0) # pylint: disable=not-callable
            image_conditioning = image_conditioning.to(x.dtype)
            return image_conditioning
        elif p.sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models
            return x.new_zeros(x.shape[0], 2*p.sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)
        else:
            return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)
    
    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):

        latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler, None) if self.hr_upscaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "None")
        if latent_scale_mode is not None:
            self.hr_force = False # no need to force anything
        if self.enable_hr and (latent_scale_mode is None or self.hr_force):
            if len([x for x in shared.sd_upscalers if x.name == self.hr_upscaler]) == 0:
                shared.log.warning(f"Cannot find upscaler for hires: {self.hr_upscaler}")
                self.enable_hr = False

        self.ops.append('txt2img')
        hypertile_set(self)
        self.sampler = modules.sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        if hasattr(self.sampler, "initialize"):
            self.sampler.initialize(self)
        x = create_random_tensors([4, self.height // 8, self.width // 8], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
        shared.state.nextjob()
        if not self.enable_hr or shared.state.interrupted or shared.state.skipped:
            return samples

        self.init_hr()
        if self.is_hr_pass:
            prev_job = shared.state.job
            target_width = self.hr_upscale_to_x
            target_height = self.hr_upscale_to_y
            decoded_samples = None
            if shared.opts.save and shared.opts.save_images_before_highres_fix and not self.do_not_save_samples:
                decoded_samples = decode_first_stage(self.sd_model, samples.to(dtype=devices.dtype_vae), self.full_quality)
                decoded_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
                for i, x_sample in enumerate(decoded_samples):
                    x_sample = processing.validate_sample(x_sample)
                    image = Image.fromarray(x_sample)
                    bak_extra_generation_params, bak_restore_faces = self.extra_generation_params, self.restore_faces
                    self.extra_generation_params = {}
                    self.restore_faces = False
                    info = processing.create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [], iteration=self.iteration, position_in_batch=i)
                    self.extra_generation_params, self.restore_faces = bak_extra_generation_params, bak_restore_faces
                    images.save_image(image, self.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=info, suffix="-before-hires")
            if latent_scale_mode is None or self.hr_force: # non-latent upscaling
                shared.state.job = 'upscale'
                if decoded_samples is None:
                    decoded_samples = decode_first_stage(self.sd_model, samples.to(dtype=devices.dtype_vae), self.full_quality)
                    decoded_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
                batch_images = []
                for _i, x_sample in enumerate(decoded_samples):
                    x_sample = processing.validate_sample(x_sample)
                    image = Image.fromarray(x_sample)

                    if self.bscript_args is not None:
                        filter_name = self.bscript_args['txt2img_filter_hresfix_before_upscale']
                        filter1 = filter.get_filter(filter_name)
                        from sd_bmab.base import Context
                        context = Context(self.bscript, self, self.bscript_args, i)
                        filter.preprocess_filter(filter1, context, image)
                        image = filter.process_filter(filter1, context, None, image, sdprocess=self)
                        filter.postprocess_filter(filter1, context)

                        if hasattr(self.bscript, 'resize_image'):
                            resized = self.bscript.resize_image(self, self.bscript_args, 0, i, image, target_width, target_height, self.hr_upscaler)
                        else:
                            image = images.resize_image(1, image, target_width, target_height, upscaler_name=self.hr_upscaler)

                        filter_name = self.bscript_args['txt2img_filter_hresfix_after_upscale']
                        filter2 = filter.get_filter(filter_name)
                        filter.preprocess_filter(filter2, context, image)
                        image = filter.process_filter(filter2, context, image, resized, sdprocess=self)
                        filter.postprocess_filter(filter2, context)
                    else:
                        if hasattr(self.bscript, 'resize_image'):
                            image = self.bscript.resize_image(self, self.bscript_args, 0, i, image, target_width, target_height, self.hr_upscaler)
                        else:
                            image = images.resize_image(1, image, target_width, target_height, upscaler_name=self.hr_upscaler)

                    image = np.array(image).astype(np.float32) / 255.0
                    image = np.moveaxis(image, 2, 0)
                    batch_images.append(image)
                resized_samples = torch.from_numpy(np.array(batch_images))
                resized_samples = resized_samples.to(device=shared.device, dtype=devices.dtype_vae)
                resized_samples = 2.0 * resized_samples - 1.0
                if shared.opts.sd_vae_sliced_encode and len(decoded_samples) > 1:
                    samples = torch.stack([self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(torch.unsqueeze(resized_sample, 0)))[0] for resized_sample in resized_samples])
                else:
                    # TODO add TEASD support
                    samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(resized_samples))
                image_conditioning = self.img2img_image_conditioning(resized_samples, samples)
            else:
                samples = torch.nn.functional.interpolate(samples, size=(target_height // 8, target_width // 8), mode=latent_scale_mode["mode"], antialias=latent_scale_mode["antialias"])
                if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                    image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples.to(dtype=devices.dtype_vae), self.full_quality), samples)
                else:
                    image_conditioning = self.txt2img_image_conditioning(samples.to(dtype=devices.dtype_vae))
                if self.latent_sampler == "PLMS":
                    self.latent_sampler = 'UniPC'
            if self.hr_force or latent_scale_mode is not None:
                shared.state.job = 'hires'
                if self.denoising_strength > 0:
                    self.ops.append('hires')
                    devices.torch_gc() # GC now before running the next img2img to prevent running out of memory
                    self.sampler = modules.sd_samplers.create_sampler(self.latent_sampler or self.sampler_name, self.sd_model)
                    if hasattr(self.sampler, "initialize"):
                        self.sampler.initialize(self)
                    samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]
                    noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, p=self)
                    modules.sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))
                    hypertile_set(self, hr=True)
                    
                    with sd_models.SkipWritingToConfig():
                        sd_models.reload_model_weights(info=self.hr_checkpoint_info)
                    
                    samples = self.sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)
                    modules.sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())
                else:
                    self.ops.append('upscale')
            x = None
            self.is_hr_pass = False
            shared.state.job = prev_job
            shared.state.nextjob()

        return samples


    




