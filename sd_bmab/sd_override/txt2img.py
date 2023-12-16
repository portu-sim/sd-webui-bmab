import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any
from modules import processing
from modules import sd_samplers
from modules import images
from modules import devices
from modules import extra_networks
from modules import sd_models
from ..external.rng import rng
from ..external.rng.rng import ImageRNG
from modules import shared
from modules.shared import opts, state, sd_model
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, decode_first_stage, create_random_tensors
from modules.sd_hijack_hypertile import hypertile_set
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes

from sd_bmab.base import filter
from sd_bmab.external.kohyahiresfix import KohyaHiresFixPreprocessor


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

        self.bscript = None
        self.bscript_args = None
        self.extra_noise = 0
        self.initial_noise_multiplier = opts.initial_noise_multiplier

    
    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        with KohyaHiresFixPreprocessor(self):
            
            hypertile_set(self)
            x = create_random_tensors([4, self.height // 8, self.width // 8], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
            x *= self.initial_noise_multiplier

            self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
            samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))

            del x

            if not self.enable_hr:
                return samples

            if self.latent_scale_mode is None:
                decoded_samples = torch.stack(processing.decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)).to(dtype=torch.float32)
            else:
                decoded_samples = None

            with sd_models.SkipWritingToConfig():
                sd_models.reload_model_weights(info=self.hr_checkpoint_info)

            devices.torch_gc()

        return self.sample_hr_pass(samples, decoded_samples, seeds, subseeds, subseed_strength, prompts)

    
    def sample_hr_pass(self, samples, decoded_samples, seeds, subseeds, subseed_strength, prompts):
        if shared.state.interrupted:
            return samples

        self.is_hr_pass = True

        target_width = self.hr_upscale_to_x
        target_height = self.hr_upscale_to_y

        def save_intermediate(image, index):
            """saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""

            if not self.save_samples() or not opts.save_images_before_highres_fix:
                return

            if not isinstance(image, Image.Image):
                image = sd_samplers.sample_to_image(image, index, approximation=0)

            info = processing.create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [], iteration=self.iteration, position_in_batch=index)
            images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format, info=info, p=self, suffix="-before-highres-fix")

        img2img_sampler_name = self.hr_sampler_name or self.sampler_name

        self.sampler = sd_samplers.create_sampler(img2img_sampler_name, self.sd_model)

        if self.latent_scale_mode is not None:
            for i in range(samples.shape[0]):
                save_intermediate(samples, i)

            samples = torch.nn.functional.interpolate(samples, size=(target_height // processing.opt_f, target_width // processing.opt_f), mode=self.latent_scale_mode["mode"], antialias=self.latent_scale_mode["antialias"])

            # Avoid making the inpainting conditioning unless necessary as
            # this does need some extra compute to decode / encode the image again.
            if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples), samples)
            else:
                image_conditioning = self.txt2img_image_conditioning(samples)
        else:
            lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

            batch_images = []
            for i, x_sample in enumerate(lowres_samples):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)

                save_intermediate(image, i)

                if self.bscript_args is not None:
                    filter_name = self.bscript_args['txt2img_filter_hresfix_before_upscale']
                    filter1 = filter.get_filter(filter_name)
                    filter.preprocess_filter(filter1, None)
                    filter.process_filter(filter1, None, None, image, sdprocess=self)
                    filter.postprocess_filter(filter1, None)

                    if hasattr(self.bscript, 'resize_image'):
                        resized = self.bscript.resize_image(self, self.bscript_args, i, 0, image, target_width, target_height, self.hr_upscaler)
                    else:
                        resized = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)

                    filter_name = self.bscript_args['txt2img_filter_hresfix_after_upscale']
                    filter2 = filter.get_filter(filter_name)
                    filter.preprocess_filter(filter2, None)
                    image = filter.process_filter(filter2, None, image, resized, sdprocess=self)
                    filter.postprocess_filter(filter2, None)
                else:
                    if hasattr(self.bscript, 'resize_image'):
                        image = self.bscript.resize_image(self, self.bscript_args, i, 0, image, target_width, target_height, self.hr_upscaler)
                    else:
                        image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)

                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                batch_images.append(image)

            decoded_samples = torch.from_numpy(np.array(batch_images))
            decoded_samples = decoded_samples.to(shared.device, dtype=devices.dtype_vae)

            if opts.sd_vae_encode_method != 'Full':
                self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method
            samples = images_tensor_to_samples(decoded_samples, approximation_indexes.get(opts.sd_vae_encode_method))

            image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

        shared.state.nextjob()

        samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]
    
        self.rng = rng.ImageRNG(samples.shape[1:], self.seeds, subseeds=self.subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w)
        noise = self.rng.next()

        # GC now before running the next img2img to prevent running out of memory
        devices.torch_gc()

        if not self.disable_extra_networks:
            with devices.autocast():
                extra_networks.activate(self, self.hr_extra_network_data)

        with devices.autocast():
            self.calculate_hr_conds()

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))

        if self.scripts is not None:
            self.scripts.before_hr(self)

        if self.initial_noise_multiplier != 1.0:
            self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
            noise *= self.initial_noise_multiplier

        samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())

        self.sampler = None
        devices.torch_gc()

        decoded_samples = processing.decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)

        self.is_hr_pass = False

        return decoded_samples

    




