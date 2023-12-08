from modules import shared, sd_models
from modules.shared import opts, state, sd_model


class VAEMethodOverride:

	def __init__(self, hiresfix=False) -> None:
		super().__init__()
		self.org_encode_method = None
		self.org_decode_method = None
		self.img2img_fix_steps = None
		self.hiresfix = hiresfix

	def __enter__(self):
		if ('sd_vae_encode_method' in shared.opts.data) and shared.opts.bmab_detail_full:
			self.encode_method = shared.opts.sd_vae_encode_method
			self.decode_method = shared.opts.sd_vae_decode_method
			shared.opts.sd_vae_encode_method = 'Full'
			shared.opts.sd_vae_decode_method = 'Full'
		if self.hiresfix and not shared.opts.img2img_fix_steps:
			self.img2img_fix_steps = shared.opts.img2img_fix_steps
			shared.opts.img2img_fix_steps = True

	def __exit__(self, *args, **kwargs):
		if ('sd_vae_encode_method' in shared.opts.data) and shared.opts.bmab_detail_full:
			shared.opts.sd_vae_encode_method = self.encode_method
			shared.opts.sd_vae_decode_method = self.decode_method
		if self.img2img_fix_steps is not None:
			shared.opts.img2img_fix_steps = self.img2img_fix_steps
