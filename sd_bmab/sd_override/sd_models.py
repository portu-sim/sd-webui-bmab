import os
import sys
from modules import sd_models
from modules import shared


def bmab_list_models():
	sd_models.checkpoints_list.clear()
	sd_models.checkpoint_aliases.clear()

	cmd_ckpt = shared.cmd_opts.ckpt
	if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
		model_url = None
	else:
		model_url = f"{shared.hf_endpoint}/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"

	model_list = sd_models.modelloader.load_models(model_path=sd_models.model_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

	second_path = shared.opts.data.get('bmab_additional_checkpoint_path', '')
	print(f'second path {second_path}')
	if os.path.exists(second_path):
		print(f'load checkpoint from {second_path}')
		model_list_seconds = sd_models.modelloader.load_models(model_path=second_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])
		length = len(sd_models.model_path)
		temp = [(x[length:], x) for x in model_list]
		length = len(second_path)
		temp.extend([(x[length:], x) for x in model_list_seconds])
		model_list = [x[1] for x in sorted(temp, key=lambda x: x[0])]

	if os.path.exists(cmd_ckpt):
		checkpoint_info = sd_models.CheckpointInfo(cmd_ckpt)
		checkpoint_info.register()
		shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
	elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
		print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {sd_models.model_path}: {cmd_ckpt}", file=sys.stderr)

	for filename in model_list:
		checkpoint_info = sd_models.CheckpointInfo(filename)
		checkpoint_info.register()


def override():
	bmab_list_models()
	sd_models.list_models = bmab_list_models

