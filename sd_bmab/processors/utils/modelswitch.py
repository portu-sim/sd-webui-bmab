
from PIL import Image

from modules import shared, sd_models

from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


base_sd_model = None


def change_model(name):
	if name is None:
		return
	info = sd_models.get_closet_checkpoint_match(name)
	if info is None:
		print(f'Unknown model: {name}')
		return
	sd_models.reload_model_weights(shared.sd_model, info)


class ApplyModel(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		return shared.opts.bmab_use_specific_model

	def process(self, context: Context, image: Image):
		global base_sd_model
		base_sd_model = shared.opts.data['sd_model_checkpoint']
		change_model(shared.opts.bmab_model)
		return image

	def postprocess(self, context: Context, image: Image):
		pass


class RollbackModel(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		return shared.opts.bmab_use_specific_model

	def process(self, context: Context, image: Image):
		global base_sd_model
		if base_sd_model is not None:
			change_model(base_sd_model)
			base_sd_model = None
		return image

	def postprocess(self, context: Context, image: Image):
		pass
