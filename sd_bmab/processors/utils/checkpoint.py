from PIL import Image

from sd_bmab import constants
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase
from sd_bmab.util import debug_print


class CheckPointChanger(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.checkpoint = constants.checkpoint_default
		self.vae = constants.vae_default
		self.loaded_vae_file = None  # Initialize loaded_vae_file attribute

	def preprocess(self, context: Context, image: Image):
		self.checkpoint = context.args['preprocess_checkpoint']
		self.vae = context.args['preprocess_vae']
		self.loaded_vae_file = context.args.get('loaded_vae_file')
		return not (self.checkpoint == constants.checkpoint_default and self.checkpoint == constants.vae_default)

	def process(self, context: Context, image: Image):
		debug_print('Change checkpoint', self.checkpoint, self.vae, self.loaded_vae_file)
		context.save_and_apply_checkpoint(self.checkpoint, self.vae, self.loaded_vae_file)
		return image


class CheckPointRestore(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()

	def preprocess(self, context: Context, image: Image):
		return True

	def process(self, context: Context, image: Image):
		context.restore_checkpoint()
		return image
