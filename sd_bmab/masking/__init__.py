from modules import shared

from sd_bmab.base import MaskBase
from sd_bmab.masking.sam import SamPredictVitB
from sd_bmab.masking.sam_hq import SamHqPredictVitB
from sd_bmab.util import debug_print


masks = [SamPredictVitB(), SamHqPredictVitB()]
dmasks = {s.name: s for s in masks}


def get_mask_generator(name='None') -> MaskBase:
	model = dmasks.get(name, dmasks[shared.opts.bmab_mask_model])
	debug_print(f'Use mask model {model.name}')
	return model


def list_mask_names():
	return [s.name for s in masks]


def release():
	SamPredictVitB.release()
	SamHqPredictVitB.release()
