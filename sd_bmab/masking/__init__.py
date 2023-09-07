from sd_bmab.base import MaskBase
from sd_bmab.masking.sam import SamPredictVitB
from sd_bmab.masking.sam_hq import SamHqPredictVitB

masks = [SamPredictVitB(), SamHqPredictVitB()]
dmasks = {s.name: s for s in masks}


def get_mask_generator(name='None') -> MaskBase:
	return dmasks.get(name, masks[0])


def release():
	SamPredictVitB.release()
	SamHqPredictVitB.release()
