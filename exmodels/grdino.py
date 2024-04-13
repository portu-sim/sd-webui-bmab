import numpy as np

import torch

from groundingdino.util.inference import load_model, predict
from modules.devices import device, torch_gc

from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T

from sd_bmab import util


dino_model = None


def dino_init():
	global dino_model
	if not dino_model:
		swint_ogc = util.lazy_loader('GroundingDINO_SwinT_OGC.py')
		swint_ogc_pth = util.lazy_loader('groundingdino_swint_ogc.pth')
		dino_model = load_model(swint_ogc, swint_ogc_pth)
	return dino_model


def dino_predict(pilimg, prompt, box_threahold=0.35, text_threshold=0.25):
	transform = T.Compose(
		[
			T.RandomResize([800], max_size=1333),
			T.ToTensor(),
			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)
	img = pilimg.convert('RGB')
	image_source = np.asarray(img)
	image, _ = transform(img, None)

	model = dino_init()
	boxes, logits, phrases = predict(
		device='cuda:0',
		model=model,
		image=image,
		caption=prompt,
		box_threshold=box_threahold,
		text_threshold=text_threshold
	)

	h, w, _ = image_source.shape
	boxes = boxes * torch.Tensor([w, h, w, h])
	annotated_frame = box_convert(boxes=boxes, in_fmt='cxcywh', out_fmt='xyxy').numpy()

	return annotated_frame, logits, phrases


def release():
	global dino_model
	dino_model = None
	torch_gc()
