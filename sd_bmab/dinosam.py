import cv2
import os
import numpy as np

import torch

from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict, annotate
from modules.paths import models_path
from modules.safe import unsafe_torch_load, load
from modules.devices import device, torch_gc, cpu

from torchvision.ops import box_convert
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
import groundingdino.datasets.transforms as T

bmab_model_path = os.path.join(models_path, "bmab")

dino_model = None
sam_model = None


def dino_init():
	global dino_model
	if not dino_model:
		dino_model = load_model('%s/GroundingDINO_SwinT_OGC.py' % bmab_model_path, '%s/groundingdino_swint_ogc.pth' % bmab_model_path)
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


def sam_init():
	MODEL_TYPE = 'vit_b'

	global sam_model
	if not sam_model:
		torch.load = unsafe_torch_load
		sam_model = sam_model_registry[MODEL_TYPE](checkpoint='%s/sam_vit_b_01ec64.pth' % bmab_model_path)
		sam_model.to(device=device)
		sam_model.eval()
		torch.load = load

	return sam_model


def sam_predict(pilimg, boxes):
	sam = sam_init()

	mask_predictor = SamPredictor(sam)

	numpy_image = np.array(pilimg)
	opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
	mask_predictor.set_image(opencv_image)

	result = Image.new('L', pilimg.size, 0)
	for box in boxes:
		x1, y1, x2, y2 = box

		box = np.array([int(x1), int(y1), int(x2), int(y2)])
		masks, scores, logits = mask_predictor.predict(
			box=box,
			multimask_output=False
		)

		mask = Image.fromarray(masks[0])
		result.paste(mask, mask=mask)

	return result


def sam_predict_box(pilimg, box):
	sam = sam_init()

	mask_predictor = SamPredictor(sam)

	numpy_image = np.array(pilimg)
	opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
	mask_predictor.set_image(opencv_image)

	x1, y1, x2, y2 = box
	box = np.array([int(x1), int(y1), int(x2), int(y2)])

	masks, scores, logits = mask_predictor.predict(
		box=box,
		multimask_output=False
	)

	return Image.fromarray(masks[0])

