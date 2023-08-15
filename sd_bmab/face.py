from PIL import Image
from PIL import ImageEnhance
from PIL import ImageDraw

from sd_bmab import dinosam, util


def process_face_lighting(a, p, images):
	if (a['face_lighting_enabled'] and a['face_lighting'] != 0) or a.get('module_config', {}).get('multiple_face'):
		for idx in range(0, len(images)):
			pidx = p.iteration * p.batch_size + idx
			a['current_prompt'] = p.all_prompts[pidx]
			img = util.tensor_to_image(images[idx])
			img = process_face_lighting_inner(a, p, img)
			images[idx] = util.image_to_tensor(img)


def process_face_lighting_inner(args, p, img):
	multiple_face = args.get('module_config', {}).get('multiple_face', [])
	if multiple_face:
		return process_multiple_face(args, p, img)

	if args['face_lighting'] == 0:
		return img

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(img, 'face')
	# print(float(logits))
	print(phrases)

	org_size = img.size
	print('size', org_size)

	face_config = dict(args.get('module_config', {}).get('face_lighting', {}))
	enhancer = ImageEnhance.Brightness(img)
	bgimg = enhancer.enhance(1 + args['face_lighting'])

	prompt = face_config.get('prompt')
	current_prompt = args.get('current_prompt', '')
	if prompt is not None and prompt.find('#!org!#') >= 0:
		face_config['prompt'] = face_config['prompt'].replace('#!org!#', current_prompt)
		print('prompt for face', face_config['prompt'])

	for box in boxes:
		face_mask = dinosam.sam_predict_box(img, box)
		img.paste(bgimg, mask=face_mask)
		options = dict(mask=face_mask)
		options.update(face_config)
		img = util.process_img2img(p, img, options=options)

	return img


def process_multiple_face(args, p, img):
	multiple_face = list(args.get('module_config', {}).get('multiple_face', []))
	print('processing multiple face')

	limit = len(multiple_face)
	if limit == 0:
		return img

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(img, 'face')

	org_size = img.size
	print('size', org_size)

	# sort
	candidate = []
	for box, logit, phrase in zip(boxes, logits, phrases):
		print('detected', phrase, float(logit))
		x1, y1, x2, y2 = box
		size = (x2 - x1) * (y2 - y1)
		candidate.append((size, box, logit, phrase))
	candidate = sorted(candidate, key=lambda c: c[0], reverse=True)

	for idx, (size, box, logit, phrase) in enumerate(candidate):
		if idx == limit:
			break

		x1, y1, x2, y2 = box
		x1 = int(x1) - 4
		y1 = int(y1) - 4
		x2 = int(x2) + 4
		y2 = int(y2) + 4

		face_mask = Image.new('L', img.size, color=0)
		dr = ImageDraw.Draw(face_mask, 'L')
		dr.rectangle((x1, y1, x2, y2), fill=255)

		# face_mask = dinosam.sam_predict_box(img, box)

		options = dict(mask=face_mask)

		prompt = multiple_face[idx].get('prompt')
		current_prompt = args.get('current_prompt', '')
		if prompt is not None and prompt.find('#!org!#') >= 0:
			multiple_face[idx]['prompt'] = multiple_face[idx]['prompt'].replace('#!org!#', current_prompt)
			print('prompt for face', multiple_face[idx]['prompt'])

		options.update(multiple_face[idx])
		img = util.process_img2img(p, img, options=options)

	return img
