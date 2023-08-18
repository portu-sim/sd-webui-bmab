from PIL import Image
from PIL import ImageEnhance
from PIL import ImageDraw

from modules import devices
from sd_bmab import dinosam, util, process


def get_mask(img, prompt):
	boxes, logits, phrases = dinosam.dino_predict(img, prompt)
	sam_mask = dinosam.sam_predict_box(img, boxes[0])
	return sam_mask


def process_face_detailing(a, p, image):
	if a['face_detailing_enabled'] or a.get('module_config', {}).get('multiple_face'):
		return process_face_detailing_inner(a, p, image)
	return image


def process_face_detailing_old(a, p, images):
	if a['face_detailing_enabled'] or a.get('module_config', {}).get('multiple_face'):
		for idx in range(0, len(images)):
			pidx = p.iteration * p.batch_size + idx
			a['current_prompt'] = p.all_prompts[pidx]
			img = util.tensor_to_image(images[idx])
			img = process_face_detailing_inner(a, p, img)
			images[idx] = util.image_to_tensor(img)


def process_face_detailing_inner(args, p, img):
	multiple_face = args.get('module_config', {}).get('multiple_face', [])
	if multiple_face:
		return process_multiple_face(args, p, img)

	dilation = args.get('module_config', {}).get('face_detailing_opt', {}).get('mask dilation', 4)

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(img, 'face')
	# print(float(logits))
	print(phrases)

	org_size = img.size
	print('size', org_size)

	face_config = dict(args.get('module_config', {}).get('face_detailing', {}))

	prompt = face_config.get('prompt')
	current_prompt = args.get('current_prompt', '')
	if prompt is not None and prompt.find('#!org!#') >= 0:
		face_config['prompt'] = face_config['prompt'].replace('#!org!#', current_prompt)
		print('prompt for face', face_config['prompt'])

	for box, logit, phrase in zip(boxes, logits, phrases):
		print('render', phrase, float(logit))
		x1, y1, x2, y2 = box
		x1 = int(x1) - dilation
		y1 = int(y1) - dilation
		x2 = int(x2) + dilation
		y2 = int(y2) + dilation

		face_mask = Image.new('L', img.size, color=0)
		dr = ImageDraw.Draw(face_mask, 'L')
		dr.rectangle((x1, y1, x2, y2), fill=255)

		print('face lighting', args['face_lighting'])
		if args['face_lighting'] != 0:
			sam_mask = dinosam.sam_predict_box(img, box)
			enhancer = ImageEnhance.Brightness(img)
			bgimg = enhancer.enhance(1 + args['face_lighting'])
			img.paste(bgimg, mask=sam_mask)
			p.extra_generation_params['BMAB face lighting'] = args['face_lighting']

		options = dict(mask=face_mask, **face_config)
		img = process.process_img2img(p, img, options=options)
	devices.torch_gc()
	return img


def process_multiple_face(args, p, img):
	multiple_face = list(args.get('module_config', {}).get('multiple_face', []))
	order = args.get('module_config', {}).get('multiple_face_opt', {}).get('order', 'scale')
	dilation = args.get('module_config', {}).get('multiple_face_opt', {}).get('mask dilation', 4)
	limit = args.get('module_config', {}).get('multiple_face_opt', {}).get('limit', -1)

	print('processing multiple face')
	print(f'config : order={order}, dilation={dilation}, limit={limit}')

	if limit < 0:
		limit = len(multiple_face)
		if limit == 0:
			return img

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(img, 'face')

	org_size = img.size
	print(f'size {org_size} boxes {len(boxes)} order {order}')

	# sort
	candidate = []
	for box, logit, phrase in zip(boxes, logits, phrases):
		x1, y1, x2, y2 = box
		if order == 'left':
			value = x1 + (x2 - x1) // 2
			print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0])
		elif order == 'right':
			value = x1 + (x2 - x1) // 2
			print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		elif order == 'size':
			value = (x2 - x1) * (y2 - y1)
			print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		else:
			value = float(logit)
			print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)

	for idx, (size, box, logit, phrase) in enumerate(candidate):
		if idx == limit:
			break
		print('render', phrase, float(logit), size)
		x1, y1, x2, y2 = box
		x1 = int(x1) - dilation
		y1 = int(y1) - dilation
		x2 = int(x2) + dilation
		y2 = int(y2) + dilation

		face_mask = Image.new('L', img.size, color=0)
		dr = ImageDraw.Draw(face_mask, 'L')
		dr.rectangle((x1, y1, x2, y2), fill=255)

		# face_mask = dinosam.sam_predict_box(img, box)

		options = dict(mask=face_mask)

		if idx < len(multiple_face):
			prompt = multiple_face[idx].get('prompt')
			current_prompt = args.get('current_prompt', '')
			if prompt is not None and prompt.find('#!org!#') >= 0:
				multiple_face[idx]['prompt'] = multiple_face[idx]['prompt'].replace('#!org!#', current_prompt)
				print('prompt for face', multiple_face[idx]['prompt'])
			options.update(multiple_face[idx])
		img = process.process_img2img(p, img, options=options)

	return img
