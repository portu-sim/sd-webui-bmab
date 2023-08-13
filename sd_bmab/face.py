from PIL import ImageEnhance

from sd_bmab import dinosam, util


def process_face_lighting(args, p, img):
	if args['face_lighting'] == 0:
		return img

	'''    
		p.extra_generation_params['BMAB face lighting'] = args['face_lighting']
		strength = 1 + args['face_lighting']
		enhancer = ImageEnhance.Brightness(bgimg)
		processed = enhancer.enhance(strength)
		face_mask = sam('face', bgimg)
		bgimg.paste(processed, mask=face_mask)
		bgimg = process_face_detailing(p, bgimg)
	'''

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
