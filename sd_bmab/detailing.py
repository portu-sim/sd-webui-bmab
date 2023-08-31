import math
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from modules import devices
from modules import shared
from sd_bmab import dinosam, util, process, constants
from sd_bmab.util import debug_print


def timecalc(func):
	def wrapper(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		end = time.time()
		debug_print(f'{end - start:.2f} sec')
		return ret
	return wrapper


def get_mask(img, prompt):
	boxes, logits, phrases = dinosam.dino_predict(img, prompt)
	sam_mask = dinosam.sam_predict_box(img, boxes[0])
	return sam_mask


def process_face_detailing(image, s, p, a):
	face_detailing_opt = a.get('module_config', {}).get('face_detailing_opt', {})
	detection_model = face_detailing_opt.get('detection_model', 'GroundingDINO')
	if a['face_detailing_enabled'] or a.get('module_config', {}).get('multiple_face'):
		if detection_model == 'GroundingDINO':
			return process_face_detailing_inner(image, s, p, a)
		else:
			return process_face_detailing_inner_using_yolo(image, s, p, a)
	return image


@timecalc
def process_face_detailing_inner(image, s, p, a):
	face_detailing_opt = a.get('module_config', {}).get('face_detailing_opt', {})
	face_detailing = dict(a.get('module_config', {}).get('face_detailing', {}))
	override_parameter = face_detailing_opt.get('override_parameter', False)
	dilation = face_detailing_opt.get('dilation', 4)
	box_threshold = face_detailing_opt.get('box_threshold', 0.35)
	order = face_detailing_opt.get('order_by', 'Score')
	limit = face_detailing_opt.get('limit', 1)
	sampler = face_detailing_opt.get('sampler', constants.sampler_default)
	max_element = shared.opts.bmab_max_detailing_element

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(image, 'people . face .', box_threahold=box_threshold)
	# debug_print(float(logits))
	debug_print(phrases)

	org_size = image.size
	debug_print('size', org_size)

	face_config = {
		'denoising_strength': face_detailing['denoising_strength'],
		'inpaint_full_res': face_detailing['inpaint_full_res'],
		'inpaint_full_res_padding': face_detailing['inpaint_full_res_padding'],
	}

	if override_parameter:
		face_config = dict(face_detailing)
	else:
		if shared.opts.bmab_keep_original_setting:
			face_config['width'] = image.width
			face_config['height'] = image.height
		else:
			face_config['width'] = p.width
			face_config['height'] = p.height

	if sampler != constants.sampler_default:
		face_config['sampler_name'] = sampler

	p.extra_generation_params['BMAB_face_option'] = util.dict_to_str(face_detailing_opt)
	p.extra_generation_params['BMAB_face_parameter'] = util.dict_to_str(face_config)

	candidate = []
	for box, logit, phrase in zip(boxes, logits, phrases):
		if phrase != 'face':
			continue
		x1, y1, x2, y2 = box
		if order == 'Left':
			value = x1 + (x2 - x1) // 2
			debug_print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0])
		elif order == 'Right':
			value = x1 + (x2 - x1) // 2
			debug_print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		elif order == 'Size':
			value = (x2 - x1) * (y2 - y1)
			debug_print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		else:
			value = float(logit)
			debug_print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)

	shared.state.job_count += min(limit, len(candidate))

	for idx, (size, box, logit, phrase) in enumerate(candidate):
		if phrase != 'face':
			continue

		if limit != 0 and idx >= limit:
			debug_print(f'Over limit {limit}')
			break

		if max_element != 0 and idx >= max_element:
			debug_print(f'Over limit MAX Element {max_element}')
			break

		prompt = face_detailing_opt.get(f'prompt{idx}')
		if prompt is not None:
			if prompt.find('#!org!#') >= 0:
				current_prompt = a.get('current_prompt', p.prompt)
				face_config['prompt'] = prompt.replace('#!org!#', current_prompt)
				debug_print('prompt for face', face_config['prompt'])
			elif prompt != '':
				face_config['prompt'] = prompt
			else:
				face_config['prompt'] = p.all_prompts[s.index]

		ne_prompt = face_detailing_opt.get(f'negative_prompt{idx}')
		if ne_prompt is not None and ne_prompt != '':
			face_config['negative_prompt'] = ne_prompt
		else:
			face_config['negative_prompt'] = p.all_negative_prompts[s.index]

		debug_print('render', phrase, float(logit))
		debug_print('delation', dilation)

		face_mask = Image.new('L', image.size, color=0)
		dr = ImageDraw.Draw(face_mask, 'L')
		dr.rectangle(box, fill=255)
		face_mask = util.dilate_mask(face_mask, dilation)

		seed, subseed = util.get_seeds(s, p, a)
		options = dict(mask=face_mask, seed=seed, subseed=subseed, **face_config)
		img2img_imgage = process.process_img2img(p, image, options=options)

		x1, y1, x2, y2 = util.fix_box_size(box)
		face_mask = Image.new('L', image.size, color=0)
		dr = ImageDraw.Draw(face_mask, 'L')
		dr.rectangle((x1, y1, x2, y2), fill=255)
		blur = ImageFilter.GaussianBlur(3)
		mask = face_mask.filter(blur)

		image.paste(img2img_imgage, mask=mask)
	devices.torch_gc()
	return image


@timecalc
def process_face_detailing_inner_using_yolo(image, s, p, a):
	face_detailing_opt = a.get('module_config', {}).get('face_detailing_opt', {})
	face_detailing = dict(a.get('module_config', {}).get('face_detailing', {}))
	override_parameter = face_detailing_opt.get('override_parameter', False)
	dilation = face_detailing_opt.get('dilation', 4)
	confidence = face_detailing_opt.get('box_threshold', 0.3)
	order = face_detailing_opt.get('order_by', 'Score')
	limit = face_detailing_opt.get('limit', 1)
	sampler = face_detailing_opt.get('sampler', constants.sampler_default)
	max_element = shared.opts.bmab_max_detailing_element

	org_size = image.size
	debug_print('size', org_size)

	face_config = {
		'denoising_strength': face_detailing['denoising_strength'],
		'inpaint_full_res': face_detailing['inpaint_full_res'],
		'inpaint_full_res_padding': face_detailing['inpaint_full_res_padding'],
	}

	if override_parameter:
		face_config = dict(face_detailing)
	else:
		if shared.opts.bmab_keep_original_setting:
			face_config['width'] = image.width
			face_config['height'] = image.height
		else:
			face_config['width'] = p.width
			face_config['height'] = p.height

	if sampler != constants.sampler_default:
		face_config['sampler_name'] = sampler

	p.extra_generation_params['BMAB_face_option'] = util.dict_to_str(face_detailing_opt)
	p.extra_generation_params['BMAB_face_parameter'] = util.dict_to_str(face_config)

	boxes = util.ultralytics_predict(image, confidence)

	candidate = []
	for box in boxes:
		x1, y1, x2, y2 = box
		if order == 'Left':
			value = x1 + (x2 - x1) // 2
			debug_print('detected', value)
			candidate.append((value, box))
			candidate = sorted(candidate, key=lambda c: c[0])
		elif order == 'Right':
			value = x1 + (x2 - x1) // 2
			debug_print('detected', value)
			candidate.append((value, box))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		elif order == 'Size':
			value = (x2 - x1) * (y2 - y1)
			debug_print('detected', value)
			candidate.append((value, box))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		else:
			value = 0
			debug_print('detected', value)
			candidate.append((value, box))

	shared.state.job_count += min(limit, len(candidate))

	for idx, (size, box) in enumerate(candidate):
		if limit != 0 and idx >= limit:
			debug_print(f'Over limit {limit}')
			break

		if max_element != 0 and idx >= max_element:
			debug_print(f'Over limit MAX Element {max_element}')
			break

		prompt = face_detailing_opt.get(f'prompt{idx}')
		if prompt is not None:
			if prompt.find('#!org!#') >= 0:
				current_prompt = a.get('current_prompt', p.prompt)
				face_config['prompt'] = prompt.replace('#!org!#', current_prompt)
				debug_print('prompt for face', face_config['prompt'])
			elif prompt != '':
				face_config['prompt'] = prompt
			else:
				face_config['prompt'] = p.all_prompts[s.index]

		ne_prompt = face_detailing_opt.get(f'negative_prompt{idx}')
		if ne_prompt is not None and ne_prompt != '':
			face_config['negative_prompt'] = ne_prompt
		else:
			face_config['negative_prompt'] = p.all_negative_prompts[s.index]

		face_mask = Image.new('L', image.size, color=0)
		dr = ImageDraw.Draw(face_mask, 'L')
		dr.rectangle(box, fill=255)
		face_mask = util.dilate_mask(face_mask, dilation)

		seed, subseed = util.get_seeds(s, p, a)
		options = dict(mask=face_mask, seed=seed, subseed=subseed, **face_config)
		image = process.process_img2img(p, image, options=options)

	devices.torch_gc()
	return image


def process_hand_detailing(image, s, p, a):
	if a['hand_detailing_enabled']:
		return process_hand_detailing_inner(image, s, p, a)
	return image


@timecalc
def process_hand_detailing_inner(image, s, p, args):
	hand_detailing = dict(args.get('module_config', {}).get('hand_detailing', {}))
	hand_detailing_opt = args.get('module_config', {}).get('hand_detailing_opt', {})
	detailing_method = hand_detailing_opt.get('detailing_method', '')

	dinosam.dino_init()

	if detailing_method == 'subframe':
		return process_hand_detailing_subframe(image, s, p, args)
	elif detailing_method == 'at once':
		mask = Image.new('L', image.size, 0)
		dr = ImageDraw.Draw(mask, 'L')
		boxes, logits, phrases = dinosam.dino_predict(image, 'person . hand')
		for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
			if phrase == 'hand':
				b = util.fix_box_size(box)
				dr.rectangle(b, fill=255)
		options = dict(mask=mask)
		options.update(hand_detailing)
		shared.state.job_count += 1
		image = process.process_img2img(p, image, options=options)
	elif detailing_method == 'each hand' or detailing_method == 'inpaint each hand':
		boxes, logits, phrases = dinosam.dino_predict(image, 'person . hand')
		for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
			debug_print(float(logit), phrase)
			if phrase == 'hand':
				x1, y1, x2, y2 = tuple(int(x) for x in box)

				width = x2 - x1
				height = y2 - y1

				mbox = (int(x1 - width), int(y1 - height), int(x2 + width), int(y2 + height))
				mbox = util.fix_box_size(mbox)
				debug_print(mbox)

				hbox = (width, height, width * 2, height * 2)
				cropped_hand = image.crop(box=mbox)
				cropped_hand_mask = Image.new('L', cropped_hand.size, 0)
				dr = ImageDraw.Draw(cropped_hand_mask, 'L')
				dr.rectangle(hbox, fill=255)

				options = dict(seed=-1)
				scale = hand_detailing_opt.get('scale', -1)
				if scale < 1:
					normalize = hand_detailing_opt.get('normalize', 768)
					if width > height:
						scale = normalize / cropped_hand.width
					else:
						scale = normalize / cropped_hand.height
				if detailing_method == 'inpaint each hand':
					options['mask'] = cropped_hand_mask

				options.update(hand_detailing)
				w, h = util.fix_size_by_scale(cropped_hand.width, cropped_hand.height, scale)
				options['width'] = w
				options['height'] = h
				debug_print(f'scale {scale} width {w} height {h}')
				shared.state.job_count += 1
				img2img_result = process.process_img2img(p, cropped_hand, options=options)
				img2img_result = img2img_result.resize(cropped_hand.size, resample=Image.LANCZOS)

				debug_print('resize to', img2img_result.size, cropped_hand_mask.size)
				blur = ImageFilter.GaussianBlur(3)
				cropped_hand_mask = cropped_hand_mask.filter(blur)
				image.paste(img2img_result, (mbox[0], mbox[1]), mask=cropped_hand_mask)
	else:
		debug_print('no such method')
		return image

	return image


def process_hand_detailing_subframe(image, s, p, args):
	hand_detailing = dict(args.get('module_config', {}).get('hand_detailing', {}))
	hand_detailing_opt = args.get('module_config', {}).get('hand_detailing_opt', {})
	dilation = hand_detailing_opt.get('dilation', 0.1)
	debug_print('dilation', dilation)

	box_threshold = hand_detailing_opt.get('box_threshold', 0.3)
	boxes, masks = get_subframe(image, dilation, box_threshold=box_threshold)
	if not boxes:
		return image

	if not hasattr(args, 'hand_mask_image'):
		c1 = image.copy()
		for box, mask in zip(boxes, masks):
			box = util.fix_box_by_scale(box, dilation)
			draw = ImageDraw.Draw(c1, 'RGBA')
			draw.rectangle(box, outline=(0, 255, 0, 255), fill=(0, 255, 0, 50), width=3)
			c2 = image.copy()
			draw = ImageDraw.Draw(c2, 'RGBA')
			draw.rectangle(box, outline=(255, 0, 0, 255), fill=(255, 0, 0, 50), width=3)
			c1.paste(c2, mask=mask)
		s.extra_image.append(c1)
		p.hand_mask_image = c1

	for box, mask in zip(boxes, masks):
		box = util.fix_box_by_scale(box, dilation)
		box = util.fix_box_size(box)
		box = util.fix_box_limit(box, image.size)
		x1, y1, x2, y2 = box

		cropped = image.crop(box=box)
		cropped_mask = mask.crop(box=box)

		scale = hand_detailing_opt.get('scale', 2)

		options = dict(mask=cropped_mask, seed=-1)
		hand_detailing = dict(args.get('module_config', {}).get('hand_detailing', {}))
		options.update(hand_detailing)
		w, h = util.fix_size_by_scale(cropped.width, cropped.height, scale)
		options['width'] = w
		options['height'] = h
		debug_print(f'Scale x{scale} ({cropped.width},{cropped.height}) -> ({w},{h})')

		if hand_detailing_opt.get('block_overscaled_image', True):
			area_org = args.get('max_area', image.width * image.height)
			area_scaled = w * h
			if area_scaled > area_org:
				debug_print(f'It is too large to process.')
				auto_upscale = hand_detailing_opt.get('auto_upscale', True)
				if not auto_upscale:
					return image
				scale = math.sqrt(area_org / (cropped.width * cropped.height))
				w, h = util.fix_size_by_scale(cropped.width, cropped.height, scale)
				options['width'] = w
				options['height'] = h
				debug_print(f'Auto Scale x{scale} ({cropped.width},{cropped.height}) -> ({w},{h})')
				if scale < 1.2:
					debug_print(f'Scale {scale} has no effect. skip!!!!!')
					return image
		shared.state.job_count += 1
		img2img_result = process.process_img2img(p, cropped, options=options)
		img2img_result = img2img_result.resize((cropped.width, cropped.height), resample=Image.LANCZOS)
		blur = ImageFilter.GaussianBlur(3)
		cropped_mask = cropped_mask.filter(blur)
		image.paste(img2img_result, (x1, y1), mask=cropped_mask)
		devices.torch_gc()

	return image


class Obj(object):
	name = None

	def __init__(self, xyxy) -> None:
		super().__init__()
		self.parent = None
		self.xyxy = xyxy
		self.objects = []
		self.inbox = xyxy

	def is_in(self, obj) -> bool:
		x1, y1, x2, y2 = self.inbox
		mx1, my1, mx2, my2 = obj.xyxy
		return mx1 <= x1 <= mx2 and mx1 <= x2 <= mx2 and my1 <= y1 <= my2 and my1 <= y2 <= my2

	def append(self, obj):
		obj.parent = self
		for ch in self.objects:
			if obj.is_in(ch):
				obj.parent = ch
				break
		self.objects.append(obj)

	def is_valid(self):
		return True

	def size(self):
		x1, y1, x2, y2 = self.xyxy
		return (x2 - x1) * (y2 - y1)

	def put(self, mask):
		for xg in self.objects:
			if not xg.is_valid():
				continue
			if xg.name == 'hand':
				dr = ImageDraw.Draw(mask, 'L')
				dr.rectangle(xg.xyxy, fill=255)

	def get_box(self):
		if not self.objects:
			return self.xyxy

		x1, y1, x2, y2 = self.xyxy
		ret = [x2, y2, x1, y1]
		for xg in self.objects:
			if not xg.is_valid():
				continue
			x = xg.xyxy
			ret[0] = x[0] if x[0] < ret[0] else ret[0]
			ret[1] = x[1] if x[1] < ret[1] else ret[1]
			ret[2] = x[2] if x[2] > ret[2] else ret[2]
			ret[3] = x[3] if x[3] > ret[3] else ret[3]

		return x1, y1, x2, ret[3]

	def log(self):
		debug_print(self.name, self.xyxy)
		for x in self.objects:
			x.log()


class Person(Obj):
	name = 'person'

	def __init__(self, xyxy, dilation) -> None:
		super().__init__(xyxy)
		self.inbox = util.fix_box_by_scale(xyxy, dilation)

	def is_valid(self):
		face = False
		hand = False
		for xg in self.objects:
			if xg.name == 'face':
				face = True
			if xg.name == 'hand':
				hand = True
		return face and hand

	def cleanup(self):
		debug_print([xg.name for xg in self.objects])
		nw = []
		for xg in self.objects:
			if xg.name == 'person':
				if len(self.objects) == 1 and xg.is_valid():
					self.xyxy = xg.xyxy
					self.objects = xg.objects
					return
				else:
					self.objects.extend(xg.objects)
			else:
				nw.append(xg)
		self.objects = nw


class Head(Obj):
	name = 'head'


class Face(Obj):
	name = 'face'


class Hand(Obj):
	name = 'hand'


def get_subframe(pilimg, dilation, box_threshold=0.30, text_threshold=0.20):
	text_prompt = "person . head . face . hand ."
	debug_print('threshold', box_threshold)
	boxes, logits, phrases = dinosam.dino_predict(pilimg, text_prompt, box_threshold, text_threshold)

	people = []

	def find_person(o):
		for person in people:
			if o.is_in(person):
				return person
		return None

	for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
		if phrase == 'person':
			p = Person(tuple(int(x) for x in box), dilation)
			parent = find_person(p)
			if parent:
				parent.append(p)
			else:
				people.append(p)
	people = sorted(people, key=lambda c: c.size(), reverse=True)

	for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
		debug_print(float(logit), phrase)
		bb = tuple(int(x) for x in box)

		if phrase == 'head':
			o = Head(bb)
			parent = find_person(o)
			if parent:
				parent.append(o)
		elif phrase == 'face' or phrase == 'head face':
			o = Face(bb)
			parent = find_person(o)
			if parent:
				parent.append(o)
		elif phrase == 'hand':
			o = Hand(bb)
			parent = find_person(o)
			if parent:
				parent.append(o)

	for person in people:
		person.cleanup()

	boxes = []
	masks = []
	for person in people:
		if person.is_valid():
			mask = Image.new('L', pilimg.size, color=0)
			person.log()
			person.put(mask)
			boxes.append(person.get_box())
			masks.append(mask)
	return boxes, masks


def process_person_detailing(image, s, p, a):
	if a['person_detailing_enabled']:
		return process_person_detailing_inner(image, s, p, a)
	return image


def dilate_mask(mask, value):
	return mask.filter(ImageFilter.MaxFilter(value))


@timecalc
def process_person_detailing_inner(image, s, p, a):
	person_detailing_opt = a.get('module_config', {}).get('person_detailing_opt', {})
	dilation = person_detailing_opt.get('dilation', 3)
	area_ratio = person_detailing_opt.get('area_ratio', 0.1)
	limit = person_detailing_opt.get('limit', 1)
	max_element = shared.opts.bmab_max_detailing_element

	p.extra_generation_params['BMAB_person_option'] = util.dict_to_str(person_detailing_opt)

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(image, 'people')
	debug_print(phrases)

	org_size = image.size
	debug_print('size', org_size)

	i2i_config = dict(a.get('module_config', {}).get('person_detailing', {}))
	debug_print(f'Max element {max_element}')

	shared.state.job_count += min(limit, len(boxes))

	for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
		if limit != 0 and idx >= limit:
			debug_print(f'Over limit {limit}')
			break

		if max_element != 0 and idx >= max_element:
			debug_print(f'Over limit MAX Element {max_element}')
			break

		debug_print('render', phrase, float(logit))
		box2 = util.fix_box_size(box)
		x1, y1, x2, y2 = box2

		mask = dinosam.sam_predict_box(image, box)
		mask = util.dilate_mask(mask, dilation)
		cropped_mask = mask.crop(box=box).convert('L')
		cropped = image.crop(box=box)

		scale = person_detailing_opt.get('scale', 4)

		area_person = cropped.width * cropped.height
		area_image = image.width * image.height
		ratio = area_person / area_image
		debug_print(f'Ratio {ratio}')
		if ratio >= area_ratio:
			debug_print(f'Person is too big to process. {ratio} >= {area_ratio}.')
			return image
		p.extra_generation_params['BMAB person ratio'] = '%.3f' % ratio

		w = cropped.width * scale
		h = cropped.height * scale
		debug_print(f'Trying x{scale} ({cropped.width},{cropped.height}) -> ({w},{h})')

		if person_detailing_opt.get('block_overscaled_image', True):
			area_org = a.get('max_area', image.width * image.height)
			area_scaled = w * h
			if area_scaled > area_org:
				debug_print(f'It is too large to process.')
				auto_upscale = person_detailing_opt.get('auto_upscale', True)
				if not auto_upscale:
					return image
				scale = math.sqrt(area_org / (cropped.width * cropped.height))
				w, h = util.fix_size_by_scale(cropped.width, cropped.height, scale)
				debug_print(f'Auto Scale x{scale} ({cropped.width},{cropped.height}) -> ({w},{h})')
				if scale < 1.2:
					debug_print(f'Scale {scale} has no effect. skip!!!!!')
					return image

		options = dict(mask=cropped_mask, **i2i_config)
		options['width'] = w
		options['height'] = h
		options['inpaint_full_res'] = 1
		options['inpaint_full_res'] = 32

		img2img_result = process.process_img2img(p, cropped, options=options)
		img2img_result = img2img_result.resize(cropped.size, resample=Image.LANCZOS)
		blur = ImageFilter.GaussianBlur(3)
		cropped_mask = cropped_mask.filter(blur)
		image.paste(img2img_result, (x1, y1), mask=cropped_mask)

	devices.torch_gc()
	return image






