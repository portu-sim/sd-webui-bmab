from PIL import Image
from PIL import ImageEnhance

from modules.processing import process_images, StableDiffusionProcessingImg2Img
from scripts.dinosam import dino_init, dino_predict, sam_init, sam_predict
from scripts.util import sam


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

    dino_init()
    boxes, logits, phrases = dino_predict(img, 'face')
    print(float(logits))
    print(phrases)

    org_size = img.size
    print('size', org_size)

    largest = (0, None)
    for box in boxes:
        x1, y1, x2, y2 = box
        size = (x2 - x1) * (y2 - y1)
        if size > largest[0]:
            largest = (size, box)

    if largest[0] == 0:
        return img

    x1, y1, x2, y2 = largest[1]

    mask = Image.new('L', img.size, 0)
    box_mask = Image.new('L', (int(x2 - x1), int(y2 - y1)), 255)
    mask.paste(box_mask, (int(x1), int(y1)))

    enhancer = ImageEnhance.Brightness(img)
    bgimg = enhancer.enhance(0.8)

    face_mask = sam('face', img)
    img.paste(bgimg, mask=face_mask)

    options = dict(mask=face_mask)
    return process_face_detailing(p, img, options=options)


def process_face_detailing(p, img, options=None):

    i2i_param = dict(
        init_images=[img],
        resize_mode=0,
        denoising_strength=0.4,
        mask=None,
        mask_blur=4,
        inpainting_fill=1,
        inpaint_full_res=True,
        inpaint_full_res_padding=32,
        inpainting_mask_invert=0,
        initial_noise_multiplier=1.0,
        sd_model=p.sd_model,
        outpath_samples=p.outpath_samples,
        outpath_grids=p.outpath_grids,
        prompt=p.prompt,
        negative_prompt=p.negative_prompt,
        styles=p.styles,
        seed=p.seed,
        subseed=p.subseed,
        subseed_strength=p.subseed_strength,
        seed_resize_from_h=p.seed_resize_from_h,
        seed_resize_from_w=p.seed_resize_from_w,
        sampler_name=p.sampler_name,
        batch_size=1,
        n_iter=1,
        steps=20, # p.steps,
        cfg_scale=7,
        width=img.width,
        height=img.height,
        restore_faces=False,
        tiling=p.tiling,
        extra_generation_params=p.extra_generation_params,
        do_not_save_samples=True,
        do_not_save_grid=True,
        override_settings={},
    )
    if options is not None:
        i2i_param.update(options)
    img2img = StableDiffusionProcessingImg2Img(**i2i_param)
    img2img.scripts = None
    img2img.script_args = None

    processed = process_images(img2img)

    return processed.images[0]
