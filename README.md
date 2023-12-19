
# BMAB - SDNEXT

trying to add extra-noise and A1111's RNG module to SDNEXT through BMAP extention

WIP
- Add A1111 RNG module
- Add noisy latent to ExtraNoiseParams for callback
- Add extra noise callback
- Add extra noise param for img2img operations
- 

BMAB는 Stable Diffusion WebUI의 확장 기능으로, 생성된 이미지를 설정에 따라 후처리하는 기능을 가지고 있습니다.   
필요에 따라 인물, 얼굴, 손을 찾아 다시 그리거나, Resize, Resample, 노이즈 추가 등의 기능을 수행할 수 있으며,   
두 장의 이미지를 합성하거나, Upscale의 기능을 수행 할 수 있습니다.

BMAB is an extension of Stable Diffusion WebUI and has the function of post-processing the generated image according to settings.
If necessary, you can find and redraw people, faces, and hands, or perform functions such as resize, resample, and add noise.
You can composite two images or perform the Upscale function.

<a href="./docs/kr/manual.md">Manual (KR)</a>


## Example

<table>
<tr>
<td colspan="2"><img src="https://i.ibb.co/ns1Kn04/00460-759278328.jpg">
Face Detailing, ControlNet Noise (0.4, 0, 0.4),
</td>
</tr>
<tr>
<td colspan="2"><img src="https://i.ibb.co/zsDs4bq/00450-3195179381.jpg">
Resize intermediate (inpaint+lama, Bottom, 0.75, 0.6), Face Detailing, ControlNet Noise (0.7, 0, 0.6), Noise Alpha (0.1)
</td>
</tr>
<tr>
<td colspan="2">
<img src="https://i.ibb.co/D9tz1NY/00180-3383798469.png">
Resize intermediate (Center, 0.5, 0.6), Face Detailing, ControlNet Noise (0.4, 0, 0.4)
</td>
</tr>
<tr>
<td colspan="2"><img src="https://i.ibb.co/P6477Vg/resize-00101-2353183853.png">
</td>
</tr>
<tr><td colspan="2"><img src="https://i.ibb.co/3vsBTFZ/resize-00183-1413773744.png"></td></tr>
<tr><td colspan="2"><img src="https://i.ibb.co/tcYzHP1/resize-00226-4176028607.png"></td></tr>
<tr><td colspan="2"><img src="https://i.ibb.co/r6G1cwy/resize-00340-4033828371.png"></td></tr>
<tr>
<td><img src="https://i.ibb.co/PmPJtVb/resize-00718-3635306692.png"></td>
<td><img src="https://i.ibb.co/Bq2PFxc/resize-00793-3980284595.png"></td>
</tr>
<tr>
<td><img src="https://i.ibb.co/ZMNC1Cm/00518-1067577565.jpg"></td>
<td><img src="https://i.ibb.co/JtjGrMX/00126-496754363.jpg"></td>
</tr>
<tr>
<td colspan="2"><img src="https://i.ibb.co/Lnh4Kpm/resize-00824-738395988.png">
Resize intermediate (Bottom, 0.5, 0.6), Face Detailing, ControlNet Noise (0.4, 0, 0.4)
</td>
</tr>
</table>
