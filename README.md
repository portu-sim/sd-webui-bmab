
# BMAB

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
        <td>
            <img src="https://i.ibb.co/jVPtgnM/00074-4133194501.jpg">
        </td>
        <td>
            <img src="https://i.ibb.co/wL1Xm2P/00079-1737359342.jpg">
        </td>
    </tr>
    <tr>
        <td>
            IC-Light(Left), Face Detailing,
            </td>
        <td>
            IC-Light(Left), Face Detailing,
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <img src="https://i.ibb.co/B2QJ0Tz/00126-1953699647.jpg">
        </td>
    </tr>
    <tr>
        <td colspan="2">
            Face Detailing, Resample, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1), cnIpAdapter(0.6, 0, 0.3)
        </td>
    </tr>
    <tr>
        <td>
            <img src="https://i.ibb.co/xD1fxg1/00755-233390832.jpg">
        </td>
        <td>
            <img src="https://i.ibb.co/TTm7CdN/00774-2729955256.jpg">
        </td>
    </tr>
    <tr>
        <td>
            Face Detailing, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1), cnIpAdapter(0.6, 0, 0.3)
            </td>
        <td>
            Face Detailing, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1), cnIpAdapter(0.6, 0, 0.3)
        </td>
    </tr>
    <tr>
        <td>
            <img src="https://i.ibb.co/yBT2YX5/00548-4054764802.jpg">
        </td>
        <td>
            <img src="https://i.ibb.co/RQtVS2g/00581-3667453446.jpg">
        </td>
    </tr>
    <tr>
        <td>
            Face Detailing, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1), cnIpAdapter(0.6, 0, 0.3)
            </td>
        <td>
            Face Detailing, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1), cnIpAdapter(0.6, 0, 0.3)
        </td>
    </tr>
    <tr>
        <td>
            <img src="https://i.ibb.co/hM8pvV2/00612-2685660966.jpg">
        </td>
        <td>
            <img src="https://i.ibb.co/H2CD8kX/00672-3470647356.jpg">
        </td>
    </tr>
    <tr>
        <td>
            Face Detailing, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1), cnIpAdapter(0.6, 0, 0.3)
            </td>
        <td>
            Face Detailing, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1), cnIpAdapter(0.6, 0, 0.3)
        </td>
    </tr>
    <tr>
        <td>
            <img src="https://i.ibb.co/WvHHKc7/00111-2484939723.jpg">
            </td>
        <td>
            <img src="https://i.ibb.co/px4YXDM/00199-2019853980.jpg">
        </td>
    </tr>
    <tr>
        <td>
            Face Detailing, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1)
            </td>
        <td>
            Face Detailing, cnNoise(0.4, 0, 0.4), cnPose(0.3, 0, 0.1)
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <img src="https://i.ibb.co/ns1Kn04/00460-759278328.jpg">
        </td>
    </tr>
    <tr>
        <td colspan="2">
            Face Detailing, ControlNet Noise (0.4, 0, 0.4),
        </td>
    </tr>
    <tr>
        <td>
            <img src="https://i.ibb.co/zsDs4bq/00450-3195179381.jpg">
        </td>
        <td>
            <img src="https://i.ibb.co/D9tz1NY/00180-3383798469.png">
        </td>
    </tr>
    <tr>
        <td>
            Resize intermediate (inpaint+lama, Bottom, 0.75, 0.6),<br>
            Face Detailing, ControlNet Noise (0.7, 0, 0.6),<br>
            Noise Alpha (0.1)
        </td>
        <td>
            Resize intermediate (Center, 0.5, 0.6),<br>
            Face Detailing, ControlNet Noise (0.4, 0, 0.4)<br>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <img src="https://i.ibb.co/P6477Vg/resize-00101-2353183853.png">
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <img src="https://i.ibb.co/3vsBTFZ/resize-00183-1413773744.png">
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <img src="https://i.ibb.co/tcYzHP1/resize-00226-4176028607.png">
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <img src="https://i.ibb.co/r6G1cwy/resize-00340-4033828371.png">
        </td>
    </tr>
    <tr>
        <td>
            <img src="https://i.ibb.co/PmPJtVb/resize-00718-3635306692.png">
        </td>
        <td>
            <img src="https://i.ibb.co/Bq2PFxc/resize-00793-3980284595.png">
        </td>
    </tr>
    <tr>
        <td>
            <img src="https://i.ibb.co/ZMNC1Cm/00518-1067577565.jpg">
        </td>
        <td>
            <img src="https://i.ibb.co/JtjGrMX/00126-496754363.jpg">
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <img src="https://i.ibb.co/Lnh4Kpm/resize-00824-738395988.png">
        </td>
    </tr>
    <tr>
        <td colspan="2">
            Resize intermediate (Bottom, 0.5, 0.6), Face Detailing, ControlNet Noise (0.4, 0, 0.4)
        </td>
    </tr>
</table>
