
# Auto Masking

## Dino Detect

* Img2Img 인물 사진을 추가한다.
* Just resize 선택
* Resize를 하면서 크기를 키워야 좋은 결과를 얻을 수 있다.   
예제의 경우는 512x768 --> 800x1200 으로 resize한 것이다.
* Denoising Strength는 0.4~0.6 정도를 설정한다.   
모델별로 다르지만, 배경까지 심하게 바뀌는 경우 숫자를 줄인다.
  
<p>
<img src="https://i.ibb.co/VpRpN5m/2023-08-16-1-47-31.png">
</p>

* BMAB를 Enable 한다.
* 반드시 Process before img2img를 체크한다.
* Dino detect enabled 체크하고 prompt에 person, 1girl, human...뭐든..

<p>
<img src="https://i.ibb.co/k525fb6/2023-08-16-1-47-45.png">
</p>

* 최종적으로 좋은 결과를 내기 위해서 합쳐진 이미지를 프롬프트로 잘 표현해야 한다.
* 프롬프트가 없거나 적당히 적으면 원하는 이미지에서 멀어지기 때문에 프롬프트가 매우 중요하다.

<p>
<img src="https://i.ibb.co/V0yKJdS/00015-188939271.png">
</p>

