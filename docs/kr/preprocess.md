
# Preprocess

## Context

BMAB에서 사용할 Checkpoint와 VAE를 지정합니다.   
특정 기능들은 자체 Checkpoint와 VAE를 설정할 수 있습니다.   
한 번 변경된 Checkpoint는 그 이후 프로세스들이 계속 사용합니다.

<img src="https://i.ibb.co/VTP5ddx/2023-11-12-3-48-54.png">

#### txt2img noise multiplier for hires.fix

hires.fix 단계에서 noise를 추가 할 수 있다.

#### txt2img extra noise multiplier for hires.fix (EXPERIMENTAL)

hires.fix 단계에서 추가적인 noise를 더 할 수 있다.

#### Hires.fix filter before upscaler

Hires.fix 단계 중 upscaler 전에 filter를 적용할 수 있다.

#### Hires.fix filter after upscaler

Hires.fix 단계 중 upscaler 후에 filter를 적용할 수 있다.


## Resample (EXPERIMENTAL)

Self resampling 기능입니다. txt2img -> hres.fix를 통해 생성된 이미지를 다시 txt2img -> hires.fix 과정을 수행하면서   
ControlNet Tile Resample을 수행합니다. 아래와 같은 경우 사용할 수 있습니다.

* 두 모델간에 결과물 차이가 큰 경우
* 두 모델간에 인물 비율이 차이나는 경우
* 두 모델간 버전이 다른 경우 (SDXL, SD15)

<table>
<tr>
<td>txt2img->hires.fix</td>
<td>Resample + BMAB Basic</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/VxPfgN0/00153-3939130001-before-resample.png"></td>
<td><img src="https://i.ibb.co/XZ9gHHN/00154-3939130001.png"></td>
</tr>
</table>

<img src="https://i.ibb.co/5hWtbmZ/e822842f656d73757ee65713317f7ba9d947472d3fe94fc3ceffc72aee31064d.jpg">
BMAB resample image by [padapari](https://www.instagram.com/_padapari_/)

<br>
<br>
<br>
<br>


<img src="https://i.ibb.co/9hD81hd/resample.png">

#### Enable self resample (EXPERIMENTAL)

이 기능을 켜고 끌 수 있습니다.

#### Save image before processing

최초 txt2img -> hires.fix를 통해 생성된 이미지가 후 처리를 위해 BMAB로 입력되면,   
해당 이미지를 프로세싱하기 전에 저장합니다. 이미지 postfix로 "-before-resample"이 붙습니다.

#### Checkpoint

SD Checkpoint를 지정할 수 있습니다. 지정하지 않는다면 앞에서 설정된 Checkpoint를 사용합니다.   
프로세스가 완료되어도 원래 것으로 돌려놓지 않습니다.

#### SD VAE

SD VAE를 지정할 수 있습니다. 지정하지 않는다면 앞에서 설정된 VAE를 사용합니다.   
프로세스가 완료되어도 원래 것으로 돌려놓지 않습니다.

#### Resample method

Resample 방법을 선택할 수 있습니다.

txt2img-1pass : txt2img을 hires.fix 없이 동작시킨다.   
txt2img-2pass : txt2img를 hires.fix로 동작시킨다. 기본적으로 이미지를 출력할 때 hires.fix가 동작해야하만 한다.   
img2img-1pass : img2img로 동작시킨다.   

#### Resample filter

Resample이 완료되고 난 이후에 외부 filter 코드를 호출하여 이미지 변환을 추가적으로 수행할 수 있다.


#### Resample prompt

resampling 과정에서 사용할 prompt입니다. 비어있는 경우 main prompt와 동일하며,   
"#!org!#" 를 입력하면 main prompt를 대체합니다. "#!org!#" 이후에 추가로 prompt를 적을 수 있습니다.   
ex) #!org!#, soft light, some more keyword

#### Resample negative prompt

resampling 과정에서 사용할 prompt입니다. 비어있는 경우 main negative prompt와 동일합니다.

#### Sampling method

프로세스에 사용할 sampling method를 지정합니다. 지정하지 않는다면 이전 프로세스와 같은 sampler를 지정합니다.

#### Upsacler

hires.fix를 사용하는 경우에 지정하는 upscaler입니다.

#### Resample sampling steps

resample process 사용할 samping steps를 지정합니다.   
(권장 20)

#### Resample CFG scale

resample process 사용할 CFG scale 값을 지정합니다.
dynamic threshold는 지원하지 않습니다.

#### Resample denoising strength

resample process가 사용할 denoising strength를 지정합니다.   
(권장 0.4)

#### Resample strength

0에 가까운 값은 입력 이미지와 멀어지고, 1에 가까울 수록 원본 이미지와 유사합니다.

#### Resample begin

sampling 단계에 적용 시작점.

#### Resample end

sampling 단계 적용 종료 시점.








## Pretraining (EXPERIMENTAL)

Pretraining detailer입니다. ultralytics로 pretraining 모델을 적용하여 detection을 수행하고   
이를 기반으로 prompt, negative prompt를 적용하여 부분적으로 이미지를 더 자세하게 그릴 수 있습니다.

<img src="https://i.ibb.co/Qkx6rQK/pretraining.png"/>

#### Enable pretraining detailer (EXPERIMENTAL)

이 기능을 켜고 끌 수 있습니다.

#### Enable pretraining before hires.fix

pretraining detailer를 hires.fix 전에 수행하도록 한다.

#### Pretraining model

ultralytics 로 학습된 detection model (*.pt)를 지정할 수 있습니다.   
stable-diffusion-webui/models/BMAB에 해당 파일이 있어야 목록에 나타납니다.


#### Pretraining prompt

pretraining detailer process 과정에서 사용할 prompt입니다. 비어있는 경우 main prompt와 동일하며,   
"#!org!#" 를 입력하면 main prompt를 대체합니다. "#!org!#" 이후에 추가로 prompt를 적을 수 있습니다.
ex) #!org!#, soft light, some more keyword

#### Pretraining negative prompt

pretraining detailer process 과정에서 사용할 prompt입니다. 비어있는 경우 main negative prompt와 동일합니다.

#### Sampling method

프로세스에 사용할 sampling method를 지정합니다. 지정하지 않는다면 이전 프로세스와 같은 sampler를 지정합니다.


#### Pretraining sampling steps

resample process 사용할 samping steps를 지정합니다.   
(권장 20)

#### Pretraining CFG scale

resample process 사용할 CFG scale 값을 지정합니다.
dynamic threshold는 지원하지 않습니다.

#### Pretraining denoising strength

resample process가 사용할 denoising strength를 지정합니다.   
(권장 0.4)

#### Pretraining dilation

detection 된 사각형의 범위를 주어진 값 만큼 크기를 크게 합니다.

#### Pretraining box threshold

Detector의 검출 값을 결정합니다. 기본값 0.35보다 작으면 face가 아닐 것으로 제외합니다.   
ultralytics predict의 confidence 값입니다.



## Edge enhancemant

이미지 경계를 강화해 선명도를 증가시키거나 디테일을 증가시키는 기능입니다.   

**<span style="color: red">Upscaler가 Latent 계열인 경우 동작하지 않습니다. (R-ESRGAN, 4x-UltraSharp 추천)</span>**

<img src="https://i.ibb.co/4sjB1Lr/edge.png">

권장설정

* Edge low threshold : 50
* Edge high threshold : 200
* Edge strength : 0.5

<p>
<img src="https://i.ibb.co/Wsw2Wrh/00598-1745587019.png" width="40%">
<img src="https://i.ibb.co/z4nCW9Z/00600-1745587019.png" width="40%">
</p>

Enabled : CHECK!!   

Contrast : 1.2   
Brightness : 0.9   
Sharpeness : 1.5   

Enable edge enhancement : CHECK!!   
Edge low threshold : 50   
Edge high threshold : 200   
Edge strength : 0.5   




## Resize

txt2img -> hires.fix 의 중간 과정에서 동작합니다..   
만약 img2img에서 사용한다면, 프로세스 시작 전에 동작합니다.

그림 속 인물중 가장 신장이 큰 사람의 길이와 그림 높이의 비율이 설정값을 넘어가면 비율을 설정값로 맞추는 기능입니다.   
설정값이 0.90이고 인물의 전체 길이: 그림 높이의 비율이 0.95라고 한다면   
배경을 늘려서 인물의 비율이 0.90이 되도록 합니다.   
배경은 Alignment에서 지정한 방식에 따라 늘어납니다.

txt2img 수행하는 단계에서 hires.fix 하기 직전에 이미지를 변경합니다.   
이 과정은 변경된 이미지가 hires.fix 과정에서 매끄럽게 변하게 하기 위한 것입니다.   
**<span style="color: red">denoising strength는 0.6~0.7 정도를 사용하셔야 주변부 이미지 왜곡이 발생하지 않습니다.</span>**   
**<span style="color: red">Upscaler가 Latent 계열인 경우 동작하지 않습니다. (R-ESRGAN, 4x-UltraSharp 추천)</span>**

#### Method

Resize 하는 방식을 지정할 수 있습니다.

* Stretching : 단순히 이미지를 외곽부분을 늘려서 배경을 확장합니다.
* inpaint : Stretching된 이미지를 mask를 사용하여 늘린 부분만 img2img inpainting을 수행합니다.
* inpaint+lama : Controlnet의 inpaint+lama 모델을 사용하여 확장된 영역을 다시 그립니다.
* inpaint_only : Controlnet의 inpaint_only를 사용하여 확장된 영역을 다시 그립니다.


#### Alignment

이미지를 확장하고 원래 이미지를 어느 방향으로 정렬할 것인지를 결정합니다.

<img src="https://i.ibb.co/g62KhZQ/align.png">

#### Resize filter

Resize가 완료되고 난 이후에 외부 filter 코드를 호출하여 이미지 변환을 추가적으로 수행할 수 있다.


#### Resize by person intermediate

인물의 크기 비율을 나타냅니다. 이 값을 초과하면 이 크기가 되도록 배경을 확장시킵니다.



<table>
<tr>
<td>Original</td>
<td>Resize 0.7</td>
<td>Resize 0.5</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/XttbBz0/00133-3615254454.png"></td>
<td><img src="https://i.ibb.co/RS4tbZs/00135-3615254454.png"></td>
<td><img src="https://i.ibb.co/mHHqBKk/00134-3615254454.png"></td>
</tr>
</table>

<table>
<tr>
<td>Original</td>
<td>Alignment center</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/hmSG5SK/00074-2037889107.png"></td>
<td><img src="https://i.ibb.co/7kPycZ5/00075-2037889107.png"></td>
</tr>
<tr>
<td>Alignment bottom</td>
<td>Alignment bottom-left</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/2gPCbr4/00076-2037889107.png"></td>
<td><img src="https://i.ibb.co/x7T91QH/00080-2037889107.png"></td>
</tr>
</table>


Resize sample

<img src="https://i.ibb.co/G07KG6M/resize-00008-4017585008.png">












## Refiner

txt2img에서 만들어진 이미지를 한번 더 그리는 과정을 수행한다.   
txt2img + hires.fix 가 된 상황에서도 유효하다.

refiner는 이미지가 생성되고 detailing 하기전에 동작하며,   
sd-webui의 hires.fix + refiner를 합친 동작과 비슷하다.



<table>
<tr>
<td>txt2img(512x768)</td>
<td>txt2img + hires.fix(800x1200)</td>
<td>txt2img + hires.fix + refiner(1200x1800)</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/JCxXc9D/resize-00268-767037284.png"></td>
<td><img src="https://i.ibb.co/zR3nWKt/resize-00269-767037284.png"></td>
<td><img src="https://i.ibb.co/R21B0fr/resize-00270-767037284.png"></td>
</tr>
</table>


(위 예제는 결과를 모두 resize하여 동일한 크기이다.)

위 예제와 같이 3단계로 처리할 수도 있으나,   
hires.fix 단계 없이 refiner로 resize하여 처리할 수도 있다.



<p>
<img src="https://i.ibb.co/GFjgJ5B/refiner.png">
</p>

#### Enable refiner

refiner 사용 여부를 체크합니다.

#### CheckPoint

refiner를 이용하여 다시 그릴때 사용할 checkpoint를 지정합니다.

#### Use this checkpoint for detailing

위에서 지정한 checkpoint를 이용하여 detailing에 적용합니다.

#### Prompt

refiner가 이미지를 다시 그릴때 사용하는 prompt를 지정합니다.   
비어있다면 main prompt와 동일하고,채워져 있다면 main prompt를 무시합니다.   
만약 #!org!# 문자열이 있다면 main prompt를 대체합니다.

#### Negative prompt

refiner가 이미지를 다시 그릴때 사용하는 negative prompt를 지정합니다.

#### Sampling method

refiner가 사용할 sampler를 지정할 수 있습니다.   
(Euler A 권장)

#### Upscaler

refiner가 이미지를 resize하는 경우 사용할 upscaler를 지정합니다.

#### Refiner sampling steps

refiner가 사용할 samping steps를 지정합니다.   
(권장 20)

#### Refiner CFG scale

refiner가 사용할 CFG scale 값을 지정합니다.
dynamic threshold는 지원하지 않습니다.

#### Refiner denoising strength

refiner가 사용할 denoising strength를 지정합니다.   
(권장 0.4)

#### Refiner scale

refiner가 주어진 이미지를 scale 값으로 resize합니다.
만약 refiner width, refiner height가 설정되어있다면 무시됩니다.

#### Refiner width

이미지 폭을 해당 값으로 강제로 설정합니다.

#### Refiner height

이미지 높이를 해당 값으로 강제로 설정합니다.

<br>
<br>
<br>
