
# BMAB

## 기본 기능

* Contrast : 대비값 조절 (1이면 변경 없음)
* Brightness : 밝기값 조절 (1이면 변경 없음)
* Sharpeness : 날카롭게 처리하는 값 조절 (1이면 변경 없음)
* Color Temperature : 색온도 조절, 6500K이 0 (0이면 변경 없음)
* Noise alpha : 프로세스 전에 노이즈를 추가하여 디테일을 올릴 수 있습니다. (권장값:0.1)
* Noise alpha at final stage : 최종 단계에서 노이즈를 추가하여 분위기를 다르게 전달할 수 있습니다.


## Imaging

### Blend Image in Img2Img

이미지 업로드 상자에 입력한 이미지와 Img2Img에 입력된 이미지를 Blending합니다.
Blend Alpha 값으로 두 개의 이미지를 합성합니다.
"Process before Img2Img" 옵션이 적용됩니다.

### Dino detect

Img2Img Inpainting 하는 경우에 마스크를 입력하지 않아도 Dino detect prompt에 있는 내용을 이용하여 자동으로 마스크를 생성합니다.
이미지를 업로드 하게되면 업로드된 이미지를 배경으로 하여 prompt로 입력된 부분을 업로드 이미지에 합성합니다.

#### Img2Img 에서 사용하는 경우

<p>
<img src="https://i.ibb.co/W5xs487/00027-3690585574.png" width="40%">
<img src="https://i.ibb.co/rk7xDSR/00467-2764185410.png" width="40%">
</p>
<p>
<img src="https://i.ibb.co/Byw3rY6/tmp3478vdur.png" width="40%">
<img src="https://i.ibb.co/7W6QhTG/00024-155186649.png" width="40%">
</p>



첫번째 image는 Img2Img 이미지로 지정
두번째 image는 BMAB의 Imaging에 Image 입력창에 지정

프로세스 과정에서 세번째 image를 합성하고 프롬프트에 따라서 결과가 얻어진다.   

Enabled : CHECK!!   

Contrast : 1.2   
Brightness : 0.9   
Sharpeness : 1.5

Enable dino detect : CHECK!!   
DINO detect Prompt : 1girl


#### Img2Img Inpaint 에서 사용하는 경우

DINO detect Prompt에 있는 내용대로 자동으로 마스크를 만들어준다.

<p>
<img src="https://i.ibb.co/W5xs487/00027-3690585574.png" width="30%">
<img src="https://i.ibb.co/80qQvDv/tmpnm78iuqo.png" width="30%">
<img src="https://i.ibb.co/mRT77BM/00028-2672855487.png" width="30%">
</p>


이번 예제에서는 배경을 변경했으니, inpaint 설정에서 "Inpaint Not Masked"를 선택해야 한다.   
반대로 "Inpaint Masked"를 하면 인물이 변경된다.


## Person

이 기능을 사용하게 되면 프로세스가 완료된 이후에, 인물을 감지하여 다시 그립니다.  
아래의 경우에 사용하면 효과적입니다.

* 인물이 배경에 비해 매우 작은 경우, 의복, 얼굴 등 인물 전체의 디테일이 올라갑니다.
* 4K와 같이 큰 이미지를 출력하는 경우, 업 스케일 이후에 인물이 작은 경우 이 기능을 사용하면 인물이 뚜렷해 집니다.
* Face Detailing과 같이 사용하면 좋은 효과를 볼 수 있습니다.


<img src="https://i.ibb.co/RSrvqM1/person.png">


#### Enable person detailing for landscape (EXPERIMENTAL)

풍경에서 인물을 자세하게 다시 그리는 기능을 활성화 합니다.

#### Block over-scaled image

이 기능이 켜지게 되면 인물을 찾아내서 크게 키워서 다시 그리는데 이때 확대된 이미지의 면적이 본래 이미지를 초과하게 되면 프로세스를 멈춥니다.   
sd-webui가 멈추거나 GPU를 보호하기 위한 목적입니다.

#### Auto scale if "Block over-scaled image" enabled

이 기능을 설정하면 위에서 언급한 "Block over-scaled image"로 차단될 경우 본래 이미지의 면적에 맞춰서 스케일을 조정하여 작업합니다.

#### Upscale Ratio

인물이 발견되면 주어진 비율로 키워서 자세하게 그립니다.

#### Denoising Strength

인물의 크기가 클 경우 0.4로 부족할 수 있습니다. 이런 경우 수치를 올려주세요.

#### Dilation mask

찾아낸 인물의 마스크를 확장합니다.

#### CFG Scale

인물을 다시 그릴때 사용하는 CFG scale 값입니다.

#### Large person area limit

인물이 이미지 속에서 차지하는 면적이 이 값을 초과하면 작업하지 않습니다.   
인물이 충분히 큰 경우 다시 그릴 필요가 없기 때문입니다.

#### Limit

이미지 속에 인물이 너무 많은 경우 면적단위로 큰 것부터 카운트하여 설정값을 초과하여 다시 그리지 않습니다.


<img src="https://i.ibb.co/n8PmL3P/00057-2574875327.jpg">
<img src="https://i.ibb.co/r2fdSmJ/00399-1097195856.png">


## Face

### Face Detailing

이 기능을 사용하게 되면 프로세스가 완료된 이후 After Detailer(AD)나 Detection Detailer(DD)와 같이    
얼굴을 보정합니다.   
이 기능을 동작시킨 후에 AD, DD가 동작하도록 설정한다면, 결과가 좋지 않을 수 있습니다.   

<img src="https://i.ibb.co/frx85BR/face.png">

최대 5개의 캐릭터에 대해 prompt를 별도로 지정할 수 있습니다.

#### Enable face detailing

face detailing 기능을 켜고 끌 수 있습니다.

#### Enable face detailing before hires.fix (EXPERIMENTAL)

face detailing 기능을 txt2img 과정의 hires.fix 직전에 한 번 더 수행합니다.   
얼굴을 보정한 이후에 upscale을 하기 때문에 더 좋은 품질의 이미지를 얻을 수 있습니다.   
하지만 부하가 더 들어가고, 이미지 변화가 심합니다.

#### Face detailing sort by

이미지 안에 여러 인물이 있는 경우 어떤 순서로 Detailing 할 것인지 결정합니다.

<img src="https://i.ibb.co/DR8g34t/00037-3214376443.png">
<img src="https://i.ibb.co/4JXdkpT/00036-3214376443.png">

왼쪽, 오른쪽 혹은 크기로 가능하며 없다면 기본적으로 Score 값이 높은 순서로 합니다.

#### Limit

이미지 않에 여러 인물이 있는 경우 위에서 정한 순서로 얼마나 수행할지 결정합니다.   
Limit이 1이라면 최대 1개만 수행한다는 뜻입니다.

#### Override Parameters

* Denoising Strength
* CFG Scale
* Width
* Height
* Steps
* Mask Blur

위 값에 대해 기본값이 아닌 UI에서 지정한 값을 사용합니다.

#### Inpaint Area

전체를 다시 그릴지 얼굴만 다시 그릴지를 결정합니다. 전체를 다시 그리는 것은 별로 추천하지 않습니다.

#### Only masked padding, pixels

기본값을 사용해 주세요.

#### Dilation

검출된 얼굴의 마스크 크기를 키웁니다.

#### Box threshold

Detector의 검출 값을 결정합니다. 기본값 0.35보다 작으면 face가 아닐 것으로 제외합니다.   
YOLO를 사용하는 경우 confidence를 대체합니다.

**좋은 결과를 얻기 위한 조언**

* Prompt에 얼굴 관련된 lora, textual inversion등 관련 내용을 뺍니다. sunglass 등은 무관합니다.
* 설정 파일에 얼굴마다 서로 다른 lora, textual inversion 등을 넣습니다.
* prompt에 lora, TI가 많을 경우 그림 생성 자유도가 떨어지는 것 같습니다.
* 그림속 모든 캐릭터가 공유되는 lora는 넣어주셔도 무방합니다.



## Hand

### Hand Detailing (EXPERIMENTAL)

손 표현이 잘못된 부분을 수정하는 기능입니다.   
만들어진 그림에서 손 부분을 자동으로 찾아내어 해당 부분을 다시 그리는 기능입니다.   
다만 손의 경우 다시 그려도 잘 그려질지 확실하지 않으며, 손을 자세하게 그리는 정도입니다.

<img src="https://i.ibb.co/fxQh9ZN/hand.png">

#### Enable hand detailing

손 보정 기능을 사용하도록 활성화 합니다.

#### Block over-scaled image

이 기능은 손을 찾아내어 확대해서 다시그리는 방법을 사용합니다.   
다시 그려야 하는 부분의 면적이 원래이미지를 초과하게 되면 작업을 수행하지 않습니다.   
이런 경우에는 Upscale Ratio를 줄이거나, 이 기능을 꺼야하는데, 이 기능을 끄면 매우 큰 그림을 다시 그릴 수도 있어서 GPU에 부하가 걸릴 수 있습니다..

#### Method
* subframe : 손을 포함하여 얼굴/머리 부분까지 찾아내어 상반신을 다시 그린다.
* each hand : 손을 찾아내여 3배 크기의 주변부 까지 다시 그려 손만 적용한다.
* each hand inpaint : 손을 찾아내어 3재 크기의 주변부를 기반으로 손만 다시 그린다.   
  매우 극단적으로 변형될 수 있어서 잘 그려지기 어렵다 모양이 갖춰진다면, subframe으로 다시 그리는 것을 추천한다.
* at once : 찾아낸 손을 모두 한번에 다시 그린다.
  

#### Prompt

Subframe에서는 입력하지 않을 것을 권장합니다.   
each hand, each hand inpaint시에 손 관련 프롬프트를 입력합니다.

#### Negative Prompt

Subframe에서는 입력하지 않을 것을 권장합니다.   
each hand, each hand inpaint시에 손 관련 네거티브 프롬프트를 입력합니다.

#### Denoising Strength

다시 그리는 경우 Denoising Strength 값 입니다.
* subframe : 0.4 권장
* 기타 0.55 이상 권장

#### CFG Scale

다시 그리는 경우 CFG Scale 값 입니다.

#### Upscale Ratio
상반신 / 손 주변을 찾아내어 얼마나 크게 확대하여 다시 그릴 것인지 지정한다.   
무조건 크게 그린다고 성공확률이 올라가는 것은 아니다.  
* subframe : 2.0
* 기타 : 2.0~4.0

#### Box Threshold

손을 찾아내지 못하는 경우 이 값을 낮추면, 찾아낼 수 있는 확률이 올라갑니다.   
하지만 잘 못 찾아낼 가능성도 올라갑니다.

#### Box Dilation

찾아낸 박스(손을 포함하여)의 외곽 부분을 얼마나 크게 할 것이 결정합니다. (only for subframe)

#### Inpaint Area

찾아낸 박스 전체를 다시 그릴 것인지, 손만 다시 그릴 것인지를 결정한다.   
손만 다시그리는 경우 손 모양이 원하지 않게 바뀔 수 있으나 크게 변경된다.   

#### Only masked padding

찾아낸 손의 내부 공간을 얼마 정도로 채울지를 결정합니다. 딱히 변경할 일 없습니다.

#### Additional Parameter

현재는 제공하지 않지만 향후 고급 사용자를 위한 옵션을 제공할 예정입니다.



## ControlNet

ControlNet을 이용하여 이미지에 노이즈를 추가하는 방법으로 디테일을 올리는 기능입니다.   
ControlNet의 Lineart 모델에 가우시안 노이즈 이미지를 입력으로 사용하여,   
결과물에 다양하고 복잡한 디테일을 추가합니다.

#### Noise Strength 

노이즈 강도를 지정합니다. (0.4 권장)

#### Begin

Sampling 단계 시작점

#### End

Sampling 단계 끝점

보통의 경우 0.4, 0, 0.4를 권장합니다. 혹은 이미지가 과도하게 그려질 경우 0.2, 0, 0.4 정도로 추천합니다.
과도하게 이미지가 그려진 경우 refiner를 사용하면 이미지를 어느 정도 안정시킬 수 있습니다.

아래는 모두 같은 seed입니다.
<table>
<tr>
<td>기본이미지</td>
<td>0.4</td>
<td>0.7</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/ypRrwmN/00007-51151519.jpg"></td>
<td><img src="https://i.ibb.co/j54HfHF/00009-51151519.jpg"></td>
<td><img src="https://i.ibb.co/MsgCZS3/00008-51151519.jpg"></td>
</tr>
</table>






<br>
<br>
<br>
