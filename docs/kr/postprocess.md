

# PostProcessing

## Resize

### Resize by person

그림 속 인물중 가장 신장이 큰 사람의 길이와 그림 높이의 비율이 설정값을 넘어가면 비율을 설정값로 맞추는 기능입니다.   
설정값이 0.90이고 인물의 전체 길이: 그림 높이의 비율이 0.95라고 한다면   
배경을 늘려서 인물의 비율이 0.90이 되도록 합니다.   
배경은 왼쪽, 오른쪽, 위쪽으로 늘어납니다.


이 기능은 2가지 방법을 제공하는데 다음과 같습니다.

#### Inpaint

Face Detailing과 같은 방법으로 이미지가 완전히 생성되고 난 후에 주변부를 확장합니다.   
이때 이미 생성된 이미지를 훼손하지 않고 주변부만 확장하기 때문에 직관적으로 확인이 가능합니다.
가장 빠르고 효과적으로 추천합니다.

#### Inpaint + lama

Inpaint와 같은 방법인데 BMAB에서 ControlNet을 불러서 Inpaint+lama를 이용해서 동작합니다.
이미지가 생성되고나서 디테일링 시작하기 전에 img2img를 이용하여 배경을 확장하여 전체적으로 인물을 작게 만드는 효과가 있습니다.

<table>
<tr>
<td><img src="https://i.ibb.co/X7P8kmH/00012-2131442442.jpg"></td>
<td><img src="https://i.ibb.co/Zc1nTz7/00022-2131442442.jpg"></td>
</tr>
<tr>
<td><img src="https://i.ibb.co/d0rSW0q/00013-2131442443.jpg"></td>
<td><img src="https://i.ibb.co/vHh2940/00023-2131442443.jpg"></td>
</tr>
<tr>
<td><img src="https://i.ibb.co/4jNJQ3b/00016-2131442446.jpg"></td>
<td><img src="https://i.ibb.co/HdcMpTH/00026-2131442446.jpg"></td>
</tr>
</table>

이 두가지 방법은 생성된 이미지를 축소하기만 할뿐 훼손하지 않습니다.   
이것이 Resize intermediate와 다른점입니다.


## Upscaler

최종적으로 이미지가 완성되고 난 이후에 이미지를 upscaler를 이용하여 크게 만듭니다.


#### Enable upscale at final stage

이미지 생성이 완료되고 난 이후에 Upscale을 수행합니다.   
960x540으로 생성하고 hires.fix를 x2로 하면 1920x1080 이미지가 나오는데   
여기서 Upscale을 x2를 하면 4K 이미지가 나오게 됩니다.

#### Detailing after upscale

이 옵션을 설정하면 위에서 언급한 Person, Face, Hand 에 대한 detailing을 upscale 이후에 수행합니다.   

#### Upscale Ratio

이미지를 얼마나 upscale할지 결정합니다.


