
## Quick Test

Enable을 체크하고 Config Tab에서 Preset "example"을 선택합니다.

contrast: 1.2   
brightness: 0.9   
sharpeness: 1.5

Edge enhancement 적용   
Face Detailing 적용   
Resize by person 적용   



## 기본 옵션

Enabled (VERSION): 기능을 켜고 끌 수 있습니다.

### Resize and fill override

Img2Img를 수행하는 경우 "Resize and fill" 을 선택하게 되면   
통상 좌우, 상하로 늘어나거나 비율이 같다면 그대로 크기만 변경됩니다.

Enabled 된 상태에서는 항상 이미지가 아래에 위치하고,   
왼쪽, 오른쪽, 윗쪽으로 비율에 맞게 늘어납니다.

인물의 윗쪽으로 여백이 없는 경우에 적용하면 효과적입니다.   
너무 크게 늘리게 되면 좋은 결과를 얻기 힘듭니다.   
대략 1.1, 1.2 정도 스케일에서 사용하시길 권장합니다.   

<p>
<img src="https://i.ibb.co/j3WzZrc/00408-3188840002.png" width="40%">
<img src="https://i.ibb.co/ZWMWVFB/00409-3188840002.png" width="40%">
</p>

<br>
<br>
<br>

# Preprocess

본 이미지 변경을 하기 전에 사전 처리 과정을 수행합니다.   
조건에 따라 hires.fix 과정에 개입할 수도 있습니다.

<a href="https://github.com/portu-sim/sd-webui-bmab/docs/kr/preprocess.md">Preprocess</a>

# BMAB

Person, Hand, Face detailer를 수행하거나, 이미지 합성 혹은 노이즈 추가등의 기능을 수행합니다.

<a href="https://github.com/portu-sim/sd-webui-bmab/docs/kr/bmab.md">bmab</a>

# Postprocess

이미지 처리가 완료된 이후, 인물의 크기에 따라 배경을 확장하거나 upscale을 할 수 있습니다.

<a href="https://github.com/portu-sim/sd-webui-bmab/docs/kr/preprocess.md">Postprocess</a>
