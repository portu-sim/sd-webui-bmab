
# BMAB 기능 설명


## 기본 옵션

Enabled : 기능을 켜고 끌 수 있습니다.

Process before Img2Img
  * 활성화 되면 Img2Img의 경우 이미지 처리전에 기능을 수행합니다.
  * 활성화 되면 Txt2Img의 경우 이미지가 생성되고 hires.fix 수행전에 기능을 수행합니다.
    Hires가 켜져있지 않다면 기능을 수행하지 않습니다.

### Random Prompt

기능이 켜지면 항상 동작합니다.
프롬프트 입력창에서 #random이 나타나면 그 이하 줄 단위로 랜덤하게 합쳐집니다.

(example)

1girl, standing,   
&#35;random   
street background,  
forest background,  
beach background,  

(결과)

"1girl, standing, street background,"   
"1girl, standing, forest background,"   
"1girl, standing, beach background,"  

셋중에 하나로 프롬프트가 결정됩니다.

### Resize and fill override

Img2Img를 수행하는 경우 "Resize and fill" 을 선택하게 되면   
통상 좌우, 상하로 늘어나거나 비율이 같다면 그대로 크기만 변경됩니다.

Enabled 된 상태에서는 항상 이미지가 아래에 위치하고,   
왼쪽, 오른쪽, 윗쪽으로 비율에 맞게 늘어납니다.

인물의 윗쪽으로 여백이 없는 경우에 적용하면 효과적입니다.   
너무 크게 늘리게 되면 좋은 결과를 얻기 힘듭니다.   
대략 1.1, 1.2 정도 스케일에서 사용하시길 권장합니다.   



## 기본 기능

* Contrast : 대비값 조절 (1이면 변경 없음)
* Brightness : 밝기값 조절 (1이면 변경 없음)
* Sharpeness : 날카롭게 처리하는 값 조절 (1이면 변경 없음)
* Color Temperature : 색온도 조절, 6500K이 0 (0이면 변경 없음)

위 네가지 기능은 "Process before Img2Img" 옵션과 무관하게 항상 마지막에 동작합니다.

* Adding noise : 노이즈를 강제로 추가합니다.

### Edge enhancemant

이미지 경계를 강화해 선명도를 증가시키거나 디테일을 증가시키는 기능입니다.

권장설정

* Edge low threshold : 50
* Edge high threshold : 200
* Edge strength : 0.5

이 기능이 켜지면 결과 이미지에 작업한 이미지가 추가적으로 나타납니다.


## Imaging

이 기능은 DDSD가 설치된 상태에서 사용할 수 있습니다.

### Blend Image in Img2Img

이미지 업로드 상자에 입력한 이미지와 Img2Img에 입력된 이미지를 Blending합니다.
Blend Alpha 값으로 두 개의 이미지를 합성합니다.
"Process before Img2Img" 옵션이 적용됩니다.

### Dino detect

Img2Img Inpainting 하는 경우에 마스크를 입력하지 않아도 Dino detect prompt에 있는 내용을 이용하여 자동으로 마스크를 생성합니다.
이미지를 업로드 하게되면 업로드된 이미지를 배경으로 하여 prompt로 입력된 부분을 업로드 이미지에 합성합니다.

## Face

이 기능은 DDSD가 설치된 상태에서 사용할 수 있습니다.

### Face lighting

이 기능을 사용하게 되면 프로스세가 완료된 이후 얼굴의 밝기를 조정합니다.


# Example
<img src="https://ibb.co/52yRWQ0"></img>
<img src="https://ibb.co/nLg4qdj"></img>

Enabled : CHECK!!   
Process before Img2Img : CHECK!!

Contrast : 1.2   
Brightness : 0.9   
Sharpeness : 1.5   

Edge enhancement enabled : CHECK!!   
Edge low threshold : 50   
Edge high threshold : 200   
Edge strength : 0.5   



