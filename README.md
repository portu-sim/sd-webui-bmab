# sd-webui-bmab

## BMAB 기능 설명


### 기본 옵션

Enabled : 기능을 켜고 끌 수 있습니다.

Process before Img2Img
  * 활성화 되면 Img2Img의 경우 이미지 처리전에 기능을 수행합니다.
  * 활성화 되면 Txt2Img의 경우 이미지가 생성되고 hires.fix 수행전에 기능을 수행합니다.
    Hires가 켜져있지 않다면 기능을 수행하지 않습니다.

### 기본 기능

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


### Imaging

이 기능은 DDSD가 설치된 상태에서 사용할 수 있습니다.

#### Blend Image in Img2Img

이미지 업로드 상자에 입력한 이미지와 Img2Img에 입력된 이미지를 Blending합니다.
Blend Alpha 값으로 두 개의 이미지를 합성합니다.
"Process before Img2Img" 옵션이 적용됩니다.

#### Dino detect

Img2Img Inpainting 하는 경우에 마스크를 입력하지 않아도 Dino detect prompt에 있는 내용을 이용하여 자동으로 마스크를 생성합니다.
이미지를 업로드 하게되면 업로드된 이미지를 배경으로 하여 prompt로 입력된 부분을 업로드 이미지에 합성합니다.

### Face

이 기능은 DDSD가 설치된 상태에서 사용할 수 있습니다.

#### Face lighting

이 기능을 사용하게 되면 프로스세가 완료된 이후 얼굴의 밝기를 조정합니다.


## Example

Enabled : CHECK!!   
Process before Img2Img : CHECK!!

Contrast : 1.2   
Brightness : 0.9   
Sharpeness : 1.5   

Edge enhancement enabled : CHECK!!   
Edge low threshold : 50   
Edge high threshold : 200   
Edge strength : 0.5   



