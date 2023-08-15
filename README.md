
# BMAB 기능 설명

## Quick Test

설치가 완료된 이후에 프롬프트 마지막줄에 "##example"을 추가한다.

```
1girl, ~~~~~~~

##example
```

그럼 아래와 같은 옵션이 적용된다.

contrast: 1.2   
brightness: 0.9   
sharpeness: 1.5

Edge enhancement 적용   
Face Detailing 적용   
Resize by person 적용   



## 기본 옵션

Enabled : 기능을 켜고 끌 수 있습니다.

기능이 꺼져있더라도 ##로 설정을 불러온 경우라면 설정파일이 적용되어 동작합니다.


Process before Img2Img
  * 활성화 되면 Img2Img의 경우 이미지 처리전에 기능을 수행합니다.
  * 활성화 되면 Txt2Img의 경우 이미지가 생성되고 hires.fix 수행전에 기능을 수행합니다.
    Hires가 켜져있지 않다면 기능을 수행하지 않습니다.

### Random Prompt (삭제 예정)

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

<p>
<img src="https://i.ibb.co/j3WzZrc/00408-3188840002.png" width="40%">
<img src="https://i.ibb.co/ZWMWVFB/00409-3188840002.png" width="40%">
</p>

### Multi face Detailer

BMAB는 이미지 내의 단일 혹은 여러 캐릭터의 얼굴을 보정하는 기능이 동작한다.   
BMAB 디렉토리 config 아래 .json파일로 필요한 프리셋을 등록하고 프롬프트에서 이를 호출할 수 있다.

예제)

```JSON
{
  "contrast": 1.2,
  "brightness": 0.9,
  "sharpeness": 1.5,
  "execute_before_img2img": true,
  "edge_flavor_enabled": true,
  "edge_low_threadhold": 50,
  "edge_high_threadhold": 200,
  "edge_strength": 0.5,
  "module_config": {
    "multiple_face": [
      {
        "denoising_strength": 0.30,
        "prompt": "첫번째 프롬프트... <lora:~~~~>, #!org!#",
        "steps": 15
      },
      {
        "denoising_strength": 0.30,
        "prompt": "두번째 프롬프트... <lora:~~~~>, #!org!#",
        "steps": 15
      },
      {
        "denoising_strength": 0.30,
        "prompt": "세번째 프롬프트... <lora:~~~~>, #!org!#",
        "steps": 15
      },
      {
        "denoising_strength": 0.30,
        "prompt": "네번째 프롬프트... <lora:~~~~>, #!org!#",
        "steps": 15
      },
      {
        "denoising_strength": 0.30,
        "prompt": "다섯번째 프롬프트... <lora:~~~~>, #!org!#",
        "steps": 15
      }
    ],
    "multiple_face_opt": {
      "mask dilation": 4,
      "limit": -1,
      "order": "right"
    }
  }
}
```

캐릭터 별로 별도로 지정할 수 있다.

<img src="https://i.ibb.co/DR8g34t/00037-3214376443.png">
<img src="https://i.ibb.co/4JXdkpT/00036-3214376443.png">

조건에 따라 왼쪽부터, 오른쪽 부터, 크기 순서대로 적용이 가능하다.


이 파일이 example.json이라고 저장되어 있다면 sd-webui 메인 프롬프트에서 "##example"를 완전히 새로운 라인에서   
추가하면 기존 UI 설정을 무시할 수 있으며, module_config/multiple_face 항목이 설정되어 있으면 디테일링을 수행한다.

```
1girl,

##example
```

예제는 최대 5개의 얼굴 크기에 따라 디테일링을 수행하며, 5개를 초과하면 디테일링을 수행하지 않는다.
1개만 등록하게 되면, 주변인들에 대한 불필요하한 디테일링을 막을 수 있으며,   
간혹 티셔츠에 그려진 얼굴이나, 액자에 그려진 얼굴에 대한 디테일링을 수행하지 않도록 설정할 수 있다.   

다만 limit 옵션을 -1 대신 20으로 주면, 5명을 초과한 20명까지 기본적인 face detailing을 수행한다.


그 밖에 denoising_strength, steps 등을 이용해 정교하게 설정할 수 있으며,   
프롬프트에 #!org!#이 있다면 그 부분은 사용자가 입력한 프롬프트로 변경된다.


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

<p>
<img src="https://i.ibb.co/Wsw2Wrh/00598-1745587019.png" width="40%">
<img src="https://i.ibb.co/z4nCW9Z/00600-1745587019.png" width="40%">
</p>

Enabled : CHECK!!   
Process before Img2Img : CHECK!!

Contrast : 1.2   
Brightness : 0.9   
Sharpeness : 1.5   

Edge enhancement enabled : CHECK!!   
Edge low threshold : 50   
Edge high threshold : 200   
Edge strength : 0.5   

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
Process before Img2Img : CHECK!!

Contrast : 1.2   
Brightness : 0.9   
Sharpeness : 1.5

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

## Face

### Face lighting

이 기능을 사용하게 되면 프로스세가 완료된 이후 얼굴의 밝기를 조정합니다.   
얼굴의 밝기를 조정한 이후에 보정을 위해 디테일링을 수행합니다.

## Resize

### Resize by person

그림 속 인물중 가장 신장이 큰 사람의 길이와 그림 높이의 비율이 설정값을 넘어가면 비율을 설정값로 맞추는 기능입니다.   
설정값이 0.90이고 인물의 전체 길이: 그림 높이가 0.95라고 한다면   
배경을 늘려서 비율이 0.90% 되도록 합니다.   
배경은 왼쪽, 오른쪽, 위쪽으로 늘어납니다.

<p>
<img src="https://i.ibb.co/j3WzZrc/00408-3188840002.png" width="40%">
<img src="https://i.ibb.co/ZWMWVFB/00409-3188840002.png" width="40%">
</p>

Enabled : CHECK!!   
Process before Img2Img : CHECK!!

Contrast : 1.2   
Brightness : 0.9   
Sharpeness : 1.5   

Resize by person : 0.85
