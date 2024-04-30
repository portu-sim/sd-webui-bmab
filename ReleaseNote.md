### v24.05.01

* New Feature
  * ControlNet IpAdapter 기능을 제공합니다.
  * sd-webui-controlnet을 이용하여 제공하며, 랜덤하게 이미지를 적용할 수 있도록 개선했습니다.
  * Setting > BMAB에 현재 ControlNet IpAdapter의 model을 적어주셔야 합니다.
  * 기본적으로 모델은 default로 적용해두었습니다. 
  
* Code Fix
  * Setting에 Additional Checkpoint Path이 sd-webui 1.8에서 동작하지 않아, 이런 경우 빈칸으로 두면 오류가 발생하지 않도록 수정했습니다.
  * Person, Face 의 prompt가 hires.fix가 켜져있다면 hires.fix 것을 사용함.

### v24.04.30

* New Feature
  * Person Detailer에 checkpoint, vae, sampler, scheduler 및 steps 추가
  * Setting에 Additional Checkpoint Path 디렉토리를 입력하면 두 디렉토리 내용을 병합하여 전체적 적용됨.

* Code Fix
  * Person, Face 의 prompt가 hires.fix가 켜져있다면 hires.fix 것을 사용함.

### v24.04.26

* Code Fix
  * Pretraining 에 checkpoint, vae 선택 추가
  * checkpoint, vae, filter, pose 등에 대한 fresh 버튼을 삭제하고 하나로 통합
  * vintage filter 추가


### v24.04.20

* New Features
  * Pose
    * Face only 옵션 추가
    * Pose를 선택할 수 있도록 추가

### v24.04.17

* New Features
  * Pose
    * sd-webui-bmab/pose 디렉토리에 있는 이미지를 이용하여 openpose를 동작한다.
    * seed에 맞춰서 임의이 순서로 동작한다.
  * Face
    * CheckPoint를 선택할 수 있도록 추가
  * Pretraining
    * Filter를 선택할 수 있도록 추가
  * Refiner
    * VAE를 선택할 수 있도록 추가
  * Noise
    * Both, Low res fix, High res fix 추가.

### v24.04.16

* Code Fix
  * 1.9.0 지원
  * 1.8.0 하위 호환
  * CheckPoint 관련 전반 수정
  * basicsr 설치 오류 수정

### v24.04.05

* Code Fix
  * 1.8.0 설치시에 동작하지 않는 버그를 수정했습니다.
  * preprocess filter 분리
  * hires.fix - upscale 시에 필터가 정확하게 적용되지 않는 버그 수정
  * ControlNet Noise 사용시에 그레이 이미지를 사용하도록 수정
  * 그 외 몇가지 코드 리펙토링

### v23.11.30.0

* Code Fix
  * Support BMAB DR
  * Img2img rollback


### v23.11.28.0

* New Feature
  * Kohya Hires.fix
    * Preprocess에서 Kohya Hires.fix기능을 추가하였습니다.
    * 이 기능을 사용할때 sd 1.5 기준 1024x1025, 1536x1536, 1024x1536, 1536x1024일 경우가 가장 잘나옵니다.
    * 이 기능은 원작자가 SDXL을 위해서 만든 기능입니다. 굳이 sd 1.5에서 사용할 필요는 없습니다.


### v23.11.27.0

* New Feature
  * Stop generating gracefully
    * BMAB 프로세스가 완료되면 batch가 남아있더라도 종료하는 기능.
    * 이미지 생성 중간에 Interrupt 를 눌러서 종료가 아니라, 이미지 생성이 완료되면 종료된다.
    * 'Enable BMAB' 오른쪽에 작게 Stop이 있다.
  * FinalFilter
    * 최종 이미지에 수정을 가할 수 있도록 필터를 적용할 수 있도록 하는 기능.
    * 필터는 구현해서 filter에 넣으면 확인할 수 있다.
* BugFix
  * Img2Img와 openpose 사용시에 inpaint area 적용되지 않는 문제 수정.
  * 약간의 코드 리펙토링



