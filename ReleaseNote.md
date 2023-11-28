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



