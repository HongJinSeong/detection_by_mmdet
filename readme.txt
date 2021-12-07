실행환경 리눅스 ubuntu 18.04 torch 1.9.0 cuda 11.2 Docker

pip install -r requirements.txt 로 실행에 필요한 라이브러리 설치

ImportError: libGL ~~ 관련 에러 발생시 아래의 명령어로 필요 라이브러리 추가설치
    apt-get update
    apt-get -y install libgl1-mesa-glx
    apt-get install libglib2.0-0


deformable DETR
==> transformer 기반에 NMS나 ROI Align을 통하여 train을 위한 detection 영역을 추려내는 것이 아니라 이러한 부분도 network를 통해서 처리하고자 함.
==> 하지만 anchor를 통한 ROI Align이나 NMS를 통한 것이 없다보니 작은 영역에 대한 detection 성능이 너무 안나옴.
==> 의료 detection 은 작은 영역에대한 검출도 잘되야 되서 적합하지 않은 모델이였던것 같음.
==> 쓰려면 환경이 제한되고 target object 사이즈가 어느정도 큰 곳에 써야할 것으로 보인다. Down stream으로 R-CNN 계열을 쓰려면 수정해야 할 곳이 많아서 안됨

Resnest-Backbone + cascase rcnn
==> 작은 영역 검출도 잘됨
==> efficient detection 보다 성능좋다고 논문에 되어있는데 실제로 학습이나 성능도 나쁘진 않음
==> 다만 의료영샹의 특성을 내가 고려못해줘서 성능이 좋지 못했다고 생각됨

======>다음에 의료영상 관련 detection 할 일 생기면 Retina-Unet 으로 사용해보기..(이유는 해당 모델은 의료 domain에서 검증이 되었기 때문에...)
