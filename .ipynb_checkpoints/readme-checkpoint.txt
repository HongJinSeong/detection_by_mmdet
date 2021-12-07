실행환경 리눅스 ubuntu 18.04 torch 1.9.0 cuda 11.2 Docker

pip install -r requirements.txt 로 실행에 필요한 라이브러리 설치

ImportError: libGL ~~ 관련 에러 발생시 아래의 명령어로 필요 라이브러리 추가설치
    apt-get update
    apt-get -y install libgl1-mesa-glx
    apt-get install libglib2.0-0


train 소스 - train.ipynb
test 소스 - test.ipynb
test에 사용한 checkpoint - checkpoints/epoch_4.pth


학습에 사용한 데이터셋 - splits/train.txt
validation에 사용한 데이터셋 - splits/val.txt
