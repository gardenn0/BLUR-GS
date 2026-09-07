# BLUR-GS

사용자가 제공한 `Blur_GS.pdf`의 **BlurTraj-GS: Geometry-Coupled Continuous Camera
Trajectory Estimation for Motion-Blurred Gaussian Splatting**를 구현하는 연구 저장소입니다.
Image-as-an-IMU를 고정된 모션 관측기로 사용하고, Gaussian 장면의 깊이와 연속 카메라
궤적을 모션 일관성 손실로 연결합니다.

현재 버전은 **실행 가능한 핵심 알고리즘 구현**입니다. CPU 합성 데이터에서 미분과
전체 학습 경로를 검증합니다. 실제 이미지·공식 사전학습 가중치를 이용한 품질 재현이나
GPU 성능 검증을 완료했다는 의미는 아닙니다. 세부 범위는 [구현 대응표](docs/method.md)를
참고하세요.

## 구조

```text
blur-3dgs/
├── configs/                 # GPU 연구 설정, CPU smoke 설정
├── docs/                    # 수식 대응, 데이터 규격, IAAI 연결, 검증 기록
├── src/blur_gs/
│   ├── geometry.py          # SE(3), 깊이 역투영, 노출 경로, 가중 최소제곱
│   ├── trajectory.py        # 두 endpoint twist의 linear se(3) 궤적
│   ├── scene.py             # anisotropic Gaussians, 표준 3DGS PLY 출력
│   ├── rendering.py         # PyTorch / gsplat RGB·깊이 렌더링, 재블러링
│   ├── motion.py            # 공식 Image-as-an-IMU 모델 및 관측값 캐시
│   ├── losses.py            # RGB, 흐름·크기·방향·경로 손실
│   ├── training.py          # 궤적 ↔ 장면 교대 최적화, joint refinement, 재개
│   ├── data.py              # 데이터 검증과 로딩
│   ├── colmap.py            # COLMAP text/binary 입력
│   ├── synthetic.py         # 수치 검증용 합성 장면
│   └── evaluation.py        # 선명한 뷰 렌더링, PSNR/SSIM
├── tests/                   # 기하·미분·좌표·학습 경로 테스트
└── .github/workflows/       # CPU 테스트 CI
```

`data/`, `outputs/`, `checkpoints/`, `third_party/`, `.venv/`는 실행 중 생성하는 로컬
디렉터리이며 Git에서 제외됩니다. 첨부 PDF와 사전학습 가중치도 저장소에 포함하지 않습니다.

## CPU에서 전체 경로 실행

Python 3.10 이상을 사용합니다. 다음 명령은 Linux/macOS 기준입니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'

python -m blur_gs synthetic --output data/smoke --size 32 --views 3
python -m blur_gs train --data data/smoke/scene.json --config configs/smoke.yaml --output outputs/smoke
python -m blur_gs evaluate --data data/smoke/scene.json --checkpoint outputs/smoke/checkpoint.pt --output outputs/smoke-eval
python -m pytest -q
python -m ruff check .
```

Windows PowerShell에서는 활성화 대신 `.\.venv\Scripts\python.exe`로 위의 `python`을
바꿔 실행할 수 있습니다. 현재 작업 폴더에는 이 가상환경이 설치되어 있습니다.

합성 데이터의 흐름은 **알려진 3D 장면으로 계산한 정답**입니다. 이 smoke 테스트는
Image-as-an-IMU 추론이나 실제 데이터셋 성능 실험이 아닙니다.

## 실제 데이터 학습

먼저 이미지와 COLMAP 카메라·희소 점군을 준비합니다. 왜곡 보정된 PINHOLE 또는
SIMPLE_PINHOLE 모델만 지원합니다. 원본 카메라가 왜곡 모델이면 COLMAP의
`image_undistorter`를 먼저 실행하세요.

```bash
python -m blur_gs import-colmap --model /path/to/undistorted/sparse --images /path/to/undistorted/images --output data/my-scene --downscale 4
```

[Image-as-an-IMU 연결 안내](docs/image_as_imu.md)에 따라 공식 `iaai` 패키지와 가중치를
준비한 후 고정된 관측값을 추출합니다. 학습에는 새로 생성한 `scene.motion.json`을 사용합니다.

```bash
python -m blur_gs prepare-motion --data data/my-scene/scene.json --checkpoint checkpoints/image-as-imu.pth --device cuda
```

GPU 학습 환경에서는 CUDA 지원 PyTorch와 빌드 도구를 설치한 후 다음을 실행합니다.
CUDA wheel은 서버의 드라이버·CUDA 환경에 맞춰 설치해야 합니다. CPU quickstart의
PyTorch를 그대로 사용하면 GPU 학습이 되지 않습니다.

```bash
python -m pip install -e '.[cuda,dev]'
python -m blur_gs train --data data/my-scene/scene.motion.json --config configs/default.yaml --output outputs/my-scene
python -m blur_gs render --data data/my-scene/scene.motion.json --checkpoint outputs/my-scene/checkpoint.pt --output outputs/my-scene-render --backend gsplat --device cuda
```

체크포인트 재개 시 학습 설정과 프레임 순서를 유지하고, 총 반복 횟수를 늘릴 수 있습니다.
현재 linear 체크포인트는 format version 2입니다. 기존 Bezier 체크포인트는 선명한 뷰
렌더링에는 사용할 수 있지만, linear 학습은 새로운 run으로 시작해야 합니다.

```bash
python -m blur_gs train --data data/my-scene/scene.motion.json --config configs/default.yaml --output outputs/my-scene --resume outputs/my-scene/checkpoint_001000.pt
```

학습 출력은 `checkpoint.pt`, `scene.ply`, `sharp/*.png`, `metrics.jsonl`, `summary.json`입니다.
PLY는 degree-zero SH를 사용하는 표준 Gaussian 표현입니다. Novel view는 새로운 `w2c`,
`K`를 가진 데이터 manifest로 렌더링할 수 있으며, 추론 시 모션 추정 모델은 필요 없습니다.

## 구현 선택과 현재 제약

- 두 6D endpoint twist를 `xi(t) = (1-t) xi_start + t xi_end`로 선형 보간하고,
  `T(t) = T0 @ exp(xi(t))`로 SE(3) pose를 생성합니다. 보간은 Lie algebra 기준이며,
  카메라 위치가 항상 월드 좌표에서 직선을 그리거나 body velocity가 일정하다는 의미는 아닙니다.
  CoMoGaussian의 Neural ODE,
  CMR 및 학습되는 픽셀별 노출 가중치는 현재 이식하지 않았습니다.
- 재블러링 virtual pose는 기본 **9개** (`exposure_samples: 9`), CPU smoke는 **5개**입니다.
  시작과 끝을 포함해 균일하게 샘플링하며, 두 학습 endpoint와 렌더링 샘플 수는 별개입니다.
  기본 9개는 `t = 0, 0.125, ..., 1`이고 가중치는 각각 `1/9`입니다.
  학습 iteration마다 중간 시점 깊이를 위한 렌더링을 별도로 한 번 수행합니다.
- Linear twist의 2차 시간 미분은 0이므로 acceleration loss는 정확히 0이며 기본 가중치도
  0입니다. 중간 시점 pose anchor는 계속 적용합니다. `linear_exposure`는 이 궤적 선택이
  아니라 선형 광도 공간에서 재블러링할지를 뜻하는 별도 설정입니다.
- Gaussian 수는 고정되고 외관은 degree-zero SH입니다. densification/pruning 및
  고차 SH는 후속 고품질 재구성 실험에서 확장할 부분입니다.
- Image-as-an-IMU 공식 모델은 흐름과 깊이를 출력합니다. 신뢰도는 BLUR-GS에서 추가한
  텍스처·포화·유효영역 휴리스틱이며 학습된 confidence head가 아닙니다.
- IAAI의 virtual-start 흐름을 중간 시점 Gaussian 깊이와 같은 3D 점에 맞춰 비교합니다.
  시간 방향은 영상 전체에 대해 선택하며, 픽셀마다 독립적으로 부호를 바꾸지 않습니다.
- SfM 깊이는 임의 스케일입니다. IAAI의 metric depth를 그대로 warm-up 깊이에 섞지 않습니다.
  외부 초기 깊이를 사용하려면 중간 시점에 정렬된 **SfM 장면 단위의 z-depth**를 제공하세요.
- 표준 static-scene/global-shutter 모델입니다. 동적 물체, 심한 가림과 rolling shutter를
  완전히 처리하지 않습니다. 흐름의 유효 비율은 학습 로그에 기록합니다.
- 기본 `magnitude: endpoint`는 IAAI 출력 의미에 맞춥니다. 비선형 경로 길이는 다른 측정량이며,
  정답 경로가 없는 상태에서 총 경로 길이를 IAAI endpoint 크기와 동일하다고 가정하지 않습니다.

## 자료

- [방법과 수식 대응](docs/method.md)
- [데이터 규격](docs/data.md)
- [Image-as-an-IMU 연동](docs/image_as_imu.md)
- [검증 기록](docs/validation.md)
- [Image-as-an-IMU 공식 코드](https://github.com/jerredchen/image-as-an-imu)
- [CoMoGaussian 공식 코드](https://github.com/Jho-Yonsei/CoMoGaussian)
- [gsplat 1.5.3 렌더링 API](https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html)

참고 구현의 소스를 이 저장소에 복사하지 않았습니다. 외부 패키지·가중치·데이터셋의
이용 조건은 각각의 원 프로젝트를 따릅니다.
