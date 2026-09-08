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

기존 3DGS 계열 연구 저장소처럼 최상위 실행 파일과 역할별 모듈을 사용합니다.
학습 코드는 실제로 `train.py`에 있으며, 특정 연구의 알고리즘을 복사한 것은 아닙니다.

```text
blur-3dgs/
├── train.py                     # 교대 학습, joint refinement, 체크포인트 재개
├── render.py                    # 선명한 뷰 렌더링
├── metrics.py                   # 체크포인트 렌더링 + sharp GT 기반 PSNR/SSIM
├── arguments/__init__.py        # TrainConfig, YAML, CLI 옵션
├── scene/
│   ├── __init__.py              # 장면 로딩 인터페이스
│   ├── gaussian_model.py        # anisotropic Gaussians, 표준 DC-SH PLY
│   ├── cameras.py               # Frame, 카메라 좌표 검증
│   ├── dataset_readers.py       # manifest·영상·모션 캐시 로딩
│   ├── colmap_loader.py         # COLMAP text/binary 입력
│   ├── trajectory.py            # 두 endpoint twist의 linear se(3) 궤적
│   └── motion_prior.py          # 고정된 공식 Image-as-an-IMU 어댑터
├── gaussian_renderer/
│   ├── __init__.py              # PyTorch / gsplat RGB·깊이 렌더링
│   └── blur_renderer.py         # virtual pose 렌더링·노출 적분
├── utils/
│   ├── pose_utils.py            # SE(3), 역투영, 노출 경로, 최소제곱
│   ├── image_utils.py           # 영상 입출력·광도 변환
│   ├── loss_utils.py            # RGB·SSIM·공통 손실
│   └── motion_loss_utils.py     # 흐름·크기·방향·경로 손실
├── scripts/
│   ├── import_colmap.py         # 데이터 준비
│   ├── prepare_motion.py        # 사전학습 모션 관측값 캐시
│   └── make_synthetic.py        # 수치 검증용 합성 장면
├── configs/                     # GPU 연구 설정, CPU smoke 설정
├── tests/                       # 기하·미분·학습·실행 파일 회귀 테스트
├── docs/                        # 수식 대응·데이터·검증 기록
├── blur_gs/                     # 이전 python -m blur_gs 명령의 얇은 호환 계층
└── .github/workflows/           # CPU 테스트 CI
```

공통 구조는 [Deblur-GS](https://github.com/Chaphlagical/Deblur-GS),
[DeblurGS](https://github.com/taekkii/deblurgs),
[CoMoGaussian](https://github.com/Jho-Yonsei/CoMoGaussian),
[BAGS](https://github.com/snldmt/BAGS)를 참고했습니다. CUDA 소스를 직접 포함하지 않고
`gsplat`을 패키지로 설치하므로 빈 `submodules/`는 만들지 않습니다.


`data/`, `outputs/`, `checkpoints/`, `third_party/`, `.venv/`는 실행 중 생성하는 로컬
디렉터리이며 Git에서 제외됩니다. 첨부 PDF와 사전학습 가중치도 저장소에 포함하지 않습니다.

## CPU에서 전체 경로 실행

Python 3.10 이상을 사용합니다. 다음 명령은 Linux/macOS 기준입니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'

python scripts/make_synthetic.py --output data/smoke --size 32 --views 3
python train.py -s data/smoke -m outputs/smoke --config configs/smoke.yaml
python metrics.py -s data/smoke -m outputs/smoke --output outputs/smoke-eval
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
python scripts/import_colmap.py --model /path/to/undistorted/sparse --images /path/to/undistorted/images --output data/my-scene --downscale 4
```

[Image-as-an-IMU 연결 안내](docs/image_as_imu.md)에 따라 공식 `iaai` 패키지와 가중치를
준비한 후 고정된 관측값을 추출합니다. 학습에는 새로 생성한 `scene.motion.json`을 사용합니다.

```bash
python scripts/prepare_motion.py --data data/my-scene/scene.json --checkpoint checkpoints/image-as-imu.pth --device cuda
```

학습 전에 데이터 경로, 모션 캐시, 유효 confidence, PyTorch CUDA 및 gsplat 설치를
한 번에 검사합니다. `status`가 `ready`일 때만 장시간 학습을 시작하세요.

```bash
python scripts/preflight.py -s data/my-scene --backend gsplat --device cuda
```

GPU 학습 환경에서는 CUDA 지원 PyTorch와 빌드 도구를 설치한 후 다음을 실행합니다.
CUDA wheel은 서버의 드라이버·CUDA 환경에 맞춰 설치해야 합니다. CPU quickstart의
PyTorch를 그대로 사용하면 GPU 학습이 되지 않습니다.

```bash
python -m pip install -e '.[cuda,dev]'
python train.py -s data/my-scene -m outputs/my-scene --config configs/default.yaml
python render.py -s data/my-scene -m outputs/my-scene --output outputs/my-scene-render --backend gsplat --device cuda
```

체크포인트 재개 시 학습 설정과 프레임 순서를 유지하고, 총 반복 횟수를 늘릴 수 있습니다.
현재 linear 체크포인트는 format version 2입니다. 기존 Bezier 체크포인트는 선명한 뷰
렌더링에는 사용할 수 있지만, linear 학습은 새로운 run으로 시작해야 합니다.

```bash
python train.py -s data/my-scene -m outputs/my-scene --config configs/default.yaml --resume outputs/my-scene/checkpoint_001000.pt
```

학습 출력은 `checkpoint.pt`, `scene.ply`, `sharp/*.png`, `metrics.jsonl`, `summary.json`입니다.
PLY는 degree-zero SH를 사용하는 표준 Gaussian 표현입니다. Novel view는 새로운 `w2c`,
`K`를 가진 데이터 manifest로 렌더링할 수 있으며, 추론 시 모션 추정 모델은 필요 없습니다.

## 실행 인터페이스와 호환성

- `train.py -s <장면 경로> -m <출력 폴더>` 형태를 지원합니다.
  `--source_path`/`--data`, `--model_path`/`--output`도 같은 옵션입니다.
- `-s`는 준비된 장면 폴더 또는 manifest 파일을 받습니다. 폴더를 지정하면
  `scene.motion.json`을 우선 선택하고, 없으면 `scene.json`을 사용합니다.
  원본 COLMAP 폴더를 자동 변환하거나 없는 모션 캐시를 생성하지 않습니다.
- `render.py`와 `metrics.py`의 `-m`은 학습 출력 폴더입니다. 중간 체크포인트는
  `--checkpoint`로 지정하세요. `-s`를 생략하면 체크포인트에 저장된 manifest 경로를
  사용하며, 데이터를 이동했다면 `-s`를 다시 지정해야 합니다.
- 기본 렌더 출력은 체크포인트 옆 `renders/`, 평가 출력은 `evaluation/`입니다.
  `metrics.py`는 체크포인트에서 다시 렌더링하여 명시된 sharp GT와 PSNR/SSIM을 비교합니다.
  기존 렌더 폴더만 평가하는 외부 3DGS의 모든 옵션을 그대로 지원하는 것은 아닙니다.
- `--config` 생략 시 기존과 동일하게 CPU/PyTorch 기본 설정입니다.
  GPU는 `--config configs/default.yaml`을 명시하세요.
  `--device`, `--backend`, `--iterations`, `--exposure-samples`로 덮어쓸 수 있습니다.
- 이전 `python -m blur_gs train|render|evaluate|synthetic|import-colmap|prepare-motion|preflight`
  명령과 `blur-gs` 콘솔 명령은 새 코드로 연결됩니다. Python import 경로는
  [구조 변경 안내](docs/layout.md)를 따르세요.
- 체크포인트 format version 2, linear 궤적을 사용합니다. 현재 기본 설정과 smoke 설정의
  virtual pose는 모두 10개입니다.

## 구현 선택과 현재 제약

### 궤적 선택

`train.py --trajectory linear|spline|ode`로 노출 궤적을 선택합니다. 모든 방식의 기본
virtual pose 수는 10개이며 `--exposure-samples`로 독립적으로 바꿀 수 있습니다.

```bash
python train.py -s data/my-scene -m outputs/linear --config configs/default.yaml --trajectory linear
python train.py -s data/my-scene -m outputs/spline --config configs/default.yaml --trajectory spline --acceleration-weight 0.01
python train.py -s data/my-scene -m outputs/ode --config configs/default.yaml --trajectory ode --ode-steps 16 --acceleration-weight 0.01
```

- `linear`: 두 endpoint twist의 선형 보간. 기존 기본값입니다.
- `spline`: 4개 twist 제어점의 clamped cubic B-spline. knot vector는
  `[0,0,0,0,1,1,1,1]`로, 단일 cubic Bezier 구간과 같습니다. 가속도 손실은 해석적으로 적분합니다.
- `ode`: 이미지마다 초기 twist, 기본 속도와 `7 → 32 → 6` tanh MLP를 학습합니다.
  `dξ/dt = velocity + MLP(t, ξ)`를 고정 step RK4로 적분합니다. `--ode-steps`는
  각 query time까지의 적분 step 수이며 virtual pose 수와 다릅니다. 가속도 손실은
  9개 정규화 시점의 유한차분 근사입니다. CoMoGaussian 전체 구조의 재현은 아닙니다.

비선형 예제의 `0.01`은 실험용 시작값이며 검증된 최적값은 아닙니다. 생략하면 기존
`acceleration_weight: 0.0`을 유지합니다. IAAI flow는 endpoint 관측이므로 비선형
궤적에서도 `magnitude: endpoint`를 유지하세요.

Linear 체크포인트는 version 2를 유지하고 spline/ODE는 version 3으로 저장합니다.
`render.py`는 체크포인트에서 궤적 종류를 자동 복원합니다. 재개할 때에는 학습 때와
동일한 `--trajectory`, `--ode-steps`, 손실 설정을 전달해야 합니다. 궤적 종류를 바꾸는
실험은 별도 출력 폴더에서 새 학습으로 시작하세요.

- 기본 linear 방식은 두 6D endpoint twist를 `xi(t) = (1-t) xi_start + t xi_end`로 선형 보간하고,
  `T(t) = T0 @ exp(xi(t))`로 SE(3) pose를 생성합니다. 보간은 Lie algebra 기준이며,
  카메라 위치가 항상 월드 좌표에서 직선을 그리거나 body velocity가 일정하다는 의미는 아닙니다.
  CoMoGaussian의 Neural ODE,
  CMR 및 학습되는 픽셀별 노출 가중치는 현재 이식하지 않았습니다.
- 재블러링 virtual pose는 기본과 CPU smoke 모두 **10개** (`exposure_samples: 10`)입니다.
  시작과 끝을 포함해 균일하게 샘플링하며, 두 학습 endpoint와 렌더링 샘플 수는 별개입니다.
  시점은 `t = 0, 1/9, 2/9, ..., 1`이고 가중치는 각각 `1/10`입니다.
  이 10개에는 `t=0.5`가 없으므로 중간 시점 깊이를 별도로 한 번 렌더링합니다.
  기존 9-sample 체크포인트를 재개하려면 `--exposure-samples 9`로 원래 설정을 유지하세요.
- Linear twist의 2차 시간 미분은 0이므로 acceleration loss는 정확히 0이며 기본 가중치도
  0입니다. 중간 시점 pose anchor는 계속 적용합니다. `linear_exposure`는 이 궤적 선택이
  아니라 선형 광도 공간에서 재블러링할지를 뜻하는 별도 설정입니다.
- 외관은 degree-zero SH입니다. GPU 기본 설정에서는 위치 gradient가 큰 Gaussian을
  분할하고 낮은 opacity Gaussian을 제거합니다. 최대 개수와 스케줄은
  `configs/default.yaml`에서 조정할 수 있으며 `densify_every: 0`으로 끌 수 있습니다.
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
