# 🏭 Smart Factory Real-Time Vision & Depth Inspection System

실시간 비전 결함 검사(Rule-based HSV & PatchCore AI), HP60C 뎁스(Depth) 검사, 12시(12H) 웨이퍼 자동 정렬, 실시간 공간 결함 히트맵 분석 및 컨베이어/키커 자동 제어를 지원하는 스마트 팩토리 교육 및 연구용 통합 플랫폼입니다.

---

## 🌟 주요 기능 (Key Features)

1. **비전 결함 검사 듀얼 엔진**:
   - **⚡ 룰베이스 (Rule-based HSV & Contour)**: 스크래치, 이물, 마커 불량을 2ms 미만의 초저지연으로 정밀 검출.
   - **🧠 딥러닝 AI (PatchCore Anomaly Detection)**: ResNet-18 임베딩 및 메모리 뱅크를 활용하여 비정형 표면 이상 탐지 및 픽셀 단위 이상치 히트맵 생성.
2. **12시 방향(12H) 자동 정렬 및 웨이퍼 상태 추적**:
   - 웨이퍼 상의 빨간색 기준점 스티커를 탐지하여 항상 상단(12H)으로 회전 정렬(Warp Affine).
   - 진입(ENTRY) $\rightarrow$ 검사 중심(INSPECT) $\rightarrow$ 이탈(EXIT) 전이 상태 머신 기반 **정지 정밀 검사(Stop-and-Inspect)** 및 **연속 이송(Continuous)** 모드 지원.
3. **가상 원형 웨이퍼 공간 결함 통계 및 경향성 분석기 (Spatial Analyzer)**:
   - 12H 기준 극좌표계 $(r, \theta)$ 기반 2D 가우시안 결함 밀도 히트맵 실시간 렌더링.
   - 4사분면(Q1~Q4) 및 반경(Center, Mid, Edge)별 불량 발생 확률(%) 산출 및 Hotspot 자동 진단.
4. **통합 GUI 대시보드 (Smart Factory Dashboard)**:
   - 비전/뎁스 듀얼 라이브 뷰, 컨베이어 속도(PWM) 및 키커 지연 시간 조절, 카메라 노출/밝기 조절, 실시간 통계 및 CSV 리포트 내보내기 지원.
5. **하드웨어 제어기 자동 감지 (Fault-tolerant Controller)**:
   - `/dev/ttyACM*`, `/dev/ttyUSB*` 포트 자동 스캔 및 통신 연결.

---

## 🛠️ 설치 및 환경 구축 (Setup Guide)

### 1. 시스템 요구사항
- **OS**: Linux (Ubuntu 20.04 / 22.04 권장)
- **Python**: Python 3.10 ~ 3.12
- **하드웨어**: USB 웹캠, HP60C 뎁스 카메라, 아두이노 스마트 팩토리 키트

### 2. 시리얼 포트 권한 설정 (최초 1회)
Linux 환경에서 아두이노(`/dev/ttyACM0`) 통신을 위해 현재 사용자를 `dialout` 그룹에 추가합니다:
```bash
sudo usermod -aG dialout $USER
# 적용을 위해 터미널 재시작 또는 로그아웃 후 재로그인
```

### 3. 가상환경 생성 및 패키지 설치
```bash
# 저장소 클론 및 이동
git clone https://github.com/intclecture/smart_factory_edu.git
cd smart_factory_edu

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 필수 의존성 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 실행 방법 (Usage)

### 1. 통합 GUI 대시보드 실행 (권장)

#### [모드 1] ⚡ 룰베이스 고속 검사 모드
```bash
source .venv/bin/activate
python factory.py --gui --mode rule
```

#### [모드 2] 🧠 딥러닝 AI (PatchCore) 이상 탐지 모드
```bash
source .venv/bin/activate
python factory.py --gui --mode patchcore
```

---

### 2. CLI 백그라운드 오케스트레이터 실행
GUI 없이 콘솔 및 OpenCV 윈도우로 직접 구동:
```bash
python factory.py --mode rule --cam-id 0
```

---

## 🧪 테스트 및 단독 실행 도구 (Tools & Tests)

| 스크립트 | 설명 | 실행 명령어 |
| :--- | :--- | :--- |
| **공간 분석기 단위 테스트** | 공간 좌표 매핑, 12H 정렬, 사분면 확률 계산 검증 | `python tests/test_spatial_analyzer.py` |
| **PatchCore AI 실시간 검사** | 단독 비전 카메라 실시간 이상치 추론 | `python inspect_anomaly_live.py` |
| **PatchCore 모델 재학습** | `dataset/train/good` 데이터를 이용한 모델 학습 | `python train_anomaly.py` |
| **HP60C 뎁스 뷰어** | 뎁스 카메라 단독 측정 및 시각화 | `python hp60c_live_viewer.py` |
| **카메라 HSV 마스크 튜닝** | 조명 환경에 따른 색상 마스크 실시간 조절 | `python check_hsv_mask.py` |

---

## 📂 디렉터리 구조 (Repository Structure)

```text
smart_factory_edu/
├── factory.py                  # 스마트 팩토리 메인 오케스트레이터
├── factory_gui.py              # Tkinter 통합 모니터링 & 제어 GUI 대시보드
├── requirements.txt            # 파이썬 의존성 패키지 목록
├── depth_cfg.json              # HP60C 뎁스 카메라 ROI 및 기준 높이 설정
├── color.cfg                   # 비전 조명 및 색상 설정
├── weights/                    # 사전 학습된 PatchCore ResNet-18 모델 가중치
│   └── patchcore_resnet18.pkl
├── dataset/                    # 웨이퍼 샘플 데이터셋 (train/good, val)
├── tests/                      # 단위 테스트 코드 (test_spatial_analyzer.py 등)
└── iotdemo/                    # 스마트 팩토리 핵심 패키지
    ├── vision/                 # 12H 정렬기, 공간 분석기, 룰베이스 검출기, 트래커
    ├── deeplearning/           # PatchCore Anomaly Detector (PyTorch)
    ├── depth/                  # HP60C RAW16 뎁스 센서 드라이버 및 검사기
    └── factory_controller/     # 아두이노 시리얼 통신 및 GPIO/PWM 액추에이터 제어기
```
