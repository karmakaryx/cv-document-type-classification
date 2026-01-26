# **[Computer Vision] Document Type Classification**

## **📋 Project Overview & Goals**

### > 문서타입분류대회로 총17종의 이미지데이터를 클래스별로 분류한다.
- 계좌번호, 자동차 번호판, 자동차 계기판, 진료비영수증, 여권, 운전면허증
- 주민등록증, 자동차 등록증, 약제비 영수증, 처방전, 통원/진료 확인서, 입퇴원 확인서
- 진단서, 진료비 납입 확인서, 이력서, 소견서, 건강보험 임신출산 진료비 지급 신청서

### > 학습데이터셋 정보
- 학습데이터: 총 1570장
- 클래스별 이미지: 46~100장

### > 테스트데이터셋 정보
- 테스트데이터: 총 3140장
- 난이도 조절을 위해 여러 augmentations 적용

### > 평가지표 (Evaluation Metric)
- Macro F1 score: 각 클래스에 대한 F1 score를 개별적으로 계산 후, 평균

<br>

## **📊 Experiment Logger**
| No | 날짜 | 모델 | 주요변경사항 | Augmentation | LR | 점수 | 결과 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #001 | 2026-01-23 | ResNet50 | 기본 제공 코드 | None (Base) | 1e-4 | 0.4195 | S |
| #002 | 2026-01-23 | EfficientNet-B3 | 검증셋 분리 | Brightness, Rotation | 1e-4 | 0.5070 | S |
| #003 | 2026-01-24 | Swin-Base 384 | Stratified 5-Fold | Flip, Noise | 5e-5 | 0.8105 | S |
| #004 | 2026-01-25 | Swin-Large 384 | Mixup, TTA |  | 5e-5 | 0.7133 | F |
| #005 | 2026-01-25 | Swin-Base 384 | Oversampling | Resize, Padding | 1e-4 | 0.8047 | F |
| #006 | 2026-01-26 | ConvNeXt-Base |  | RandomRotate90,<br>Perspective | 1e-4 | 0.8678 | S |

<br>

## **🏆 Champion Model Info**
- **Version:** V4 (ConvNeXt-Base)
- **Training Time:** 1h 52m
- **Time per Epoch:** 2m 9s
- **Accuracy:** 86.78%
- **GPU:** 	NVIDIA GeForce RTX 3090

<br>

## **📜 Version Log**

### V1: Baseline Format Check
- Jupyter Notebook을 Python script로 변환
- Baseline code에서 hyperparameter 변경

### V2: EfficientNet-B3
- Path env 설정
- Seed CuDNN 결정론적 연산 설정 추가
- Code formatting
- Model & Optimizer 변경: EfficientNet-B3, AdamW
- Augmentation 추가
- Training / validation sets 분리

### V3: Swin-Base 384
- Best val macro F1 checkpoint 저장
- Early stopping
- Stratified K-Fold + fold ensemble 추론
- Model 변경: Swin-Base 384
- Augmentation 추가
- Hyperparameter 변경

### V4: ConvNeXt-Base
- WandB 적용, Confusion Matrix 적용
- Oversampling 적용
- Image Size 증가 후 padding 적용
- Model 변경: ConvNeXt-Base
- Augmentation 추가

<br>

## **🚀 Project Development Log**

### 2026-01-18 (Sun)
- **Key Task:** 프로젝트 착수
- **Note:** 일정 수립 (Notion 사용), GitHub 설정

### 2026-01-19 (Mon)
- **Key Task:** 개발 환경 설정
- **Note:** VS Code Extensions & library 설치, 서버 설정, SSH 접속 확인

### 2026-01-23 (Fri)
- **Key Task:** Leaderboard 첫 제출 완료
- **Note:** Baseline pipeline 검증, V1, V2 개발

### 2026-01-24 (Sat)
- **Key Task:** 검증셋 분리, 정체된 f1 score 개선
- **Result:** Leaderboard 🥇 갱신
- **Note:** V3 개발 (Swin Transformer와 Stratified 5-Fold가 극적 효과)

### 2026-01-25 (Sun)
- **Key Task:** f1 score 최고점 갱신 시도
- **Result:** Fail (0.0972 하락)
- **Note:** Swin-Large, Mixup, TTA 시도해 봤으나 모두 실패, WandB logging 적용

### 2026-01-26 (Mon)
- **Key Task:** Confusion Matrix 적용
- **Note:** 실패한 모델은 폐기하고 best로 실험환경 원복하는 기준 적용
<br>V4 개발 (Confusion Matrix를 통해 문제있는 클래스들을 적발, oversampling과 맞춤형 증강 추가)

<br>
