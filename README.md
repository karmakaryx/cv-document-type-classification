![banner_cv](./assets/banner_cv.jpg)

## **💻 Project Overview**
### Environment
- **OS:** Linux Ubuntu 20.04.6 LTS
- **System Memory:** 256GB RAM
- **Computing Power:** 24-Core / 48-Thread Multi-core CPU
- **GPU:** NVIDIA GeForce RTX 3090 (24GB)
- **NVIDIA Driver Version:** 535.86.10
- **CUDA Version:** 12.2 (Runtime: 11.8)
- **Tool:** VS Code (SSH), Google Colab
- **Language:** Python 3.10.13
- **Prerequisites:** GUI 및 그래픽 라이브러리가 생략된 경량 Linux는 OpenCV 실행 위해 시스템 패키지 설치 필요
```bash
apt update && apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
```

### Requirements
```
albumentations==1.3.1                             scikit-learn==1.7.2
augraphy==8.2.6                                   seaborn==0.13.2
jupyter==1.0.0                                    timm==0.9.12
matplotlib==3.10.8                                torch==2.1.0
numpy==1.26.4                                     torchvision==0.16.0
opencv-python<4.10                                tqdm==4.66.3
pandas==2.1.4                                     transformers==4.36.0
Pillow==12.2.0                                    wandb==0.24.0
python-dotenv==1.2.1
```

---

## **📋 Competition Info**
### 문서 타입 분류 (Document Type Classification)
- 총 17종의 아날로그 문서 데이터의 종류를 식별하고자 이미지 데이터를 클래스별로 분류한다.
> 계좌번호, 자동차 번호판, 자동차 계기판, 진료비영수증, 여권, 운전면허증,<br>
> 주민등록증, 자동차 등록증, 약제비 영수증, 처방전, 통원/진료 확인서, 입퇴원 확인서,<br>
> 진단서, 진료비 납입 확인서, 이력서, 소견서, 건강보험 임신출산 진료비 지급 신청서

### 일정 (Timeline)
- 2026.01.23 09:00 ~ 2026.02.04 18:00 (Competition)
- 2026.02.05 15:00 ~ 2026.02.05 17:00 (Seminar)

### 훈련데이터셋 정보 (Train Dataset Info)
- 훈련데이터: 총 1,570장의 문서 이미지
- 클래스별 이미지: 46~100장

### 평가데이터셋 정보 (Test Dataset Info)
- 평가데이터: 총 3,140장의 문서 이미지
- 난이도 조절을 위해 여러 augmentations 적용

### 평가지표 (Evaluation Metric)
- Macro F1 score: 각 클래스에 대한 F1 score를 개별적으로 계산 후 평균
- Precision과 Recall의 조화평균 (클래스마다 개수가 불균형할 때 모델의 성능을 더욱 정확하게 평가)
- Public은 전체 평가데이터 중 랜덤 샘플링된 50%, Private은 나머지 50%

### 규정 (Rule)
- 외부 데이터셋 사용 금지
- 평가데이터의 분석은 가능하나 평가데이터를 학습에 활용하는 행위는 금지 (eg. pseudo labeling)
- 사전학습 가중치 사용 규정: ImageNet 등 public하게 공개된 모든 기학습 가중치 사용은 허용
- API 사용 규정: 무료로 사용 가능한 API에 한정
- 상용 OCR 모델, LLM 모델 사용은 다른 참가자간 형평성 문제로 금지 (무료 한정해서는 허용)

---

## **💾 Data Description**
### EDA (Exploratory Data Analysis)
#### 1. 이미지 파일 mapping 정보
> meta.csv: 클래스 인덱스(0~16)와 클래스 이름 사이의 mapping 정보 (target, class_name)<br>
> train.csv: 훈련 이미지 이름과 클래스 인덱스 사이의 mapping 정보 (ID, target)

#### 2. Qualitative Glimpse
> 훈련데이터는 clean, 평가데이터는 noisy<br>
> 훈련데이터는 문서 전체가 정상적으로 찍혀있으나 평가데이터는 일부가 잘려 데이터가 손실된 케이스 많음<br>
> 계좌번호, 자동차 번호판, 자동차 계기판: 사진 형태<br>
> 여권, 운전면허증, 주민등록증, 자동차 등록증: 텍스트가 소량 있는 사진 형태<br>
> 약제비 영수증, 처방전, 통원/진료 확인서, 입퇴원 확인서, 진단서, 진료비 납입 확인서, 이력서, 소견서, 건강보험 임신출산 진료비 지급 신청서: 스캔형 문서 형태로 텍스트 작고 많음

#### 3. Class Label Distribution
> 클래스마다 대부분 이미지 100장씩 동일하게 분포되어 있으나 #1, #13, #14 클래스는 장수 미달<br>
> #1: 건강보험 임신출산 진료비 지급 신청서 (application_for_payment_of_pregnancy_medical_expenses)<br>
> #13: 이력서 (resume)<br>
> #14: 소견서 (statement_of_opinion)
![eda_class](./assets/eda_class.png)

#### 4. Image Size Distribution
> 가로 범위: 384px ~ 753px (평균: 497.6px)<br>
> 세로 범위: 348px ~ 682px (평균: 538.2px)
![eda_size](./assets/eda_size.png)

#### 5. Image File Size Distribution
> 훈련 이미지: 최소 25KB ~ 최대 164KB<br>
> 평가 이미지: 최소 25KB ~ 최대 149KB

#### 6. Confusion Matrix (V4 실험 결과로 중간 점검)
> 평균적으로 #3, #7, #4, #14 클래스가 오탐지 빈도 가장 높음
![eda_cm](./assets/eda_cm.png)

### Data Preprocessing
- 검증셋은 훈련데이터를 80:20으로 분리, 훈련셋은 과적합을 피하기 위해 K-Fold 옵션으로 쪼개되 17종 클래스 비율이 동일하게 들어가도록 Stratified를 사용해 모델의 일반화 성능을 높인다.

- 검증셋에는 augmentation 없이 정규화만 적용한다.<br>
원본 훈련데이터 1본, 코드상에서 원본 훈련데이터를 복제하여 각종 디지털 노이즈를 적용시킨 증강본, 이렇게 투트랙으로 2배 증식시켜 훈련데이터의 양적부족도 커버하면서 다양성을 높인다.

- 기본 100장보다도 더 모자란 3개 클래스는 oversampling으로 다른 클래스들과 키를 맞춘다.<br>
그 후 특히 혼동되는 클래스들은 따로 골라 추가 oversampling하고 가중치를 더 강하게 준다.

- 이미지 사이즈가 너무 작으면 식별이 어려우므로 최대한 키우고 싶었으나 그러면 소요되는 GPU 메모리 용량과 훈련시간을 감당할 수 없다. 모델들의 특성과 스펙을 꼼꼼하게 점검한 후 일괄 512px로 결정하고 같은 모델 내에서도 512에 적합한 버전을 골랐다.

- 평가데이터는 노이즈와 반전, 회전, 크롭, 마스킹 등으로 가득차 있다. 따라서 가장 난이도가 높은 #3 입퇴원확인서, #7 통원진료확인서, #4 진단서, #14 소견서만 전문적으로 식별하기 위해 마련된 일명 지옥의 노이즈 좀비 잡는 특공대를 마련. 이 특공대는 원본, 메인 모델의 증강본 외에 극악의 3단계 매운맛 노이즈 증강본이 존재한다.<br>
다뤄야 하는 클래스 수가 줄었으니 가중치, 오버샘플링도 더 세게. 이 4개 클래스만 뽑아 relabeling한 뒤 최종 결과물에 inverse mapping을 적용하여 메인모델과 cascade하고 이를 ensemble에 적용하여 최종 결과를 도출하는 방식이다.

- **Augmentation 전략 (Albumentations vs Augraphy)**<br>
평가 데이터의 현실적인 문서 오염을 모방하기 위해 일반 이미지 증강과 문서 특화 증강을 차별화하여 적용

  | 라이브러리 | 작동 방식 및 특징 | 주요 적용 효과 |
  | :--- | :--- | :--- |
  | **Albumentations** | 이미지 전체 픽셀을 수학적으로 변환 (기하학적/광학적 접근) | `RandomRotate90`, `Crop`, `Brightness` |
  | **Augraphy** | 문서를 Ink(글씨)와 Paper로 분리 후 물리적 노화 시뮬레이션 | `InkBleed`(잉크번짐), `DirtyRollers`(롤러자국) |

---

## **🧠 Modeling**
### Model Description
#### 1. MaxViT Base (maxvit_base_tf_512.in21k_ft_in1k)
- Multi-Axis Attention: Blocked Attention (국소적 정보) + Grid Attention (전역적 정보)
- MBConv(CNN 구조)와 Attention(Transformer 구조)의 하이브리드

#### 2. ConvNeXt V2 (convnextv2_base.fcmae_ft_in22k_in1k)
- Transformer의 장점을 흡수한 완성형 CNN
- FCMAE (Fully Convolutional Masked Autoencoder)
- GRN (Global Response Normalization) Layer

#### 3. DeiT III (deit3_base_patch16_384.fb_in22k_ft_in1k)
- 강력한 증강 기법과 향상된 학습 레시피
- 순수 Transformer 구조 기반

#### 4. Swin Transformer V2 (swinv2_base_window12to16_192to256.ms_in22k_ft_in1k)
- 계층적 구조 (Hierarchical Feature Maps)
- 윈도우 기반 어텐션 (Shifted Window Attention)
- V2에서 개선된 안정성 (Log-CPB & Post-Norm)

### Modeling Process
- ResNet50, EfficientNet-B3, Swin-Base, Swin-Large, ConvNeXt, DeiT, MaxViT 순으로 테스트
- 전통적인 CNN 계열부터 시작해서 문서 이해에 더 최적화된 SOTA 모델들까지 모두 적용해 본 뒤 CNN과 Transformer 계열로 나눠 각각 성과가 가장 좋은 모델들만 골라 앙상블
- IMG_SIZE = 512, LR = 1e-5 / 5e-5, EPOCHS = 20, BATCH_SIZE = 4~16, NUM_WORKERS = 16 기본값
- DeiT는 가볍고 중간 성과도 좋았으나 시간 부족으로 비슷한 계열의 더 고성능 모델인 MaxViT만 반영하고 우선순위 밀림
- Swin은 고전적 CNN 모델에 비해서는 좋은 성과를 보이나 ConvNeXt 이후로 계속 성능면에서 밀리며 결국 앙상블 제외
- MaxViT는 성능은 좋으나 지나치게 무거운게 단점
- 최종 MaxViT와 ConvNeXt의 Model Ensemble 선정: 둘은 다른 계열이라 공략하는 클래스가 다르기 때문
- Seed Ensemble: 적용하는 모델들의 seed를 전부 다르게 둔다. 그러나 동일 코드에 대해서 소수점까지 재현이 가능하도록 CuDNN 결정론적 연산 설정을 추가했다.
- 그 외 Hyperparameter Ensemble, Snapshot Ensemble도 적용해 보고 Model Ensemble은 모델별 가중치를 부여한 Weighted Soft Voting을 사용하여 모든 경우의 수를 다 적용해본다.

---

## **🕵️‍♀️ Hypothesis Notes**

---

## **💡 Insights from Trial and Error**

---

## **📊 Experiment Logger**
> 리더보드 총 36회나 제출한 관계로 일부 건만 기재
<table>
  <thead>
    <tr>
      <th>NO.</th>
      <th>DATE</th>
      <th>MODEL</th>
      <th>KEY CHANGES</th>
      <th>AUGMENTATION</th>
      <th>F1 (CV)</th>
      <th>F1 (LB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">34</td>
      <td align="center">20260204</td>
      <td>MaxViT+ConvNeXt</td>
      <td>Ensemble</td>
      <td>Augraphy</td>
      <td align="center">0.9783</td>
      <td align="center"><b>0.9742</b></td>
    </tr>
    <tr>
      <td align="center">30</td>
      <td align="center">20260201</td>
      <td>ConvNeXt+DeiT</td>
      <td>Stratified K-Fold</td>
      <td>Augraphy</td>
      <td align="center">0.9721</td>
      <td align="center"><b>0.9555</b></td>
    </tr>
    <tr>
      <td colspan="7" align="center">이후 실험 로그 기록할 시간도 없이 시간에 쫓김!😵‍💫 내가 한 짓은 W&B가 알고 있다..ㅎㅎ</td>
    </tr>
    <tr>
      <td align="center">14</td>
      <td align="center">20260126</td>
      <td>ConvNeXt-Base</td>
      <td>TTA</td>
      <td>Crop</td>
      <td align="center">0.9289</td>
      <td align="center"><b>0.9049</b></td>
    </tr>
    <tr>
      <td align="center">12</td>
      <td align="center">20260126</td>
      <td>ConvNeXt-Base</td>
      <td></td>
      <td>RandomRotate90</td>
      <td align="center">0.9416</td>
      <td align="center"><b>0.8678</b></td>
    </tr>
    <tr>
      <td align="center">11</td>
      <td align="center">20260125</td>
      <td>Swin-Base 384</td>
      <td>Oversampling</td>
      <td>Resize, Padding</td>
      <td align="center">0.9390</td>
      <td align="center"><b>0.8047</b></td>
    </tr>
    <tr>
      <td align="center">08</td>
      <td align="center">20260125</td>
      <td>Swin-Large 384</td>
      <td>Mixup, TTA</td>
      <td></td>
      <td align="center">0.8641</td>
      <td align="center"><b>0.7133</b></td>
    </tr>
    <tr>
      <td align="center">06</td>
      <td align="center">20260124</td>
      <td>Swin-Base 384</td>
      <td>Stratified 5-Fold</td>
      <td>Flip, Noise</td>
      <td align="center">0.9435</td>
      <td align="center"><b>0.8105</b></td>
    </tr>
    <tr>
      <td align="center">04</td>
      <td align="center">20260123</td>
      <td>EfficientNet-B3</td>
      <td>검증셋 분리</td>
      <td>Brightness, Rotation</td>
      <td align="center">0.8651</td>
      <td align="center"><b>0.5070</b></td>
    </tr>
    <tr>
      <td align="center">01</td>
      <td align="center">20260123</td>
      <td>ResNet50</td>
      <td>Baseline Code</td>
      <td>N/A</td>
      <td align="center">0.8264</td>
      <td align="center"><b>0.4195</b></td>
    </tr>
  </tbody>
</table>

![wandb1](./assets/wandb1.png)
![wandb2](./assets/wandb2.png)
![wandb3](./assets/wandb3.png)
![wandb4](./assets/wandb4.png)

---

## **🚀 Result**
### Champion Model Info
- **Version:** V6 (MaxViT & ConvNeXt Ensemble)
- **Training Time:** 17h 21m (MaxViT 기준)
- **Time per Epoch:** 14m 40s (MaxViT 기준)
- **Accuracy (Public):** 0.9742
- **Accuracy (Private):** 0.9638 (unselected)

### Leaderboard Rank: No. 1 🏆 [mid F1: 0.9742 / final F1: 0.9634]
![submission](./assets/submission.png)
![leaderboard](./assets/leaderboard.png)
![leaderboard_mid](./assets/leaderboard_mid.png)
![leaderboard_final](./assets/leaderboard_final.png)

### Presentation
- [[PDF] CV Seminar Presentation](./assets/seminar_cv.pdf)

---

## **📜 Version Log**
### [Releases](https://github.com/karmakaryx/cv-document-type-classification/releases)
### V1: Baseline Format Check
- 일정 수립, GitHub 설정
- 개발 환경 설정
- Jupyter Notebook을 Python script로 변환
- baseline code에서 hyperparameter 변경

### V2: EfficientNet-B3
- path env 설정
- seed CuDNN 결정론적 연산 설정 추가
- code formatting
- model & optimizer 변경: EfficientNet-B3, AdamW
- augmentation 추가
- training / validation sets 분리

### V3: Swin-Base 384
- best val macro F1 checkpoint 저장
- early stopping 적용
- Stratified K-Fold + fold ensemble 추론
- model 변경: Swin-Base 384
- augmentation 추가
- hyperparameter 변경

### V4: ConvNeXt-Base
- W&B 적용, Confusion Matrix 적용
- oversampling 적용
- image size 증가 후 padding 적용
- model 변경: ConvNeXt-Base
- augmentation 추가

### V5: ConvNeXt-Base
- code cleanup (Stratified K-Fold 단일 운영)
- TTA (Test Time Augmentation) 적용
- augmentation 추가

### V6: MaxViT & ConvNeXt Ensemble
- Augraphy 적용
- #7, #3, #4, #14 클래스에 특화되고 추가 노이즈 증강본이 적용된 스페셜 코드 작성
- 별도 스페셜 코드를 ConvNeXt와 cascade
- cascade한 ConvNeXt를 MaxViT와 앙상블 (Weighted Soft Voting)

---

## **⚙️ Components**
### Pipeline (using Mermaid Markdown)
```mermaid
graph TD
    %% 스타일 정의
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#111111;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#111111;
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#111111;
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#111111;

    %% 1. Input Stage
    TRAIN_DATA[Train CSV / Images]:::input
    TEST_DATA[Test CSV / Images]:::input

    %% 2. Data Prep
    OVERSAMPLE[Stratified 5-Fold & Oversampling]:::process

    %% 3. Augmentations
    subgraph Augmentations [Data Transforms]
        BASE_TF[Base & Val Transform]:::process
        AUG_TF[Augraphy Pipeline]:::process
        HELL_TF["Extreme Transform (For Cascade)"]:::process
    end

    %% 4. Models
    subgraph Models [Model Zoo]
        M_MAIN["Main Ensemble Models<br>- ConvNeXt Base<br>- DeiT III Base<br>- MaxViT Base"]:::model
        M_CASC["Cascade Specialist Model<br>- ConvNeXt V2 (Classes 3,4,7,14)"]:::model
    end

    %% 5. Inference
    TTA_RUN[8-Way TTA & Snapshot Ensemble]:::process
    SOFTMAX[Softmax & Weighted Avg Ensemble]:::process

    %% 6. Cascade Gate
    CASCADE_DECISION{"Is Primary Prediction<br>in Classes 3, 4, 7, 14?"}:::output
    RUN_CASCADE[Apply Cascade Model Prediction]:::output
    FINAL_ARGMAX[Final Argmax Label]:::output
    SUBMISSION[Final Submission CSV]:::input

    %% --- 흐름 연결 ---
    TRAIN_DATA --> OVERSAMPLE

    OVERSAMPLE --> BASE_TF
    OVERSAMPLE --> AUG_TF
    OVERSAMPLE --> HELL_TF

    BASE_TF & AUG_TF --> M_MAIN
    HELL_TF --> M_CASC

    TEST_DATA --> TTA_RUN
    M_MAIN --> TTA_RUN

    TTA_RUN --> SOFTMAX
    SOFTMAX --> CASCADE_DECISION

    CASCADE_DECISION -- Yes --> RUN_CASCADE
    M_CASC -. Cascade Feed .-> RUN_CASCADE

    RUN_CASCADE --> FINAL_ARGMAX
    CASCADE_DECISION -- No --> FINAL_ARGMAX

    FINAL_ARGMAX --> SUBMISSION
```

### Directory
```
├── assets/...                 # README images
├── code/
│   ├── baseline.ipynb         # baseline code (GitHub 관리 제외)
│   ├── eda.ipynb              # EDA Notebook (GitHub 관리 제외)
│   ├── cascade.py             # ConvNeXt + ConvNeXt special code 병합 (GitHub 관리 제외)
│   ├── cv_dtc_v1.py           # cv_dtc_v1.py ~ cv_dtc_v5.py
│   ├── cv_dtc_v6_conv.py      # ConvNeXt V2
│   ├── cv_dtc_v6_convspec.py  # ConvNeXt special
│   ├── cv_dtc_v6_deit.py      # DeiT III
│   ├── cv_dtc_v6_maxvit.py    # MaxViT Base
│   ├── cv_dtc_v6_swin.py      # Swin Transformer V2
│   ├── snapshot_conv.py       # snapshot 실수 복원
│   └── snapshot_convspec.py   # 실패한 fold 제외 후 재실험
├── data/                      # (GitHub 관리 제외)
│   ├── test/...               # test images
│   ├── train/...              # train images
│   ├── meta.csv               # class mapping info
│   ├── sample_submission.csv  # 0으로 초기화된 제출파일 template
│   └── train.csv              # train mapping info
├── output/                    # (GitHub 관리 제외)
│   ├── checkpoints/...        # 모델 가중치 저장
│   ├── confusionmatrix/...    # fold별 Confusion Matrix 파일
│   └── submission.csv         # 제출파일 생성
├── wandb/...                  # W&B log (GitHub 관리 제외)
├── .env.example               # 경로설정 template
├── .gitignore
├── README.md
└── requirements.txt
```

---

## **🛠️ etc.**
### Reference
- [[PyTorch] Reference API](https://docs.pytorch.org/docs/stable/pytorch-api.html)
- [[Augraphy] Official Documentation](https://augraphy.readthedocs.io/en/latest/)
- [[GitHub] Albumentations Public Archive](https://github.com/albumentations-team/albumentations)
- [[Weights & Biases] Official Documentation](https://docs.wandb.ai/)

### Role & Project Management
- **역할:** 팀장 (Project Lead) & Main System Architect
- **협업방식:** Slack 채널 중심의 일정 관리 및 의견 공유. 대회이므로 각자 개발하여 리더보드 제출 (최소 제출 횟수 의무화)
- **기여도 (90%):** 프로젝트 일정 관리, End-to-End 파이프라인 설계, 단독 개발 및 실험, Git 구축, 최종 산출물 작성 (팀원은 데이터 시각화 지원), 세미나 발표
- **Strategy:** 이전 대회에서 모호한 R&R(팀장 없음)과 소통 부재로 인해 홀로 프로젝트를 전담해야 했던 리스크를 경험. 이번 대회에서는 팀장으로서 명확한 방향성을 제시하고 팀 내 발생 가능한 리스크를 선제적으로 예방할 필요성을 느낌. 팀원들에게 부담을 주지 않으면서도 최소한의 참여를 보장할 수 있도록, "리더보드 의무 제출 각자 3회 이상"이라는 구체적인 가이드라인을 제시하고 합의를 이끌어냄.<br>
End-to-End 파이프라인 구조 설계, 데이터 전처리, 모델 실험 및 검증 등 핵심 개발 과정을 주도하여 안정적인 대회 1위 달성.<br>
팀원들이 효율적으로 기여할 수 있는 태스크(산출물 시각화 협조)를 배분하고 취합하여 세미나 발표까지 성공적으로 마무리.

### Project Retrospective
개발이 진행되면서 훈련 시간이 천문학적으로 늘며 CODE>WAIT>CODE>쪽잠>REPEAT의 반복이었던 2주였습니다. 특히 14시간짜리 실험이 대실패로 끝나거나, 학습 도중 0.01대의 괴랄한 F1로 모델이 폭주해버려 성공한 fold만 일부 수습해서 재실험을 시도한다거나, 엄청난 기대를 걸었던 #3, #7 클래스 귀신잡는 특공대가 의외로 하찮은 성과를 내거나, 온갖 시행착오를 겪으며 실전감각을 몸으로 익히는게 자학적으로(ㅠㅠ) 즐거웠습니다.<br>
특정 클래스의 가중치를 한 스푼만 높여도, 앙상블 도중 한쪽의 soft voting 비율을 1%만 낮춰도 개복치 같은 리더보드에 배신당하기를 반복하며 내가 베이킹을 하는건지 AI 개발을 하는건지 현타가 올 때도 있었지만, OCD 성향을 살려 데이터를 집요하게 비교분석하고 클래스별로 약점을 핀셋 공략해 결국 제 가설이 성공으로 입증되는 과정은 꽤 뿌듯했습니다.

<br>
