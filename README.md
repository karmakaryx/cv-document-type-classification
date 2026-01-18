## **[Computer Vision Competition] Document Type Classification**

## 📋 대회 개요 및 목표 (Project Overview & Goals)

#### 문서타입분류대회로 총17종의 이미지데이터를 클래스별로 분류한다.
- 계좌번호, 자동차 번호판, 자동차 계기판, 진료비영수증, 여권, 운전면허증
- 주민등록증, 자동차 등록증, 약제비 영수증, 처방전, 통원/진료 확인서, 입퇴원 확인서
- 진단서, 진료비 납입 확인서, 이력서, 소견서, 건강보험 임신출산 진료비 지급 신청서

#### 데이터셋 정보
- 학습데이터: 총 1570장
- 클래스별 이미지: 46~100장
<br><br>
- 테스트데이터: 총 3140장
- 특징: 난이도 조절을 위해 여러 augmentations 적용

#### 데이터 구조
- train/train.csv
- train/meta.csv
- test/sample_submission.csv

#### 평가지표 (Evaluation Metric)
- Macro f1 score: 각 클래스에 대한 f1 score를 개별적으로 계산 후, 평균
