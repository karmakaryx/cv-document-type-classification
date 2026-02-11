### 1. Import Library & Define Functions ###
import os
import random

import timm
import torch
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import wandb
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 경로 설정
load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
if not DATA_PATH:
    raise ValueError('DATA_PATH 환경변수가 설정되지 않았습니다.')

TRAIN_CSV = os.path.join(DATA_PATH, 'train.csv')
TRAIN_DIR = os.path.join(DATA_PATH, 'train')
TEST_CSV = os.path.join(DATA_PATH, 'sample_submission.csv')
TEST_DIR = os.path.join(DATA_PATH, 'test')

# 시드 고정
SEED = 777
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True  #CuDNN 결정론적 연산 설정 추가
torch.backends.cudnn.benchmark = False

# 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, data, path, transform=None):
        # data: csv 경로(str | os.PathLike) 또는 pandas.DataFrame
        if isinstance(data, (str, os.PathLike)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError('data는 csv 경로 또는 pandas.DataFrame 이어야 합니다.')

        if 'ID' not in df.columns or 'target' not in df.columns:
            raise ValueError('입력 데이터는 ID, target 컬럼을 포함해야 합니다.')

        self.df = df.reset_index(drop=True)
        self.id = self.df['ID'].tolist()
        self.target = self.df['target'].fillna(0).astype(int).tolist()
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        name = self.id[idx]
        target = self.target[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target

# one epoch 학습을 위한 함수
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    pred_list = []
    target_list = []

    pbar = tqdm(loader)
    for image, target in pbar:
        image = image.to(device)
        target = target.to(device)

        model.zero_grad(set_to_none=True)

        pred = model(image)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_list.extend(pred.argmax(dim=1).detach().cpu().numpy())
        target_list.extend(target.detach().cpu().numpy())

        pbar.set_description(f'Loss: {loss.item():.4f}')

    train_loss /= len(loader)
    train_acc = accuracy_score(target_list, pred_list)
    train_f1 = f1_score(target_list, pred_list, average='macro')

    ret = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
    }

    return ret

# 검증을 위한 함수
def validate_one_epoch(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    pred_list = []
    target_list = []

    with torch.no_grad():
        for image, target in tqdm(loader, desc='Validating'):
            image = image.to(device)
            target = target.to(device)

            pred = model(image)
            loss = loss_fn(pred, target)

            val_loss += loss.item()
            pred_list.extend(pred.argmax(dim=1).detach().cpu().numpy())
            target_list.extend(target.detach().cpu().numpy())

    val_loss /= len(loader)
    val_acc = accuracy_score(target_list, pred_list)
    val_f1 = f1_score(target_list, pred_list, average='macro')
    cm = confusion_matrix(target_list, pred_list)

    ret = {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'confusion_matrix': cm,
        'pred_list': pred_list,
        'target_list': target_list,
    }

    return ret

# loader 순서대로 softmax 확률 반환
def predict_proba(loader, model, device):
    model.eval()
    probs_list = []

    with torch.no_grad():
        for image, _ in tqdm(loader, desc='Infer with 8-Way TTA'):
            image = image.to(device)
            batch_probs = []

            # 0, 90, 180, 270도 회전하며 예측
            for k in range(4):
                rotated = torch.rot90(image, k=k, dims=[2, 3])

                # 회전된 상태 그대로 예측
                logits = model(rotated)
                batch_probs.append(torch.softmax(logits, dim=1))

                # 회전된 상태에서 좌우 반전하여 예측
                flipped = torch.flip(rotated, dims=[3])
                logits_f = model(flipped)
                batch_probs.append(torch.softmax(logits_f, dim=1))

            # 8개 결과의 평균 계산
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            probs_list.append(avg_probs.detach().cpu())

    return torch.cat(probs_list, dim=0).numpy()

def save_checkpoint(path, model, epoch, metric_name, metric_value, extra=None):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metric_name': metric_name,
        'metric_value': float(metric_value),
    }
    if extra:
        ckpt['extra'] = extra
    torch.save(ckpt, path)

def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return ckpt

# Confusion Matrix 시각화 함수
def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', save_path=None, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        label = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        label = 'Confusion Matrix'

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion Matrix saved to: {save_path}')

    return plt.gcf()


### 2. Hyper-parameters ###
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model config
model_name = 'convnext_base'

# training config
IMG_SIZE = 512
LR = 1e-4
EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 0

# 운영 옵션
N_SPLITS = 5                    # k-fold 개수
EARLY_STOPPING_PATIENCE = 5     # val_f1이 개선되지 않으면 중단할 epoch 수
EARLY_STOPPING_MIN_DELTA = 0.0  # 개선으로 인정할 최소 증가량
CHECKPOINT_DIR = os.path.join(DATA_PATH, 'checkpoints')
CM_SAVE_DIR = os.path.join(DATA_PATH, 'confusion_matrix')


### 3. Load Data ###
# augmentation을 위한 transform 코드
trn_transform = A.Compose([
    # 크기 및 비율 유지
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0,0,0)),

    # 기하학적 변형 (회전, 반전 + 원근감 추가)
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    # 미세 회전
    A.ShiftScaleRotate(
        rotate_limit=20,
        scale_limit=0.1,
        p=0.5,
        border_mode=0,
        value=(0,0,0)
    ),

    # 문서가 비스듬하게 찍힌 경우
    A.Perspective(scale=(0.05, 0.1), p=0.3),

    # 노이즈 주기 전 문서 선명화
    A.OneOf([
        A.Sharpen(p=1.0),
        A.CLAHE(clip_limit=4.0, p=1.0),
        A.Emboss(p=1.0),
    ], p=0.4),  #40% 확률로 글자를 선명하게 깎아줌

    # 문서의 국소적 특징 학습 강화
    A.RandomGridShuffle(grid=(3, 3), p=0.2),

    # 노이즈 및 블러
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0),
    ], p=0.8),  #80% 확률로 이 중 하나의 노이즈가 걸림

    # 화질 저하 및 밝기 조절
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 검증용 transform (augmentation 없음)
val_transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0,0,0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# test image 변환을 위한 transform 코드
tst_transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0,0,0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 원본 train / test 로드
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# test dataset/loader (k-fold 앙상블에서 공통 사용)
tst_dataset = ImageDataset(test_df, TEST_DIR, transform=tst_transform)
tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


### 4. Train Model ###
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CM_SAVE_DIR, exist_ok=True)

# 클래스별 가중치 설정 (기본값값 1.0)
weights = torch.ones(17).to(device)

# 2. Confusion Matrix에서 많이 틀린 클래스들에 벌점 추가
weights[7] = 1.5   #통원진료확인서
weights[14] = 1.5  #소견서
weights[3] = 1.2   #입퇴원확인서

# 가중치가 적용된 Loss Function 선언
loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

torch.cuda.empty_cache()

# 클래스 이름 정의 (17개 클래스)
class_names = [
    "account_number",                                         #0
    "application_for_payment_of_pregnancy_medical_expenses",  #1
    "car_dashboard",                                          #2
    "confirmation_of_admission_and_discharge",                #3
    "diagnosis",                                              #4
    "driver_lisence",                                         #5
    "medical_bill_receipts",                                  #6
    "medical_outpatient_certificate",                         #7
    "national_id_card",                                       #8
    "passport",                                               #9
    "payment_confirmation",                                   #10
    "pharmaceutical_receipt",                                 #11
    "prescription",                                           #12
    "resume",                                                 #13
    "statement_of_opinion",                                   #14
    "vehicle_registration_certificate",                       #15
    "vehicle_registration_plate"                              #16
]

# WandB 초기화 (wandb login)
try:
    wandb.init(
        project='cv-document-classifier',
        name=f'cv_dtc_v4_{model_name}',
        config={
            'model_name': model_name,
            'img_size': IMG_SIZE,
            'lr': LR,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'num_workers': NUM_WORKERS,
            'n_splits': N_SPLITS,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA,
            'seed': SEED,
            'label_smoothing': 0.1,
            'weight_decay': 1e-2,
            'drop_path_rate': 0.2,
        }
    )
    wandb_enabled = True
except Exception as e:
    print(f'WandB 초기화 실패: {e}')
    wandb_enabled = False

pred_list = []
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_best_scores = []
test_probs_sum = None

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df['ID'], train_df['target'])):
    print(f'\n========== Fold {fold + 1}/{N_SPLITS} ==========')
    fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
    fold_val_df = train_df.iloc[va_idx].reset_index(drop=True)

    # 분할된 train 데이터 내에서만 부족한 클래스 증식 (1번: 46, 13번: 74, 14번: 50)
    target_counts = {1: 150, 14: 150, 13: 150}
    new_rows = []
    for target_id, target_max in target_counts.items():
        target_df = fold_train_df[fold_train_df['target'] == target_id]
        current_count = len(target_df)
        add_count = target_max - current_count
        if add_count > 0:
            oversampled_df = target_df.sample(n=add_count, replace=True, random_state=SEED)
            new_rows.append(oversampled_df)

    fold_train_df = pd.concat([fold_train_df] + new_rows).reset_index(drop=True)

    trn_dataset = ImageDataset(fold_train_df, TRAIN_DIR, transform=trn_transform)
    val_dataset = ImageDataset(fold_val_df, TRAIN_DIR, transform=val_transform)

    trn_loader = DataLoader(
        trn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17,
        drop_path_rate=0.2,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_fold{fold}.pt')
    patience = 0

    for epoch in range(EPOCHS):
        train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device)
        val_ret = validate_one_epoch(val_loader, model, loss_fn, device=device)
        ret = {**train_ret, **val_ret, 'epoch': epoch}

        improved = ret['val_f1'] > (best_val_f1 + EARLY_STOPPING_MIN_DELTA)
        if improved:
            best_val_f1 = ret['val_f1']
            patience = 0
            save_checkpoint(
                best_ckpt_path,
                model,
                epoch=epoch,
                metric_name='val_f1',
                metric_value=best_val_f1,
                extra={'fold': fold, 'model_name': model_name}
            )
            best_tag = ' (best saved)'

            # Best 모델일 때 Confusion Matrix 저장
            cm_save_path = os.path.join(CM_SAVE_DIR, f'fold_{fold}_epoch_{epoch}_best_cm.png')
            plot_confusion_matrix(
                ret['confusion_matrix'],
                class_names=class_names,
                title=f'Confusion Matrix - Fold {fold+1}, Epoch {epoch} (Best)',
                save_path=cm_save_path,
                normalize=False
            )
            plt.close()
        else:
            patience += 1
            best_tag = ''

        log = f'epoch: {epoch}\n'
        log += f'train_loss: {ret["train_loss"]:.4f}\ntrain_acc: {ret["train_acc"]:.4f}\ntrain_f1: {ret["train_f1"]:.4f}\n'
        log += f'val_loss: {ret["val_loss"]:.4f}\nval_acc: {ret["val_acc"]:.4f}\nval_f1: {ret["val_f1"]:.4f}{best_tag}\n'
        log += f'best_val_f1: {best_val_f1:.4f}\npatience: {patience}/{EARLY_STOPPING_PATIENCE}\n'
        print(log)

        # WandB 로깅 (fold별로 구분)
        if wandb_enabled:
            log_dict = {
                'fold': fold,
                'epoch': epoch,
                f'fold_{fold}/train_loss': ret['train_loss'],
                f'fold_{fold}/train_acc': ret['train_acc'],
                f'fold_{fold}/train_f1': ret['train_f1'],
                f'fold_{fold}/val_loss': ret['val_loss'],
                f'fold_{fold}/val_acc': ret['val_acc'],
                f'fold_{fold}/val_f1': ret['val_f1'],
                f'fold_{fold}/best_val_f1': best_val_f1,
                f'fold_{fold}/patience': patience,
            }

            # Best 모델일 때 WandB에 Confusion Matrix 이미지 로깅
            if improved:
                cm_fig = plot_confusion_matrix(
                    ret['confusion_matrix'],
                    class_names=class_names,
                    title=f'Confusion Matrix - Fold {fold+1}, Epoch {epoch}',
                    normalize=False
                )
                log_dict[f'fold_{fold}/confusion_matrix'] = wandb.Image(cm_fig)
                plt.close()

            wandb.log(log_dict)

        if patience >= EARLY_STOPPING_PATIENCE:
            print('Early stopping triggered.')
            break

    # fold best 로드 후 test 확률 예측 (앙상블)
    _ = load_checkpoint(best_ckpt_path, model, device=device)
    fold_best_scores.append(best_val_f1)

    # Best 모델로 최종 검증셋 Confusion Matrix 생성
    final_val_ret = validate_one_epoch(val_loader, model, loss_fn, device=device)
    final_cm_save_path = os.path.join(CM_SAVE_DIR, f'fold_{fold}_final_cm.png')
    plot_confusion_matrix(
        final_val_ret['confusion_matrix'],
        class_names=class_names,
        title=f'Final Confusion Matrix - Fold {fold+1}',
        save_path=final_cm_save_path,
        normalize=False
    )
    plt.close()

    # 정규화된 Confusion Matrix도 저장
    final_cm_norm_save_path = os.path.join(CM_SAVE_DIR, f'fold_{fold}_final_cm_normalized.png')
    plot_confusion_matrix(
        final_val_ret['confusion_matrix'],
        class_names=class_names,
        title=f'Normalized Confusion Matrix - Fold {fold+1}',
        save_path=final_cm_norm_save_path,
        normalize=True
    )
    plt.close()

    fold_test_probs = predict_proba(tst_loader, model, device=device)
    if test_probs_sum is None:
        test_probs_sum = fold_test_probs
    else:
        test_probs_sum += fold_test_probs

    # 메모리 관리 (해당 Fold 리소스 정리)
    del model, optimizer, trn_loader, val_loader, trn_dataset, val_dataset
    torch.cuda.empty_cache()

print(f'\nK-Fold best val_f1 per fold: {fold_best_scores}')
print(f'K-Fold mean best val_f1: {np.mean(fold_best_scores):.4f} ± {np.std(fold_best_scores):.4f}')

# K-Fold 결과 WandB 로깅
if wandb_enabled:
    wandb.log({
        'kfold/mean_val_f1': np.mean(fold_best_scores),
        'kfold/std_val_f1': np.std(fold_best_scores),
    })
    for i, score in enumerate(fold_best_scores):
        wandb.log({f'kfold/fold_{i}_best_val_f1': score})

test_probs_avg = test_probs_sum / N_SPLITS
pred_list = test_probs_avg.argmax(axis=1).tolist()

# WandB 종료
if wandb_enabled:
    wandb.finish()


### 5. Inference & Save File ###
pred_df = pd.DataFrame({'ID': tst_dataset.id, 'target': pred_list})
sample_submission_df = pd.read_csv(TEST_CSV)
assert (sample_submission_df['ID'] == pred_df['ID']).all()

pred_df.to_csv(f'{DATA_PATH}/output.csv', index=False)
print(pred_df.head(20))
