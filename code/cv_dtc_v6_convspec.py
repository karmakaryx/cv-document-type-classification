### 1. Import Library & Define Functions ###
import os
import random

import albumentations as A
import augraphy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import wandb
from albumentations.pytorch import ToTensorV2
from augraphy import BrightnessTexturize, DelaunayTessellation, InkBleed
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import ConcatDataset, Dataset, DataLoader
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
def train_one_epoch(loader, model, optimizer, loss_fn, device, scheduler=None, epoch=0):
    model.train()
    train_loss = 0
    pred_list = []
    target_list = []

    pbar = tqdm(loader)
    for i, (image, target) in enumerate(pbar):
        image = image.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(image)
        loss = loss_fn(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch + i / len(loader))

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

# Albumentations에서 쓰기 위한 wrapper
def apply_augraphy_middle(image, **kwargs):
    data = augraphy_middle(image)
    if isinstance(data, dict):
        return data['output']
    return data

def apply_augraphy_hell(image, **kwargs):
    data = augraphy_hell(image)
    if isinstance(data, dict):
        return data['output']
    return data

### 2. Hyper-parameters ###
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model config
model_name = 'convnextv2_base.fcmae_ft_in22k_in1k'

# training config
IMG_SIZE = 512
LR = 3e-5
EPOCHS = 20
BATCH_SIZE = 8
NUM_WORKERS = 16

# 운영 옵션
N_SPLITS = 5                    #k-fold 개수
EARLY_STOPPING_PATIENCE = 5     #val_f1이 개선되지 않으면 중단할 epoch 수
EARLY_STOPPING_MIN_DELTA = 0.0  #개선으로 인정할 최소 증가량
CHECKPOINT_DIR = os.path.join(DATA_PATH, 'checkpoints_convspec')
CM_SAVE_DIR = os.path.join(DATA_PATH, 'confusionmatrix_convspec')


### 3. Load Data ###
augraphy_middle = augraphy.AugraphyPipeline([
    InkBleed(intensity_range=(0.1, 0.2), p=0.4),
    DelaunayTessellation(n_points_range=(500, 800), p=0.4),
    BrightnessTexturize(texturize_range=(0.9, 1.1), p=0.4),
])

augraphy_hell = augraphy.AugraphyPipeline([
    augraphy.Letterpress(n_samples=(50, 150), p=0.4),  #인쇄불량
    augraphy.Stains(p=0.2),                            #얼룩
    augraphy.DirtyRollers(p=0.3),                      #복사기오염
    DelaunayTessellation(n_points_range=(200, 400), p=0.2),
])

base_transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0, 0, 0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

trn_transform = A.Compose([
    # 기하학적 변형 (회전, 반전)
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.1, border_mode=0, value=(0, 0, 0), p=0.5),
    A.Perspective(scale=(0.02, 0.08), p=0.3),

    # 크기 및 비율 유지
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0, 0, 0)),

    # 문서가 잘린 경우
    A.RandomResizedCrop(
        height=IMG_SIZE,
        width=IMG_SIZE,
        scale=(0.8, 1.0),
        ratio=(1.0, 1.0),
        p=0.3
    ),

    # Augraphy
    A.Lambda(name='AugraphyEffect', image=apply_augraphy_middle, p=0.4),

    # 채도 및 색상 파괴
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=1, sat_shift_limit=5, val_shift_limit=5, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    ], p=0.4),
    A.ToGray(p=0.3),

    # 노이즈 주기 전 문서 선명화
    A.OneOf([
        A.Sharpen(alpha=(0.3, 0.5), lightness=(0.5, 1.0), p=0.9),
        A.CLAHE(clip_limit=4.0, p=0.9),
    ], p=0.5),

    # 노이즈 및 블러
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0),
    ], p=0.7),

    # 화질 저하 및 정규화
    A.ImageCompression(quality_lower=50, quality_upper=90, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

hell_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.1, border_mode=0, value=(0, 0, 0), p=0.5),

    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0, 0, 0)),

    A.RandomResizedCrop(
        height=IMG_SIZE, width=IMG_SIZE,
        scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3,
    ),

    A.Lambda(name='AugraphyEffect', image=apply_augraphy_hell, p=0.4),

    A.OneOf([
        A.GaussNoise(var_limit=(20.0, 100.0), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.2, 0.5), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
        A.Downscale(scale_min=0.6, scale_max=0.8, p=1.0),
    ], p=0.6),

    A.RandomShadow(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ImageCompression(quality_lower=50, quality_upper=80, p=0.4),

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 검증용 transform (노이즈 추가)
val_transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0,0,0)),

    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ], p=0.4),

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

# Cascade Classifier
# 심각: 7, 3
# 보통: 4, 14
TARGET_CLASSES = [7, 3, 4, 14]
train_df = train_df[train_df['target'].isin(TARGET_CLASSES)].reset_index(drop=True)

class_map = {original: new for new, original in enumerate(TARGET_CLASSES)}
train_df['target'] = train_df['target'].map(class_map)
reverse_class_map = {new: original for original, new in class_map.items()}

test_df = pd.read_csv(TEST_CSV)

# test dataset/loader (k-fold 앙상블에서 공통 사용)
tst_dataset = ImageDataset(test_df, TEST_DIR, transform=tst_transform)
tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


### 4. Train Model ###
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CM_SAVE_DIR, exist_ok=True)

# 클래스별 가중치 설정
weights = torch.ones(len(TARGET_CLASSES)).to(device)
weights[0] = 1.5  #7번
weights[1] = 1.5  #3번
weights[2] = 1.2  #4번
weights[3] = 1.2  #14번
loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

torch.cuda.empty_cache()

# 클래스 이름 정의
class_names = [
    'medical_outpatient_certificate',           #7
    'confirmation_of_admission_and_discharge',  #3
    'diagnosis',                                #4
    'statement_of_opinion',                     #14
]

# WandB 초기화 (wandb login)
try:
    wandb.init(
        project='cv-document-classifier',
        name=f'V6_{model_name}_special',
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

    target_counts = {
        0: 600, 1: 500,  #(7, 3)
        2: 350, 3: 300,  #(4, 14)
    }

    new_rows = []
    for target_id, target_max in target_counts.items():
        target_df = fold_train_df[fold_train_df['target'] == target_id]
        current_count = len(target_df)
        add_count = target_max - current_count
        if add_count > 0:
            oversampled_df = target_df.sample(n=add_count, replace=True, random_state=SEED)
            new_rows.append(oversampled_df)

    fold_train_df = pd.concat([fold_train_df] + new_rows).reset_index(drop=True)

    base_dataset = ImageDataset(fold_train_df, TRAIN_DIR, transform=base_transform)
    aug_dataset = ImageDataset(fold_train_df, TRAIN_DIR, transform=trn_transform)
    hell_dataset = ImageDataset(fold_train_df, TRAIN_DIR, transform=hell_transform)

    # hell_dataset에서 30%만 무작위로 추출
    hell_size = int(len(hell_dataset) * 0.3)
    unused_size = len(hell_dataset) - hell_size

    hell_subset, _ = torch.utils.data.random_split(
        hell_dataset, [hell_size, unused_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # 최종 합체: base(1.0) + aug(1.0) + hell(0.3)
    trn_dataset = ConcatDataset([base_dataset, aug_dataset, hell_subset])
    val_dataset = ImageDataset(fold_val_df, TRAIN_DIR, transform=val_transform)

    trn_loader = DataLoader(
        trn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=len(TARGET_CLASSES),
        drop_path_rate=0.2,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    num_train_steps = len(trn_loader) * EPOCHS
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_fold{fold}.pt')
    patience = 0

    for epoch in range(EPOCHS):
        train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device, scheduler=scheduler, epoch=epoch)
        val_ret = validate_one_epoch(val_loader, model, loss_fn, device=device)
        ret = {**train_ret, **val_ret, 'epoch': epoch}

        improved = ret['val_f1'] > (best_val_f1 + EARLY_STOPPING_MIN_DELTA)
        if improved:
            best_val_f1 = ret['val_f1']
            patience = 0
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_fold{fold}_epoch{epoch}.pt')
            save_checkpoint(best_ckpt_path, model, epoch, 'val_f1', best_val_f1)
            print(f'New best model saved at epoch {epoch} (Val F1: {best_val_f1:.4f})')
            best_tag = ' (best saved)'

            # Best 모델일 때 Confusion Matrix 저장
            cm_save_path = os.path.join(CM_SAVE_DIR, f'fold_{fold}_epoch_{epoch}_best_cm.png')
            plot_confusion_matrix(
                ret['confusion_matrix'],
                class_names=class_names,
                title=f'Confusion Matrix - Fold {fold+1}, Epoch {epoch} (Best)',
                save_path=cm_save_path,
                normalize=False,
            )
            plt.close()
        else:
            patience += 1
            best_tag = ''

        if (epoch + 1) % 5 == 0:
            snapshot_path = os.path.join(CHECKPOINT_DIR, f'snapshot_fold{fold}_epoch{epoch}.pt')
            save_checkpoint(snapshot_path, model, epoch, 'val_f1', ret['val_f1'])
            print(f'Snapshot saved at epoch {epoch} (End of Cycle)')

        log = f'epoch: {epoch}\n'
        log += f'train_loss: {ret['train_loss']:.4f}\ntrain_acc: {ret['train_acc']:.4f}\ntrain_f1: {ret['train_f1']:.4f}\n'
        log += f'val_loss: {ret['val_loss']:.4f}\nval_acc: {ret['val_acc']:.4f}\nval_f1: {ret['val_f1']:.4f}{best_tag}\n'
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
                    normalize=False,
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
        normalize=False,
    )
    plt.close()

    # 정규화된 Confusion Matrix도 저장
    final_cm_norm_save_path = os.path.join(CM_SAVE_DIR, f'fold_{fold}_final_cm_normalized.png')
    plot_confusion_matrix(
        final_val_ret['confusion_matrix'],
        class_names=class_names,
        title=f'Normalized Confusion Matrix - Fold {fold+1}',
        save_path=final_cm_norm_save_path,
        normalize=True,
    )
    plt.close()

    # Snapshot Ensemble 추론 (시간 문제로 취소)
    print(f'\n[Fold {fold + 1}] Snapshot Ensemble OFF - Inferring with Best Model Only')

    load_checkpoint(best_ckpt_path, model, device)
    fold_test_probs = predict_proba(tst_loader, model, device)

    if test_probs_sum is None:
        test_probs_sum = fold_test_probs
    else:
        test_probs_sum += fold_test_probs

    # 메모리 관리 (해당 fold 리소스 정리)
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
real_pred_list = [reverse_class_map[p] for p in pred_list]
pred_df = pd.DataFrame({'ID': tst_dataset.id, 'target': real_pred_list})

sample_submission_df = pd.read_csv(TEST_CSV)
assert (sample_submission_df['ID'] == pred_df['ID']).all()

pred_df.to_csv(f'{DATA_PATH}/output_convspec.csv', index=False)
np.save(f'{DATA_PATH}/probs_convspec.npy', test_probs_avg)
print(pred_df.head(10))
