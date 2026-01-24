### 1. Import Library & Define Functions ###
import os
import random
import time

import timm
import torch
import albumentations as A
import numpy as np
import pandas as pd
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
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
SEED = 42
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

    ret = {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
    }

    return ret

# loader 순서대로 softmax 확률 반환: (N, C) numpy
def predict_proba(loader, model, device):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for image, _ in tqdm(loader, desc='Infer'):
            image = image.to(device)
            logits = model(image)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            probs_list.append(probs)
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


### 2. Hyper-parameters ###
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model config
model_name = 'swin_base_patch4_window12_384'

# training config
IMG_SIZE = 384
LR = 5e-5
EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 0
VAL_RATIO = 0.2  #검증셋 비율 (80:20 split)

# 운영 옵션
USE_KFOLD = True                # True: Stratified K-Fold 학습/앙상블, False: 단일 train/val split
N_SPLITS = 5                    # k-fold 개수 (USE_KFOLD=True일 때)
EARLY_STOPPING_PATIENCE = 5     # val_f1이 개선되지 않으면 중단할 epoch 수
EARLY_STOPPING_MIN_DELTA = 0.0  # 개선으로 인정할 최소 증가량
CHECKPOINT_DIR = os.path.join(DATA_PATH, 'checkpoints')


### 3. Load Data ###
# augmentation을 위한 transform 코드
trn_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),

    # 기하학적 변형 (회전, 반전 등 테스트셋의 무작위성에 대비)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(rotate_limit=15, p=0.5),

    # 노이즈 및 블러
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
    ], p=0.7),  #70% 확률로 이 중 하나의 노이즈가 걸림

    # 화질 저하 (압축 노이즈 재현)
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),

    # 밝기 및 대비 (조명 조건 노이즈)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 검증용 transform (augmentation 없음)
val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# test image 변환을 위한 transform 코드
tst_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
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
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# 로그출력 시작시간
start_tick = time.time()
print(f'Log start time: {time.ctime(start_tick)}')

pred_list = []
if USE_KFOLD:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_best_scores = []
    test_probs_sum = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df['ID'], train_df['target'])):
        print(f'\n========== Fold {fold + 1}/{N_SPLITS} ==========')
        fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[va_idx].reset_index(drop=True)

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
            else:
                patience += 1
                best_tag = ''

            log = f'epoch: {epoch}\n'
            log += f'train_loss: {ret["train_loss"]:.4f}\ntrain_acc: {ret["train_acc"]:.4f}\ntrain_f1: {ret["train_f1"]:.4f}\n'
            log += f'val_loss: {ret["val_loss"]:.4f}\nval_acc: {ret["val_acc"]:.4f}\nval_f1: {ret["val_f1"]:.4f}{best_tag}\n'
            log += f'best_val_f1: {best_val_f1:.4f}\npatience: {patience}/{EARLY_STOPPING_PATIENCE}\n'
            print(log)

            if patience >= EARLY_STOPPING_PATIENCE:
                print('Early stopping triggered.')
                break

        # fold best 로드 후 test 확률 예측 (앙상블)
        _ = load_checkpoint(best_ckpt_path, model, device=device)
        fold_best_scores.append(best_val_f1)
        fold_test_probs = predict_proba(tst_loader, model, device=device)
        if test_probs_sum is None:
            test_probs_sum = fold_test_probs
        else:
            test_probs_sum += fold_test_probs

    print(f'\nK-Fold best val_f1 per fold: {fold_best_scores}')
    print(f'K-Fold mean best val_f1: {np.mean(fold_best_scores):.4f} ± {np.std(fold_best_scores):.4f}')

    test_probs_avg = test_probs_sum / N_SPLITS
    pred_list = test_probs_avg.argmax(axis=1).tolist()
else:
    # 단일 split (stratified)
    train_data, val_data = train_test_split(
        train_df,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=train_df['target']  #클래스 비율 유지
    )

    print(f'Train: {len(train_data)} samples, Val: {len(val_data)} samples')
    print(f'Train class distribution:\n{train_data["target"].value_counts().sort_index()}')
    print(f'Val class distribution:\n{val_data["target"].value_counts().sort_index()}')

    trn_dataset = ImageDataset(train_data.reset_index(drop=True), TRAIN_DIR, transform=trn_transform)
    val_dataset = ImageDataset(val_data.reset_index(drop=True), TRAIN_DIR, transform=val_transform)

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
        num_classes=17
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_single_split.pt')
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
                extra={'model_name': model_name}
            )
            best_tag = ' (best saved)'
        else:
            patience += 1
            best_tag = ''

        log = f'epoch: {epoch}\n'
        log += f'train_loss: {ret["train_loss"]:.4f}\ntrain_acc: {ret["train_acc"]:.4f}\ntrain_f1: {ret["train_f1"]:.4f}\n'
        log += f'val_loss: {ret["val_loss"]:.4f}\nval_acc: {ret["val_acc"]:.4f}\nval_f1: {ret["val_f1"]:.4f}{best_tag}\n'
        log += f'best_val_f1: {best_val_f1:.4f}\npatience: {patience}/{EARLY_STOPPING_PATIENCE}\n'
        print(log)

        if patience >= EARLY_STOPPING_PATIENCE:
            print('Early stopping triggered!')
            break

    # best 모델 로드 후 test 예측
    _ = load_checkpoint(best_ckpt_path, model, device=device)
    test_probs = predict_proba(tst_loader, model, device=device)
    pred_list = test_probs.argmax(axis=1).tolist()

# 로그출력 종료시간
end_tick = time.time()
print(f'Log end time: {time.ctime(end_tick)}')
print(f'Log elapsed time: {round((end_tick - start_tick) / 60, 1)} mins')


### 5. Inference & Save File ###
pred_df = pd.DataFrame({'ID': tst_dataset.id, 'target': pred_list})
pred_df['target'] = pred_list

sample_submission_df = pd.read_csv(TEST_CSV)
assert (sample_submission_df['ID'] == pred_df['ID']).all()

pred_df.to_csv(f'{DATA_PATH}/output.csv', index=False)
print(pred_df.head(10))
