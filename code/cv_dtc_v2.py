### 1. Import Library & Define Functions ###
import os
import random

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

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
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
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


### 2. Hyper-parameters ###
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model config
model_name = 'efficientnet_b3'

# training config
IMG_SIZE = 384
LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = 0
VAL_RATIO = 0.2  #검증셋 비율 (80:20 split)


### 3. Load Data ###
# augmentation을 위한 transform 코드
trn_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),  #이미지 크기 조정
    A.ShiftScaleRotate(rotate_limit=10, p=0.5),  #살짝 돌리기
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),  #흐릿하게 만들기
    A.RandomBrightnessContrast(p=0.2),  #밝기 조절
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #images normalization
    ToTensorV2(),  #numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
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

# train.csv를 train/val로 분리
train_df = pd.read_csv(TRAIN_CSV)
train_data, val_data = train_test_split(
    train_df,
    test_size=VAL_RATIO,
    random_state=SEED,
    stratify=train_df['target'],  #클래스 비율 유지
)

# 임시 CSV 파일로 저장
TRAIN_SPLIT_CSV = os.path.join(DATA_PATH, 'train_split.csv')
VAL_SPLIT_CSV = os.path.join(DATA_PATH, 'val_split.csv')
train_data.to_csv(TRAIN_SPLIT_CSV, index=False)
val_data.to_csv(VAL_SPLIT_CSV, index=False)

print(f'Train: {len(train_data)} samples, Val: {len(val_data)} samples')
print(f'Train class distribution:\n{train_data["target"].value_counts().sort_index()}')
print(f'Val class distribution:\n{val_data["target"].value_counts().sort_index()}')

# Dataset 정의
trn_dataset = ImageDataset(
    TRAIN_SPLIT_CSV,
    TRAIN_DIR,
    transform=trn_transform,
)
val_dataset = ImageDataset(
    VAL_SPLIT_CSV,
    TRAIN_DIR,
    transform=val_transform,
)
tst_dataset = ImageDataset(
    TEST_CSV,
    TEST_DIR,
    transform=tst_transform,
)
print(f'Dataset sizes: Train {len(trn_dataset)}, Val {len(val_dataset)}, Test {len(tst_dataset)}')

# DataLoader 정의
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
tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


### 4. Train Model ###
# load model
model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=17,
).to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

for epoch in range(EPOCHS):
    train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device)
    val_ret = validate_one_epoch(val_loader, model, loss_fn, device=device)
    ret = {**train_ret, **val_ret, 'epoch': epoch}

    log = ''
    for k, v in ret.items():
      if k == 'epoch':
          log += f'{k}: {v}\n'
      else:
          log += f'{k}: {v:.4f}\n'
    print(log)


### 5. Inference & Save File ###
pred_list = []

model.eval()
for image, _ in tqdm(tst_loader):
    image = image.to(device)

    with torch.no_grad():
        pred = model(image)
    pred_list.extend(pred.argmax(dim=1).detach().cpu().numpy())

pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = pred_list

sample_submission_df = pd.read_csv(TEST_CSV)
assert (sample_submission_df['ID'] == pred_df['ID']).all()

pred_df.to_csv(f'{DATA_PATH}/output.csv', index=False)
print(pred_df.head())
