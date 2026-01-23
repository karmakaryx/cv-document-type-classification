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

# 경로 설정
load_dotenv()
BASE_DATA_PATH = os.getenv('BASE_DATA_PATH')
if not BASE_DATA_PATH:
    raise ValueError('BASE_DATA_PATH 환경변수가 설정되지 않았습니다.')

TRAIN_CSV = os.path.join(BASE_DATA_PATH, 'train.csv')
TRAIN_DIR = os.path.join(BASE_DATA_PATH, 'train')
TEST_CSV = os.path.join(BASE_DATA_PATH, 'sample_submission.csv')
TEST_DIR = os.path.join(BASE_DATA_PATH, 'test')

# 시드 고정
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True #CuDNN 결정론적 연산 설정 추가
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
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        preds = model(image)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f'Loss: {loss.item():.4f}')

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
    }

    return ret


### 2. Hyper-parameters ###
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data config
data_path = BASE_DATA_PATH

# model config
model_name = 'efficientnet_b3'

# training config
img_size = 384
LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 32
num_workers = 0


### 3. Load Data ###
# augmentation을 위한 transform 코드
trn_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),  #이미지 크기 조정
    A.ShiftScaleRotate(rotate_limit=10, p=0.5),  #살짝 돌리기
    A.RandomBrightnessContrast(p=0.2),  #밝기 조절
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #images normalization
    ToTensorV2()  #numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
])

# test image 변환을 위한 transform 코드
tst_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Dataset 정의
trn_dataset = ImageDataset(
    TRAIN_CSV,
    TRAIN_DIR,
    transform=trn_transform
)
tst_dataset = ImageDataset(
    TEST_CSV,
    TEST_DIR,
    transform=tst_transform
)
print(len(trn_dataset), len(tst_dataset))

# DataLoader 정의
trn_loader = DataLoader(
    trn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False
)
tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


### 4. Train Model ###
# load model
model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=17
).to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

for epoch in range(EPOCHS):
    ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device)
    ret['epoch'] = epoch

    log = ''
    for k, v in ret.items():
      if k == 'epoch':
          log += f'{k}: {v}\n'
      else:
          log += f'{k}: {v:.4f}\n'
    print(log)


### 5. Inference & Save File ###
preds_list = []

model.eval()
for image, _ in tqdm(tst_loader):
    image = image.to(device)

    with torch.no_grad():
        preds = model(image)
    preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())

pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list

sample_submission_df = pd.read_csv(TEST_CSV)
assert (sample_submission_df['ID'] == pred_df['ID']).all()

pred_df.to_csv(f'{BASE_DATA_PATH}/output.csv', index=False)
print(pred_df.head())
