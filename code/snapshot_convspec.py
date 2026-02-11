import os

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
if not DATA_PATH:
    raise ValueError('DATA_PATH 환경변수가 설정되지 않았습니다.')

TEST_CSV = os.path.join(DATA_PATH, 'sample_submission.csv')
TEST_DIR = os.path.join(DATA_PATH, 'test')
CHECKPOINT_DIR = os.path.join(DATA_PATH, 'checkpoints_convspec')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'convnextv2_base.fcmae_ft_in22k_in1k'

IMG_SIZE = 512
LR = 3e-5
EPOCHS = 20
BATCH_SIZE = 8
NUM_WORKERS = 16
TARGET_CLASSES = [7, 3, 4, 14, 10, 11, 12]

class ImageDataset(Dataset):
    def __init__(self, data, path, transform=None):
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

def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return ckpt

tst_transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0,0,0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class_map = {original: new for new, original in enumerate(TARGET_CLASSES)}
reverse_class_map = {new: original for original, new in class_map.items()}
test_df = pd.read_csv(TEST_CSV)

tst_dataset = ImageDataset(test_df, TEST_DIR, transform=tst_transform)
tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

best_ckpt_files = [
    'best_fold0_epoch8.pt',   #1폴드 (백업본 파일명)
    'best_fold1_epoch3.pt',   #2폴드 (새로 나올 파일명)
    'best_fold2_epoch8.pt',   #3폴드 (백업본 파일명)
    'best_fold3_epoch16.pt',  #4폴드 (새로 나올 파일명)
    'best_fold4_epoch12.pt'   #5폴드 (새로 나올 파일명)
]

all_fold_probs = []

model = timm.create_model(
    model_name,
    pretrained=False,  #추론시에는 False로 속도 향상
    num_classes=len(TARGET_CLASSES)
).to(device)

# 각 폴드별 Best 모델 로드 및 추론
for i, ckpt_name in enumerate(best_ckpt_files):
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    print(f'\n[Fold {i}] Loading: {ckpt_name}')

    load_checkpoint(ckpt_path, model, device)

    # 8-Way TTA 추론
    fold_probs = predict_proba(tst_loader, model, device)
    all_fold_probs.append(fold_probs)
    torch.cuda.empty_cache()

# 확률값 평균 내기 (Soft Voting)
if len(all_fold_probs) > 0:
    final_probs = np.mean(all_fold_probs, axis=0)
else:
    print('추론된 데이터가 없습니다. 체크포인트 파일명을 확인하세요!')

# 최종 라벨 결정 및 역매핑
final_pred_indices = final_probs.argmax(axis=1)
real_pred_list = [reverse_class_map[p] for p in final_pred_indices]

# 제출 파일 생성
submission_df = pd.read_csv(TEST_CSV)
submission_df['target'] = real_pred_list
save_path = f'{DATA_PATH}/output_convspec_recovery.csv'
submission_df.to_csv(save_path, index=False)

np.save(f'{DATA_PATH}/probs_convspec_recovery.npy', final_probs)
