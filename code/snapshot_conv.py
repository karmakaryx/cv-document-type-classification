import glob
import os

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
if not DATA_PATH:
    raise ValueError('DATA_PATH 환경변수가 설정되지 않았습니다.')

TEST_CSV = os.path.join(DATA_PATH, 'sample_submission.csv')
TEST_DIR = os.path.join(DATA_PATH, 'test')
CHECKPOINT_DIR = os.path.join(DATA_PATH, 'checkpoints')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'convnextv2_base.fcmae_ft_in22k_in1k'
IMG_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 32
N_SPLITS = 5

class ImageDataset(Dataset):
    def __init__(self, data, path, transform=None):
        if isinstance(data, (str, os.PathLike)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        self.df = df.reset_index(drop=True)
        self.id = self.df['ID'].tolist()
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        name = self.id[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, 0

def predict_proba(loader, model, device):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for image, _ in tqdm(loader, desc='8-Way TTA Inferring'):
            image = image.to(device)
            batch_probs = []
            for k in range(4):
                rotated = torch.rot90(image, k=k, dims=[2, 3])
                logits = model(rotated)
                batch_probs.append(torch.softmax(logits, dim=1))

                flipped = torch.flip(rotated, dims=[3])
                logits_f = model(flipped)
                batch_probs.append(torch.softmax(logits_f, dim=1))

            avg_probs = torch.stack(batch_probs).mean(dim=0)
            probs_list.append(avg_probs.detach().cpu())
    return torch.cat(probs_list, dim=0).numpy()

# 스냅샷 복원
if __name__ == '__main__':
    test_df = pd.read_csv(TEST_CSV)

    # 원본 tst_transform
    tst_transform = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=(0,0,0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    tst_dataset = ImageDataset(test_df, TEST_DIR, transform=tst_transform)
    tst_loader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    total_test_probs = None

    for fold in range(N_SPLITS):
        print(f'\n[Fold {fold+1}] 스냅샷 탐색 중..')

        ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, f'best_fold{fold}_epoch*.pt'))
        ckpt_files = sorted(ckpt_files, key=lambda x: os.path.getmtime(x), reverse=True)[:3]

        if not ckpt_files:
            print(f'Fold {fold}에 해당하는 파일이 없습니다! 경로를 확인하세요.')
            continue

        print(f'선발된 스냅샷: {[os.path.basename(f) for f in ckpt_files]}')

        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=17,
            drop_path_rate=0.1
        ).to(DEVICE)

        fold_probs_list = []
        for ckpt_path in ckpt_files:
            print(f'로딩 중: {os.path.basename(ckpt_path)}')
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state['model_state_dict'])

            probs = predict_proba(tst_loader, model, DEVICE)
            fold_probs_list.append(probs)

        # Fold 내부 스냅샷 평균
        fold_mean_probs = np.mean(fold_probs_list, axis=0)

        if total_test_probs is None:
            total_test_probs = fold_mean_probs
        else:
            total_test_probs += fold_mean_probs

        del model
        torch.cuda.empty_cache()

    # K-Fold 전체 평균
    final_probs = total_test_probs / N_SPLITS
    final_preds = final_probs.argmax(axis=1)

    # 저장
    test_df['target'] = final_preds
    output_name = 'output_conv_recovery.csv'
    test_df.to_csv(os.path.join(DATA_PATH, output_name), index=False)
    np.save(os.path.join(DATA_PATH, 'probs_conv_recovery.npy'), final_probs)
