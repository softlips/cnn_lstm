import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from config import DATA_DIR, CLASS_NAMES, SEQUENCE_LENGTH, BATCH_SIZE


class GestureDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_path = os.path.join(DATA_DIR, class_name)
            for sample_dir in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample_dir)
                if os.path.isdir(sample_path):
                    # 按帧号（数字）排序，确保顺序正确（如0.png,1.png...）
                    frames = sorted(
                        [f for f in os.listdir(sample_path) if f.endswith('.png')],
                        key=lambda x: int(x.split('.')[0])
                    )
                    # 采样：每4帧取1帧（论文逻辑，可调整步长）
                    sampled_frames = frames[::4]  # 步长4，如0,4,8...
                    if len(sampled_frames) >= SEQUENCE_LENGTH:
                        sampled_frames = sampled_frames[:SEQUENCE_LENGTH]  # 取前10帧
                        frame_paths = [os.path.join(sample_path, f) for f in sampled_frames]
                        samples.append((frame_paths, class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for path in frame_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
            if self.transform:
                img = transform(img)  # 应用预处理
            frames.append(img)
        frames = torch.stack(frames)  # 形状：(seq_len, 3, 299, 299)
        return frames, label

# 预处理（Inception-v3输入要求）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dataloaders(batch_size=BATCH_SIZE):
    dataset = GestureDataset(transform=transform)
    # 划分训练集（80%）和验证集（20%）， stratify确保类别平衡
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=42, stratify=[label for _, label in dataset.samples]
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader