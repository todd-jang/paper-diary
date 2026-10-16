"""data.py — MAE용 데이터 파이프라인 (논문과 동일: RandomResizedCrop + HFlip, 정규화 없음)"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def build_transform(img_size=224):
    # MAE official: 픽셀을 [0,1]로만 변환, mean/std 정규화 없음 (손실에서 per-patch 정규화)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def build_train_loader(root, img_size=224, batch_size=64, num_workers=4):
    tf = build_transform(img_size)
    ds = datasets.ImageFolder(root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=False, drop_last=True), ds


def build_eval_transform(img_size=224, resize=256):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def build_eval_loader(root, img_size=224, batch_size=128, num_workers=4):
    tf = build_eval_transform(img_size)
    ds = datasets.ImageFolder(root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers), ds
