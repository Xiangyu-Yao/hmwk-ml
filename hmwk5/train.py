import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import logging
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import JaccardIndex
from tqdm import tqdm
from torch.cuda.amp import GradScaler,autocast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import shutil
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
os.environ['PYTHONUNBUFFERED'] = '1'

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.image_names = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path)
        image = image.convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image= np.array(image)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask


def evaluate_model(model, dataloader, criterion):
    model.eval()
    logger.info(f"val dataset's length is {len(dataloader)}")
    total_loss = 0
    total_miou = 0
    miou_metric = JaccardIndex(task="multiclass",num_classes=5, ignore_index=None).to(device)  # 定义mIoU指标

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False,file=sys.stdout):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # 计算miou
            preds = torch.argmax(outputs, dim=1)
            miou = miou_metric(preds, masks)
            total_miou += miou.item()
    avg_loss = total_loss / len(dataloader)
    avg_miou = total_miou / len(dataloader)

    return avg_loss, avg_miou


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    best_model_path,
    model_path,
    log_dir,
    num_epochs=5,
):
    # tensorboard记录器
    writer = SummaryWriter(log_dir=log_dir)
    scaler = GradScaler()
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_miou = 0

    # 模型训练
    logger.info("begin train")
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",file=sys.stdout):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]  # 获取当前的学习率

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, lr: {current_lr}"
        )
        writer.add_scalar("train_loss", avg_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)  # 记录学习率

        # 验证
        val_loss, val_miou = evaluate_model(model, test_loader, criterion)
        logger.info(f"Val Loss [{epoch+1}/{num_epochs}]: {val_loss}, mIoU: {val_miou}")
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("miou", val_miou, epoch)

        # 更新最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_miou = val_miou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"best val_loss's epoch={best_epoch}, best_val_Loss={best_val_loss}, best_mIoU={best_val_miou}"
            )

        # 更新学习率
        scheduler.step(val_loss)
        if epoch%5==0:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Epoch{epoch},save the models")
    
    writer.close()
    torch.save(model.state_dict(), model_path)
    logger.info("模型已保存")
    logger.info(
        f"best val_loss={best_val_loss}, best val_miou={best_val_miou}, bset epoch={best_epoch}"
    )


# 预测
def predict(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image, _ = transform(image, image)  # 这里第二个参数是dummy mask
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    return torch.argmax(output, dim=1).cpu().numpy()


# 数据预处理
def data_process(dir_path):
    data_path_1 = os.path.join(dir_path, "1704")
    data_path_2 = os.path.join(dir_path, "3288")
    merged_path = os.path.join(dir_path, "merged")
    train_path = os.path.join(dir_path, "train")
    val_path = os.path.join(dir_path, "val")

    # 创建文件夹
    if not os.path.exists(merged_path):
        os.makedirs(merged_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    # 合并数据
    for data_path in [data_path_1, data_path_2]:
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                dest_path = os.path.join(merged_path, file_name)
                try:
                    shutil.copy(file_path, dest_path)
                except Exception:
                    logger.info(f"Failed to copy file: {file_path}")

                # 检查文件是否成功复制，如果没有则重试
                if not os.path.exists(dest_path):
                    logger.info(f"Retry copying file: {file_path}")
                    try:
                        shutil.copy(file_path, dest_path)
                    except Exception:
                        logger.info(f"Failed to copy file on retry: {file_path}")

    # 获取merged文件夹中的所有文件名
    file_names = [f for f in os.listdir(merged_path) if f.endswith(".jpg")]

    # 计算划分比例
    train_ratio = 0.9
    num_train = int(len(file_names) * train_ratio)
    num_val = len(file_names) - num_train

    logger.info(f"Total files: {len(file_names)}, Train: {num_train}, Val: {num_val}")

    # 随机打乱文件名顺序
    random.shuffle(file_names)

    # 划分训练集和验证集
    for i, file_name in enumerate(file_names):
        img_path = os.path.join(merged_path, file_name)
        mask_name = file_name.replace(".jpg", ".png")
        mask_path = os.path.join(merged_path, mask_name)

        if os.path.exists(img_path) and os.path.exists(mask_path):
            if i < num_train:
                shutil.move(img_path, os.path.join(train_path, file_name))
                shutil.move(mask_path, os.path.join(train_path, mask_name))
            else:
                shutil.move(img_path, os.path.join(val_path, file_name))
                shutil.move(mask_path, os.path.join(val_path, mask_name))
        else:
            logger.info(f"File pair missing: {img_path}, {mask_path}")


# 颜色映射
def decode_segmap(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    label_colors = np.array(
        [
            (0, 0, 0),         # 背景
            (128, 0, 0),       # 水藻
            (0, 128, 0),       # 枯枝败叶
            (128, 128, 0),     # 垃圾
            (0, 0, 128)    # 水体
        ]
    )

    for label in range(0, 5):
        r[mask == label] = label_colors[label, 0]
        g[mask == label] = label_colors[label, 1]
        b[mask == label] = label_colors[label, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb / 255.0  # 保持归一化到 [0, 1]


# 结果可视化
def visualize_prediction(model, test_loader, device, epoch):
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            predictions = torch.argmax(outputs, dim=1)

            images = images.cpu().numpy()
            predictions = predictions.cpu().numpy()
            masks = masks.cpu().numpy()

            # 逆归一化操作
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            images = (images * std[None, :, None, None]) + mean[None, :, None, None]
            images = np.clip(images, 0, 1)  # 确保范围在 [0, 1]

            # 确保有至少3个样本进行可视化
            num_samples = min(images.shape[0], 3)
            for i in range(num_samples):
                fig, ax = plt.subplots(1, 3, figsize=(18, 6))

                ax[0].imshow(images[i].transpose(1, 2, 0))  # 原始图片
                ax[0].set_title("Original Image")
                ax[0].axis("off")

                ax[1].imshow(decode_segmap(predictions[i]))  # 预测的mask
                ax[1].set_title("Predicted Mask")
                ax[1].axis("off")

                ax[2].imshow(decode_segmap(masks[i]))  # 真实的mask
                ax[2].set_title("Ground Truth Mask")
                ax[2].axis("off")

                plt.savefig(
                    f"/project/train/result-graphs/prediction-epoch{i}.png"
                )
                plt.close(fig)  # 确保每次绘制后关闭图形


def main():
    batch_size = 32  # 根据显存大小调整
    num_workers = 8  # 根据 CPU 核心数调整
    num_epochs = 100 # 训练轮数

    # 数据增强
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.GaussianBlur(p=0.5),  # 添加高斯模糊
        A.RandomBrightnessContrast(p=0.5),  # 添加随机亮度对比度调整
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True, p=1.0),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True, p=1.0),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
    

    # 设置路径
    data_dir = "/home/data/"
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    model_path = "/project/train/models/deeplabv3.pth"
    log_dir = "/project/train/tensorboard/"
    best_model_path = "/project/train/models/deeplabv3_best.pth"

    # 加载预训练模型
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 5, kernel_size=1)  # 通道数修改为5，对应类别数
    # 加载训练最好的模型
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    # 数据预处理
    data_process(dir_path=data_dir)

    # 数据集加载
    train_dataset = SegmentationDataset(train_path, train_transform)
    test_dataset = SegmentationDataset(val_path,val_transform)  # test通常不做数据增强
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=True, num_workers=num_workers
    )

    # 损失函数，优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True,min_lr=1e-8)

    train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        best_model_path,
        model_path,
        log_dir,
        num_epochs,
    )

    model.load_state_dict(torch.load(best_model_path))
    
    # 调用预测和可视化函数
    visualize_prediction(model, test_loader, device,num_epochs+1)


logger.info("start")

main()