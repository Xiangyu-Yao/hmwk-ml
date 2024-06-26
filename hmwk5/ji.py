import json
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 初始化模型
def init():
    model = torch.load("ev_sdk/src/model_1599.pkl",map_location=torch.device('cuda'))
    model.classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=1)  # 通道数修改为5，对应类别数
    model.load_state_dict(torch.load('/project/train/models/deeplabv3.pth',map_location=torch.device('cuda')))
    model.to(torch.device('cuda'))
    model.eval()
    return model

# 处理图像
def process_image(handle=None, input_image=None, args=None, **kwargs):
    args = json.loads(args)
    mask_output_path = args.get('mask_output_path', '')

    # 图像转换
    image_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True, p=1.0),
        ToTensorV2()
    ])

    # 将 BGR 转换为 RGB 并应用图像转换
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    image_array = np.array(input_image)
    input_tensor = image_transform(image=image_array)['image'].unsqueeze(0).to(torch.device('cuda'))

    # 生成掩码
    with torch.no_grad():
        output = handle(input_tensor)['out'][0]
        
    output_predictions = torch.argmax(output, dim=0)
    output_predictions = output_predictions.cpu().numpy().astype(np.uint8)
    
    height, width = input_image.shape[:2]

    transforms_mask=A.Compose([
        A.Resize(height,width),
    ])
    output_predictions= transforms_mask(image=output_predictions)['image']

    if input_image.shape[:2] != output_predictions.shape:
        raise ValueError(f"Mismatch in dimensions: Input image shape {input_image.shape[:2]}, Output predictions shape {output_predictions.shape}")
    # 保存掩码
    if mask_output_path:
        output_image = Image.fromarray(output_predictions)
        output_image.save(mask_output_path)

    # 解析输出，查找目标对象
    
    objects = []
    target_info = []
    is_alert = False

    # 查找垃圾对象
    garbage_mask = (output_predictions == 3).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(garbage_mask, connectivity=8)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        area_ratio = area / (width * height)

        if area_ratio > 0.02:  # 阈值为0.02
            target = {
                "x": int(x),
                "y": int(y),
                "height": int(h),
                "width": int(w),
                "name": "garbage",
                "area_ratio": float(area_ratio)
            }
            objects.append(target)
            target_info.append(target)
            is_alert = True

    result = {
        "algorithm_data": {
            "is_alert": is_alert,
            "target_count": len(target_info),
            "target_info": target_info
        },
        "model_data": {
            "objects": objects
        }
    }

    if mask_output_path:
        result["model_data"]["mask"] = mask_output_path

    return json.dumps(result, indent=4)
