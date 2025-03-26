import os
import random
import shutil

# 原始数据集目录和划分后的目标目录
original_images_dir = '/home/ldk/datasets/dataset_defect/images/train'
original_labels_dir = '/home/ldk/datasets/dataset_defect/labels/train'
base_dir = '/home/ldk/datasets/data'
os.makedirs(base_dir, exist_ok=True)

# 定义训练集和验证集的目录
train_images_dir = os.path.join(base_dir, 'train', 'images')
train_labels_dir = os.path.join(base_dir, 'train', 'labels')
val_images_dir = os.path.join(base_dir, 'val', 'images')
val_labels_dir = os.path.join(base_dir, 'val', 'labels')
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 获取所有图像文件名列表
all_images = os.listdir(original_images_dir)
random.shuffle(all_images)  # 随机打乱顺序

# 计算划分点，这里按照9:1的比例划分
split_point = int(0.9 * len(all_images))

# 分别将图像和标签复制到训练集和验证集目录
for i, image_name in enumerate(all_images):
    image_path = os.path.join(original_images_dir, image_name)
    label_name = image_name.replace('.jpg', '.txt')  # 假设标签文件与图像文件同名，后缀为.txt
    label_path = os.path.join(original_labels_dir, label_name)

    if i < split_point:
        shutil.copyfile(image_path, os.path.join(train_images_dir, image_name))
        shutil.copyfile(label_path, os.path.join(train_labels_dir, label_name))
    else:
        shutil.copyfile(image_path, os.path.join(val_images_dir, image_name))
        shutil.copyfile(label_path, os.path.join(val_labels_dir, label_name))

# 打印划分结果
print(f"Total images: {len(all_images)}")
print(f"Training images: {len(os.listdir(train_images_dir))}")
print(f"Validation images: {len(os.listdir(val_images_dir))}")
