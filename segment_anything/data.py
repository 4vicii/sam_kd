import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
import random

possible_transforms = [
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
]

# 创建一个函数，它每次都会返回一个随机的增强组合
def random_transforms():
    if random.random() < 0.1:  # 有10%的概率不应用任何增强
        return transforms.Compose([])  # 返回一个不执行任何操作的Compose对象
    num_transforms = random.randint(1, len(possible_transforms))
    chosen_transforms = random.sample(possible_transforms, num_transforms)
    return transforms.Compose(chosen_transforms)

class UnlabeledDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder)
                            if os.path.isfile(os.path.join(image_folder, f))
                            and f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        transform = random_transforms()
        try:
            image = Image.open(img_name)
            image = transform(image)
            image = np.asarray(image)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            if idx + 1 < len(self.image_files):  # 如果当前索引+1仍在列表内，返回下一张图片
                return self.__getitem__(idx + 1)
            else:
                return None

        return image

# Transformations (如需要的话)
# data_augmentation_transforms = transforms.Compose([
#     transforms.RandomRotation(15),                      # Random rotation
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Random crop and resize
#     transforms.RandomHorizontalFlip(),                   # Random horizontal flip
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color perturbations
#     transforms.ToTensor(),
# ])

# dataset = UnlabeledDataset(image_folder='path_to_your_image_folder', transform=data_augmentation_transforms)
#
# # dataloader
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
# # 在训练/蒸馏循环中使用DataLoader
# for images in dataloader:
#     # 使用images进行操作...
#     pass
