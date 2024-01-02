import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
# from dataset.dataset import  DatasetLoad
from FPNet import FPNet
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置训练集、验证集、测试集路径
base_path = os.path.join(os.path.dirname(__file__))
train_cover_path = os.path.join(base_path, "train","cover")
train_stego_path = os.path.join(base_path, "train", "stego")
val_cover_path = os.path.join(base_path, "val", "cover")
val_stego_path = os.path.join(base_path, "val", "stego")
test_cover_path = os.path.join(base_path, "test", "cover")
test_stego_path = os.path.join(base_path, "test", "stego")

# 设置参数
train_size = 0.6  # 训练集大小比例
val_size = 0.3    # 验证集大小比例
test_size = 0.1 

train_epoch = 10  # 训练轮数


# 数据加载
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import os
from typing import Tuple

class DatasetLoad(Dataset):
    """This class returns the data samples."""

    def __init__(
        self,
        cover_path: str,
        stego_path: str,
        size: int = None,
        transform: Tuple = None,
    ) -> None:
        """Constructor.

        Args:
            cover_path (str): path to cover images.
            stego_path (str): path to stego images.
            size (int): no. of images in any of (cover / stego) directory for
              training.
            transform (Tuple, optional): _description_. Defaults to None.
        """
        self.cover = cover_path
        self.stego = stego_path
        self.transforms = transform
        self.data_size = size
        if self.data_size == None:
            self.data_size = len(os.listdir(self.cover))

    def __len__(self) -> int:
        """returns the length of the dataset."""
        return int(self.data_size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the (cover, stego) pairs for training.

        Args:
            index (int): a random int value in range (0, len(dataset)).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cover and stego pair.
        """
        index += 1
        img_name = str(index) + ".pgm"
        cover_img = Image.open(os.path.join(self.cover, img_name))
        stego_img = Image.open(os.path.join(self.stego, img_name))

        label1 = torch.tensor(0, dtype=torch.long).to(device)
        label2 = torch.tensor(1, dtype=torch.long).to(device)

        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)
        else:
            cover_img = ToTensor()(cover_img)
            stego_img = ToTensor()(stego_img)
        
        sample = {"cover": cover_img, "stego": stego_img}
        sample["label"] = [label1, label2]

        return sample

train_data = DatasetLoad(
    train_cover_path,
    train_stego_path,
    transform=transforms.Compose([
        transforms.RandomRotation(degrees=90),
        # 使用transforms.ToTensor()将PIL图像转换为Tensor
        transforms.ToTensor(),
    ]),
)

val_data = DatasetLoad(
    val_cover_path,
    val_stego_path,
    transform=transforms.ToTensor(),
)

test_data = DatasetLoad(
    test_cover_path,
    test_stego_path,
    transform=transforms.ToTensor(),
)

print("train_data:", len(train_data))
print("val_data:", len(val_data))
print("test_data:", len(test_data))


train_loader = DataLoader(train_data, batch_size=2, shuffle=True)  # 批大小
valid_loader = DataLoader(val_data, batch_size=2, shuffle=False)  # 批大小
test_loader = DataLoader(test_data, batch_size=2, shuffle=False)  # 批大小

print("train_loader:", len(train_loader))
print("valid_loader:", len(valid_loader))
print("test_loader:", len(test_loader))

# 设置模型
model = FPNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 学习率
criterion = nn.CrossEntropyLoss()

print("开始训练。。。")
# 训练循环
for epoch in range(train_epoch):  
    print(f"Epoch: {epoch+1}/{train_epoch}")
    model.train()
    for i, batch in enumerate(train_loader):
        print(
            f"Epoch: {epoch+1}/{train_epoch}, Batch: {i+1}/{len(train_loader)}")
        images = torch.cat((batch["cover"], batch["stego"]), 0).to(device)
        labels = torch.cat((batch["label"][0], batch["label"][1]), 0).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}/{train_epoch}, Batch: {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # 在验证集上评估性能
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in valid_loader:
            images = torch.cat((batch["cover"], batch["stego"]), 0).to(device)
            labels = torch.cat((batch["label"][0], batch["label"][1]), 0).to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy after Epoch {epoch+1}: {accuracy:.2%}")

# 保存模型参数
torch.save(model.state_dict(), 'model/fpnet_model.pth')
print("模型参数已保存")
