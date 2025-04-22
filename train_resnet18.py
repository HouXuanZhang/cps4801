import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import time
from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision.models import resnet18

# 训练设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 准备数据集
# train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
# test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 导入归一化所需均值和标准差
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# 数据增强用于训练集
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),     # 随机裁剪
    torchvision.transforms.RandomHorizontalFlip(),        # 随机水平翻转
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)           # 标准化
])

# 测试集不需要增强，只需要归一化
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

# 应用新 transform
train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True,
    transform=transform_train, 
    download=True)

test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False,
    transform=transform_test, 
    download=True)



train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"train data size is: {train_data_size}")
print(f"test data size is: {test_data_size}")

# 加载 ResNet18 模型，并修改全连接层适配 CIFAR10
resnet18_model = resnet18(pretrained=False)#来自的是 torchvision.models 提供的官方预训练模型
resnet18_model.fc = nn.Linear(512, 10)
resnet18_model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
optimizer = torch.optim.SGD(resnet18_model.parameters(), lr=1e-2)

# 日志与训练参数
writer = SummaryWriter("./logs_resnet18")
total_train_step = 0
total_test_step = 0
epoch = 20
start_time = time.time()

# 创建保存目录
os.makedirs("model_save", exist_ok=True)

# 训练开始
for i in range(epoch):
    print(f"----The {i+1} round of training begins----")
    resnet18_model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = resnet18_model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            duration = time.time() - start_time
            print(f"{duration:.2f}s - Training times:{total_train_step}, loss:{loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试开始
    resnet18_model.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = resnet18_model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    print(f"The Loss on the overall test set: {total_test_loss:.4f}")
    print(f"The accuracy rate on the overall test set: {total_accuracy / test_data_size:.4f}")

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(resnet18_model, f"model_save/resnet18_epoch{i}.pth")
    print("The model has been saved.\n")

writer.close()
