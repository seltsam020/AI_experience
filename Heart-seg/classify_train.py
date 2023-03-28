from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
import pandas as pd
from pandas import Series
import cv2
import matplotlib
import copy
import time
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import warnings

matplotlib.use('TKAgg')
warnings.filterwarnings("ignore")
###################
#resnet18==>91.4%
####################
train_path = './Heart Data/'
# 图片格式
FORM = '.png'
CLASS=["Image_DCM","Image_HCM","Image_NOR"]
tclass={"Image_DCM":0,"Image_HCM":1,"Image_NOR":2}
filename=[]
label=[]

for c in CLASS:
    for _,dirs,_ in os.walk(train_path+c+"/png/Image/"):
        for dir in dirs:
            for _, _, files in os.walk(train_path + c + "/png/Image/"+dir):
                for file in files:
                    filename.append(train_path + c + "/png/Image/"+dir+"/"+file)
                    label.append(tclass[c])



# 从[resnet18, resnet34, resnet50]中选择模型

MODEL_NAME = "resnet18"
# 模型目录
MODEL_DIR = "classify_models/"

# 我训练的模型名称
MY_MODEL_NAME = MODEL_NAME + "_model"

# 图片尺寸
IMG_SIZE = 256

# 分类数 (健康:0 / 患病:>0)
NUM_CLASSES = 3

# 训练集比例 train / (train + valid）
RATIO = 0.8

# 训练的批量大小（根据内存量而变化）
BATCH_SIZE = 100

# 训练的 epoch 数
NUM_EPOCHS = 100

# worker数
NUM_WORKERS = 0

# 用于特征提取的标志       当为False时，我们微调整个模型，当True时我们只更新重新形成的图层参数
FEATURE_EXTRACT = False

# 检测是否有可用的GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id_code_np = np.array(filename)

diagnosis = np.array(label)

image_name = id_code_np


# 模型训练和验证函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    # 计算开始时间
    since = time.time()

    # 记录验证准确率，用于最后的plot分析
    val_acc_history = []

    # 记录所有的预测标签和真实标签，用于绘制Confusion matrix
    all_preds = torch.FloatTensor([]).cuda()
    all_labels = torch.FloatTensor([]).cuda()

    # 记录最佳模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Start training!")
    print('-' * 15 + '\n')

    for epoch in range(num_epochs):
        # optimizer.param_groups[0]["lr"] = 0.001/(1+0.01*epoch)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # print("lr:",str(optimizer.param_groups[0]["lr"]))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 参数梯度归零
                optimizer.zero_grad()

                # 前向传播
                # 如果在训练时则跟踪轨迹
                with torch.set_grad_enabled(phase == 'train'):
                    # 获取模型输出并计算损失
                    # 开始的特殊情况，因为在inception模型的训练中它有一个辅助输出
                    # 在训练模式下，我们通过将最终输出和辅助输出相加来计算损失
                    # 但在测试中我们只考虑最终输出
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 只在训练时反向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计loss和accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'valid' and epoch == (num_epochs - 1):
                    all_preds = torch.cat((all_preds, preds.float()), dim=0)
                    all_labels = torch.cat((all_labels, labels.float()), dim=0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}     Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 记录最佳模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()  # 空一行打印

    # 计算所用时间
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, all_preds, all_labels, best_acc


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 初始化将在此if语句中设置的这些变量。
    # 每个变量都是模型特定的。
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet34":
        """ Resnet34"""
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50"""
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet"""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet"""
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "convnext":
        """ convnext"""
        model_ft = models.convnext_tiny(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet":
        """ Densenet"""
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
         Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 处理辅助网络
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理主要网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("无效的模型名字，正在退出...")
        exit()

    return model_ft, input_size


model_ft, input_size = initialize_model(MODEL_NAME, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=False)
if os.path.exists(MODEL_DIR + 'resnet18.pth'):
    model_ft = torch.load(MODEL_DIR + 'resnet18.pth')
    print("loaded pretain!")
# 将模型发送到device
model_ft = model_ft.to(DEVICE)

# 收集要优化/更新的参数。如果正在进行微调，将更新所有参数；如果正在进行特征提取，只更新刚刚初始化的参数，即`requires_grad`参数为True。
params_to_update = model_ft.parameters()
#params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
# 定义优化器
optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9, weight_decay=0.001)
#optimizer_ft = optim.AdamW(params_to_update, lr=5e-4, weight_decay=5e-2)
"""
optimizer_ft = optim.Adam(params_to_update,
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
"""
print("optimizer定义完成！")

# 预处理
train_transforms = transforms.Compose([
    transforms.Resize([input_size, input_size]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),
])
valid_transforms = transforms.Compose([
    transforms.Resize([input_size, input_size]),
    transforms.ToTensor(),
])

# 图片处理整合成 loader
def img_loader(img_path, transforms):
    img_pil = Image.open(img_path)
    img_pil = img_pil.convert("RGB")

    img_tensor = transforms(img_pil)

    return img_tensor


class RetinaImageDataset(Dataset):
    """
    Args:
        img_path (string): 图像文件路径
        transforms: transform 操作
        loader: 单张图片文件处理(打开图片，预处理)
    """

    def __init__(self, img_path, diagnosis, transforms=None, loader=img_loader):
        # 图片的文件名
        self.img_path = img_path
        # 图片的标签（0~4）
        self.diagnosis = diagnosis
        # transform
        self.transforms = transforms
        # 单张图片文件处理
        self.loader = loader
        # 计算 length
        self.data_len = len(self.img_path)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # 得到单张图片路径
        single_img_path = self.img_path[index]
        # 单张图片文件处理 (返回图片为tensor类型)
        single_image = self.loader(single_img_path, self.transforms)
        # 得到单张图片的 label

        single_diagnosis = self.diagnosis[index]

        return (single_image, single_diagnosis)


# 初始化模型
sss = ShuffleSplit(n_splits=1, test_size=1-RATIO, random_state=42)
indices = range(len(filename))
for train_index, val_index in sss.split(indices):
    image_name_train = image_name[train_index]
    image_name_valid = image_name[val_index]

    diagnosis_train = diagnosis[train_index]
    diagnosis_valid = diagnosis[val_index]

    # 类别数量对比
    print("训练集图片数: {}   训练集标签数: {} \n验证集图片数: {}    验证集标签数: {}".format(
        len(image_name_train), len(diagnosis_train), len(image_name_valid), len(diagnosis_valid)))

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()

    train_dataset = RetinaImageDataset(image_name_train, diagnosis_train, transforms=train_transforms)

    valid_dataset = RetinaImageDataset(image_name_valid, diagnosis_valid, transforms=valid_transforms)

    image_datasets = {'train': train_dataset, 'valid': valid_dataset}

    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
                        for x in ['train', 'valid']}

    model_ft, hist, all_preds, all_labels, best_x = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                                                num_epochs=NUM_EPOCHS,
                                                                is_inception=(MODEL_NAME == "inception"))
    torch.save(model_ft, "./seg_models/"+MODEL_NAME+'_3.pth')  # 直接保存模型

    ohist = [h.cpu().numpy() for h in hist]
    
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, NUM_EPOCHS + 1), ohist)
    plt.ylim((0.2, 1.0))
    plt.xticks(np.arange(1, NUM_EPOCHS + 1, 1.0))
    plt.legend()
    plt.show()