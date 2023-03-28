import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np

MODEL_DIR = "classify_models/"
MY_MODEL_NAME = "resnet18.pth"
FORM = '.png'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_type = {0: "-DCM", 1: "-HCM", 2: "-NOR"}
input_size=224

valid_transforms = transforms.Compose([
    transforms.Resize([input_size, input_size]),
    transforms.ToTensor(),
])

def predict(img_path):
    model = torch.load(MODEL_DIR + MY_MODEL_NAME,map_location=DEVICE)  # 直接加载模型
    model.eval()
    model = model.to(DEVICE)

    img_pil = Image.open(img_path)
    img_pil = img_pil.convert("RGB")
    img_tensor = valid_transforms(img_pil)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(DEVICE)

    output = model(img_tensor)
    _, predict = torch.max(output, 1)

    return predict


def classify(input_path):
    TEST_DIR =input_path+'/'
    # print(TEST_DIR)
    filenames = []
    for _, dirs, _ in os.walk(TEST_DIR):
        for dir in dirs:
            for _, _, files in os.walk(TEST_DIR + dir):
                for file in files:
                    filenames.append(TEST_DIR + dir + "/" + file)
    # print(filenames)
    ans = []
    for img in filenames:
        print("正在分类：",img)
        pred = predict(img)
        label_old = img.replace("Image", "Label-07").replace("image", "label")
        label_new = label_old.split(".png")[0] + class_type[int(pred)] + '.png'
        os.rename(label_old, label_new)  # 重命名
        ans.append([img, int(pred)])
    # print(ans)
    list_new = []
    for inf in ans:
        people = str(inf[0]).split('/')[-2]
        img_name = str(inf[0]).split('/')[-1]
        classes = ["DCM", 'HCM', 'NOR']
        list_new.append([people, img_name, classes[int(inf[1])]])
    print(list_new)
    np.save('./classify.npy', list_new)
    print("==========预测完成！！！==========")