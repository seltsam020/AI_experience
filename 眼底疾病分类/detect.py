import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np

MODEL_DIR = "./models/"
MY_MODEL_NAME = "convnext.pth"
FORM = '.png'
TEST_DIR = './test/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size=224

valid_transforms = transforms.Compose([
    transforms.Resize([input_size, input_size]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_path):
    model = torch.load(MODEL_DIR + MY_MODEL_NAME,map_location=DEVICE)  # 直接加载模型
    model.eval()
    model = model.to(DEVICE)

    img_pil = Image.open(img_path)
    img_tensor = valid_transforms(img_pil)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(DEVICE)

    output = model(img_tensor)
    _, predict = torch.max(output, 1)

    return predict


list = os.listdir(TEST_DIR)
ans = []
for img in list:
    path = os.path.join(TEST_DIR, img)
    pred=predict(path)
    ans.append([img, int(pred)])


np.save('./pre.npy', ans)
test = np.load('./pre.npy')
print(test)