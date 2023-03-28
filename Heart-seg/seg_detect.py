import os
import cv2
from PIL import Image
import utils.joint_transforms as joint_transforms
import seg_train
from utils import helpers
import utils.transforms as extended_transforms
from utils import bladder
from utils.loss import *

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import shutil

def val(input_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOSS = False
    # numpy 高维数组打印不显示...
    np.set_printoptions(threshold=9999999)
    batch_size = 1

    val_input_transform = extended_transforms.ImgToTensor()

    center_crop = joint_transforms.Compose([
        joint_transforms.Scale(256)]
    )

    target_transform = extended_transforms.MaskToTensor()
    # 验证用的模型名称
    model_name = seg_train.model_name
    loss_name = seg_train.loss_name
    times = seg_train.times
    extra_description = seg_train.extra_description
    model = torch.load("./seg_models/{}.pth".format(model_name + loss_name + times + extra_description),
                       map_location=DEVICE)
    model.eval()
    # 效果展示图片数
    # root="./testdata/image"
    output_path= input_path.split("/")
    output_path[-1]="Label-07"
    output_path='/'.join(output_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    print(output_path)

    patientid = os.listdir(input_path)
    print("输出位置：", output_path)
    # print(patientid)
    for j in patientid:
        for imname in os.listdir(os.path.join(input_path, j)):
            img = Image.open(os.path.join(input_path, j, imname))
            w, h = img.size
            img = img.resize((256, 256), Image.NEAREST)
            img = np.asarray(img)
            img = np.expand_dims(img, axis=2)
            img = img.transpose([2, 0, 1])
            img = np.expand_dims(img, axis=3)
            img = img.transpose([3, 0, 1, 2])
            img = val_input_transform(img)
            img = img.to(DEVICE)
            model = model.to(DEVICE)
            pred = model(img)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
            # 用来看预测的像素值
            pred = helpers.onehot_to_mask(pred, bladder.palette)
            pred = np.uint8(pred)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            if not os.path.exists(os.path.join(output_path, j)):
                os.mkdir(os.path.join(output_path, j))
            pred = Image.fromarray(pred)
            print("正在分割：", imname)
            imname=imname.replace("image","label")
            pred.save(os.path.join(output_path, j, imname))


if __name__ == '__main__':
    input_path = input("请输入图片路径：")
    val(input_path)
