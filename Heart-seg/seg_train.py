import time
import os
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils import bladder
from utils.loss import *
#from utils import tools
from utils.metrics import diceCoeffv2
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from networks.u_net import *


crop_size = 128
batch_size = 2
n_epoch = 120
num_classes=4
model_name = 'U_Net'
loss_name = 'dice_'
times = 'no3_'
extra_description = ''
writer = SummaryWriter(os.path.join('log', 'bladder_exp', model_name + loss_name + times + extra_description))


def main():
    net = U_Net(img_ch=1, num_classes=4).cuda()

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(256),
        # joint_transforms.RandomRotate(10),
        # joint_transforms.RandomHorizontallyFlip()
    ])
    center_crop = joint_transforms.CenterCrop(crop_size)
    train_input_transform = extended_transforms.ImgToTensor()

    target_transform = extended_transforms.MaskToTensor()
    train_set = bladder.Bladder('./traindata', 'train',
                                joint_transform=train_joint_transform, center_crop=center_crop,
                                transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


    if loss_name == 'dice_':
        criterion = SoftDiceLoss(num_classes,activation='sigmoid').cuda()
    elif loss_name == 'bce_':
        criterion = nn.BCEWithLogitsLoss().cuda()
    #elif loss_name == 'wbce_':
    #    criterion = nn.WeightedBCELossWithSigmoid().cuda()
    #elif loss_name == 'er_':
    #    criterion = EdgeRefinementLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=5e-5)
    train(train_loader, net, criterion, optimizer, n_epoch, 0)


def train(train_loader, net, criterion, optimizer, num_epoches , iters):
    for epoch in range(1, num_epoches + 1):
        st = time.time()
        b_dice = 0.0
        t_dice = 0.0
        h_dice= 0.0
        th_dice =0.0
        d_len = 0
        for inputs, mask in train_loader:
            X = inputs.cuda()
            y = mask.cuda()
            optimizer.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            # CrossEntropyLoss
            # loss = criterion(output, torch.argmax(y, dim=1))
            output = torch.sigmoid(output)
            output[output < 0.5] = 0
            output[output > 0.5] = 1

            bladder_dice = diceCoeffv2(output[:, 0:1, :], y[:, 0:1, :], activation=None).cpu().item()
            tumor_dice = diceCoeffv2(output[:, 1:2, :], y[:, 1:2, :], activation=None).cpu().item()
            heart_dice = diceCoeffv2(output[:, 2:3, :], y[:, 2:3, :], activation=None).cpu().item()
            three_dice =diceCoeffv2(output[:, 3:4, :], y[:, 3:4, :], activation=None).cpu().item()
            mean_dice = (bladder_dice + tumor_dice+heart_dice+three_dice) / 4

            d_len += 1
            b_dice += bladder_dice
            t_dice += tumor_dice
            h_dice += heart_dice
            th_dice += three_dice

            loss.backward()
            optimizer.step()
            iters += batch_size
            string_print = "Epoch = %d iters = %d Current_Loss = %.4f Mean Dice=%.4f Bladder Dice=%.4f Tumor Dice=%.4f  heart Dice=%.4f Time = %.2f"\
                           % (epoch, iters, loss.item(), mean_dice,
                              bladder_dice, tumor_dice, heart_dice, time.time() - st)
            #tools.log(string_print)
            st = time.time()
            writer.add_scalar('train_main_loss', loss.item(), iters)
        b_dice = b_dice / d_len
        t_dice = t_dice / d_len
        h_dice = h_dice / d_len
        th_dice = th_dice / d_len
        m_dice = (b_dice + t_dice+h_dice+ th_dice) / 4
        print('Epoch {}/{},Train Mean Dice {:.4}, Bladder Dice {:.4}, Tumor Dice {:.4}, heart Dice {:.4}, three dice {:.4}'.format(
            epoch, num_epoches, m_dice, b_dice, t_dice , h_dice , th_dice
        ))
        if epoch == num_epoches:
            torch.save(net, './seg_models/{}.pth'.format(model_name + loss_name + times + extra_description))
            writer.close()

if __name__ == '__main__':
    main()
