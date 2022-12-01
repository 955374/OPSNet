from Dataset import SiameseTrain
import torch
import argparse
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn as nn
import torch.optim as optim
from OPS_tracking import Pointnet_Tracking
import time
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
import os
from loss.losses import FocalLoss, RegL1Loss
from loss.utils import _sigmoid

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=160, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--data_dir', type=str, default='/usr/training', help='dataset path')
parser.add_argument('--category_name', type=str, default='Car', help='Object to Track (Car/Pedestrian/Van/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='./models/', help='output folder')
parser.add_argument('--model', type=str, default='', help='model name for training resume')
parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')
parser.add_argument('--tiny', type=bool, default=True)
parser.add_argument('--offset_BB', type=float, default=0.0)
parser.add_argument('--scale_BB', type=float, default=1.25)
parser.add_argument('--input_size', type=int, default=1024)
opt = parser.parse_args()

train_data = SiameseTrain(
    input_size=opt.input_size,
    path=opt.data_dir,
    split='Train' if not opt.tiny else 'TinyTrain',
    category_name=opt.category_name,
    offset_BB=opt.offset_BB,
    scale_BB=opt.scale_BB,
    voxel_size=[0.3, 0.3, 0.3],
    xy_size=[0.3, 0.3])
train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=opt.workers,
    pin_memory=True)


def train_label_generator(input_dict, search_output_dict):
    idxs = search_output_dict['idxs']
    b = idxs[0].shape[0]
    idxs_list = defaultdict(list)
    for i in range(len(idxs)):
        for bi in range(b):
            if i == 0:
                idxs_list[i].append(idxs[0][bi])
            else:
                idxs_list[i].append(idxs_list[i - 1][bi][idxs[i][bi].long()])
    final_idx = torch.stack(idxs_list[len(idxs) - 1], dim=0)  # 128
    mid_idx = torch.stack(idxs_list[len(idxs) - 2], dim=0)  # 256
    first_idx = torch.stack(idxs_list[len(idxs) - 3], dim=0)  # 512

    cls_label = [torch.gather(input_dict['search_all_label'], 1, first_idx.long()),
                 torch.gather(input_dict['search_all_label'], 1, mid_idx.long()),  # 256
                 torch.gather(input_dict['search_all_label'], 1, final_idx.long())]  # 128
    return cls_label


# model initialization
netR = Pointnet_Tracking()
if opt.model != '':
    netR.load_state_dict(torch.load(os.path.join(opt.save_root_dir, opt.model)))
netR = netR.cuda()
optimizer = optim.Adam(netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)
scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
# loss defination
criterion_hm = FocalLoss().cuda()
criterion_loc = RegL1Loss().cuda()
criterion_z_axis = RegL1Loss().cuda()
criterion_cls = nn.BCEWithLogitsLoss()
criterion_fuse = FocalLoss().cuda()

# memory bank initialization


for epoch in range(opt.nepoch):
    print('------------------ epoch : %d ------------------' % epoch)

    torch.cuda.synchronize()
    netR.train()
    train_mse = 0.0
    timer = time.time()

    batch_comp_part_loss = 0.0
    batch_hm_loss = 0.0
    batch_loc_loss = 0.0
    batch_z_loss = 0.0
    batch_num = 0.0
    for i, input_dict in enumerate(tqdm(train_dataloader, 0)):
        if len(input_dict['search']) == 1:
            continue

        for k, v in input_dict.items():
            input_dict[k] = Variable(v, requires_grad=False).cuda()
        # print(input_dict['template'].reshape(-1, 3).shape)

        _, input_size, _ = input_dict['template'].shape

        # print(input_dict['reg_label'].shape)
        optimizer.zero_grad()

        pred_hm, pred_loc, pred_z_axis, search_dict = netR(input_dict)

        pred_score = search_dict['score'][-1].squeeze(-1)
        cls_label = train_label_generator(input_dict, search_dict)[1]
        cls_label = cls_label * 0.95 + (1 - cls_label) * 0.05 / cls_label.shape[1]
        loss_cls = criterion_cls(pred_score, cls_label)
        labels = train_label_generator(input_dict, search_dict)

        pred_hm = _sigmoid(pred_hm)
        loss_hm = criterion_hm(pred_hm, input_dict['hm'])
        loss_loc = criterion_loc(pred_loc, input_dict['ind_offsets'], input_dict['loc_reg'])
        loss_z = criterion_z_axis(pred_z_axis, input_dict['ind_ct'], input_dict['z_axis'])

        loss = 1.0 * loss_hm + 1.0 * loss_loc + 2.0 * loss_z + 1.0 * loss_cls
        # loss = 0.1 * loss_prev + 0.5 * loss_final

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        train_mse = train_mse + loss.data * len(input_dict['search'])
        batch_hm_loss += loss_hm.data
        batch_loc_loss += loss_loc.data
        batch_z_loss += loss_z.data
        batch_num += len(input_dict['search'])
        if (i + 1) % 20 == 0:
            print('\n ---- batch: %03d ----' % (i + 1))
            print('hm_loss: %f,loc_loss: %f,z_loss: %f' % (
                batch_hm_loss / 20, batch_loc_loss / 20, batch_z_loss / 20))
            # print('pos_points:{} neg_points:{}'.format(num_pos,num_neg))
            batch_hm_loss = 0.0
            batch_loc_loss = 0.0
            batch_z_loss = 0.0
            batch_num = 0.0
    torch.save(netR.state_dict(), '%s/%s_%g_%d.pth' % (opt.save_root_dir, opt.category_name, opt.learning_rate, epoch))
