import time
import os
import logging
import argparse
import random

import numpy as np
from tqdm import tqdm
import torch
import kitty_utils as utils
from metrics import AverageMeter, Success, Precision
from metrics import estimateOverlap, estimateAccuracy
from Dataset import SiameseTest
from torch.autograd import Variable
from OPS_tracking import Pointnet_Tracking
from decode import mot_decode
import matplotlib.pyplot as plt


def test(loader, model, epoch=-1, shape_aggregation="", reference_BB="", max_iter=-1, IoU_Space=3):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    post_process_time = AverageMeter()

    Success_main = Success()
    Precision_main = Precision()

    Success_batch = Success()
    Precision_batch = Precision()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    dataset = loader.dataset
    batch_num = 0
    tracklet_nums = 0
    track_anno_nums = 0
    key_track_anno_nums = 0
    with tqdm(enumerate(loader), total=len(loader.dataset.list_of_anno)) as t:
        for batch in loader:
            batch_num = batch_num + 1
            # measure data loading time
            data_time.update((time.time() - end))
            for PCs, BBs, list_of_anno in batch:  # tracklet
                results_BBs = []
                track_anno_nums += len(PCs)
                tracklet_nums += 1
                for i, _ in enumerate(PCs):
                    this_anno = list_of_anno[i]
                    this_BB = BBs[i]
                    this_PC = PCs[i]


                    new_PC = utils.cropPC(this_PC, this_BB, offset=2 * dataset.offset_BB, scale=4 * dataset.scale_BB)

                    new_label, align_gt_PC = utils.getlabelPC(new_PC, this_BB, offset=dataset.offset_BB,
                                                              scale=dataset.scale_BB)
                    # INITIAL FRAME
                    if i == 0:
                        data_time.update((time.time() - end))
                        end = time.time()
                        box = BBs[i]
                        results_BBs.append(box)
                        recall = torch.tensor(1.0).cuda()

                    else:
                        previous_BB = BBs[i - 1]

                        # DEFINE REFERENCE BB
                        if ("previous_result".upper() in reference_BB.upper()):
                            ref_BB = results_BBs[-1]
                        elif ("previous_gt".upper() in reference_BB.upper()):
                            ref_BB = previous_BB
                        elif ("current_gt".upper() in reference_BB.upper()):
                            ref_BB = this_BB

                        candidate_PC, candidate_label, candidate_reg, new_ref_box, new_this_box ,align_gt_PC= utils.cropAndCenterPC_label_test(
                            this_PC,
                            ref_BB, this_BB,
                            offset=dataset.offset_BB,
                            scale=dataset.scale_BB,
                            limit_area=dataset.area_extents)
                        candidate_PCs, candidate_labels, candidate_reg,align_gt_PC,all_search_label = utils.regularizePCwithlabel(candidate_PC,
                                                                                                     align_gt_PC,
                                                                                                     candidate_label,
                                                                                                     candidate_reg,
                                                                                                     dataset.input_size,
                                                                                                     istrain=False)

                        candidate_PCs_torch = candidate_PCs.unsqueeze(0).cuda()


                        # AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
                        if ("firstandprevious".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[0], PCs[i - 1]], [results_BBs[0], results_BBs[i - 1]],
                                                      offset=dataset.offset_BB, scale=dataset.scale_BB)
                        elif ("first".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[0]], [results_BBs[0]], offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)
                        elif ("previous".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[i - 1]], [results_BBs[i - 1]], offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)
                        elif ("all".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)
                        else:
                            model_PC = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)

                        model_PC_torch = utils.regularizePC(model_PC, dataset.input_size, istrain=False).unsqueeze(0)
                        model_PC_torch = Variable(model_PC_torch, requires_grad=False).cuda()
                        candidate_PCs_torch = Variable(candidate_PCs_torch, requires_grad=False).cuda()

                        data_time.update((time.time() - end))
                        end = time.time()
                        # (B,128) (B, 3, 128)，(B, 3+2, 64)，(B, 64, 3)
                        input_dict = {}
                        input_dict['search'] = candidate_PCs_torch
                        input_dict['template'] = model_PC_torch
                        pred_hm, pred_loc,pred_z_axis, search_dict = model(input_dict)

                        fea = search_dict['feature'].squeeze(0)
                        fea = fea.detach().cpu().numpy()
                        plt.scatter(fea[1,:],fea[3,:])
                        plt.show()


                        batch_time.update(time.time() - end)
                        end = time.time()
                        hm=pred_hm.sigmoid_()
                        xy_img_z_ry=mot_decode(hm,pred_loc,pred_z_axis,K=1)
                        xy_img_z_ry_cpu = xy_img_z_ry.squeeze(0).detach().cpu().numpy()
                        xy_img_z_ry_cpu[:,:2]=(xy_img_z_ry_cpu[:,:2]+dataset.min_img_coord)*dataset.xy_size
                        estimation_box_cpu=xy_img_z_ry_cpu[0]
                        box = utils.getOffsetBBtest(ref_BB, estimation_box_cpu[0:4])
                        results_BBs.append(box)

                    # estimate overlap/accuracy for current sample
                    this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=IoU_Space)
                    this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)

                    Success_main.add_overlap(this_overlap)
                    Precision_main.add_accuracy(this_accuracy)

                    Success_batch.add_overlap(this_overlap)
                    Precision_batch.add_accuracy(this_accuracy)
                    # measure elapsed time
                    post_process_time.update(time.time() - end)
                    end = time.time()

                    t.update(1)

                    if Success_main.count >= max_iter and max_iter >= 0:
                        return Success_main.average, Precision_main.average

                t.set_description('Test {}: '.format(epoch) +
                                  'forward {:.3f}s '.format(batch_time.sum) +
                                  '(it:{:.3f}s) '.format(batch_time.avg) +
                                  'pre:{:.3f}s '.format(data_time.sum) +
                                  '(it:{:.3f}s), '.format(data_time.avg) +
                                  '(post:{:.3f}s), '.format(post_process_time.sum) +
                                  '(it:{:.3f}s), '.format(post_process_time.avg) +
                                  'Succ/Prec:' +
                                  '{:.1f}/'.format(Success_main.average) +
                                  '{:.1f} '.format(Precision_main.average)
                                  )
                logging.info('track_id:{} '.format(this_anno["track_id"]) + 'Succ/Prec:' +
                             '{:.1f}/'.format(Success_batch.average) +
                             '{:.1f}'.format(Precision_batch.average)+
                             ' tracklet_frames:{}'.format(len(PCs)))
                Success_batch.reset()
                Precision_batch.reset()


    return Success_main.average, Precision_main.average


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--model_dir', type=str, default='./models/', help='output folder')
    parser.add_argument('--data_dir', type=str, default='/usr/training/',
                        help='dataset path')#/opt/data/common/nuScenes/KITTI_style/val /opt/data/common/kitti_tracking/kitti_t_o/training
    parser.add_argument('--model', type=str, default='car_model.pth', help='model name for training resume')
    parser.add_argument('--category_name', type=str, default='Car', help='Object to Track (Car/Pedestrian/Van/Cyclist)')
    parser.add_argument('--shape_aggregation', required=False, type=str, default='firstandprevious',
                        help='Aggregation of shapes (first/previous/firstandprevious/all)')
    parser.add_argument('--reference_BB', required=False, type=str, default="previous_result",
                        help='previous_result/previous_gt/current_gt')
    parser.add_argument('--model_fusion', required=False, type=str, default="pointcloud",
                        help='early or late fusion (pointcloud/latent/space)')
    parser.add_argument('--tiny', type=bool, default=1)
    parser.add_argument('--IoU_Space', required=False, type=int, default=3, help='IoUBox vs IoUBEV (2 vs 3)')
    parser.add_argument('--offset_BB', type=float, default=0.0)
    parser.add_argument('--scale_BB', type=float, default=1.25)
    parser.add_argument('--input_size', type=int, default=1024)
    opt = parser.parse_args()
    print(opt)


    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)

    dataset_Test = SiameseTest(
        input_size=opt.input_size,
        path=opt.data_dir,
        split='Test' if not opt.tiny else 'TinyTest',
        category_name=opt.category_name,
        offset_BB=opt.offset_BB,
        scale_BB=opt.scale_BB,
        voxel_size=[0.3, 0.3, 0.3],
        xy_size=[0.3, 0.3])

    test_loader = torch.utils.data.DataLoader(
        dataset_Test,
        collate_fn=lambda x: x,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    netR = Pointnet_Tracking()
    state_dict_ = torch.load(os.path.join(opt.model_dir, opt.model))
    print('loaded {}'.format(os.path.join(opt.model_dir, opt.model)))
    netR.load_state_dict(state_dict_)
    netR.cuda()
    torch.cuda.synchronize()


    Succ, Prec = test(
        test_loader,
        netR,
        epoch=1,
        shape_aggregation='firstandprevious',
        reference_BB='previous_result',
        IoU_Space=3

    )
    Success_run = AverageMeter()
    Precision_run = AverageMeter()
    Success_run.update(Succ)
    Precision_run.update(Prec)
    logging.info("mean Succ/Prec {}/{}".format(Success_run.avg, Precision_run.avg))
