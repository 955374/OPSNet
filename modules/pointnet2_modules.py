from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.ops import GroupAll
from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import QueryAndGroup, gather_points
from .object_highlighting import AdaptiveCross

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


def calc_square_dist(a, b, return_cos=False):
    """
    Calculating square distance between a and b
    a: [bs, c, n]
    b: [bs, c, m]
    """
    a = a.transpose(1, 2)
    b = b.transpose(1, 2)
    n = a.shape[1]
    m = b.shape[1]
    num_channel = a.shape[-1]
    a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
    b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
    a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
    b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
    a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
    b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

    coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

    if not return_cos:
        dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
    else:
        dist = coor / torch.sqrt(a_square) / torch.sqrt(b_square)
    return dist


class _PointnetSAModuleBase(nn.Module):
    def __init__(self, sample_method=None):
        super(_PointnetSAModuleBase, self).__init__()
        self.groupers = None
        self.mlps = None
        self.sample_method = sample_method
        self.cosine = nn.CosineSimilarity(dim=1)
        self.classify_mlp_phase2 = nn.Sequential(nn.Linear(128, 128), nn.Dropout(),
                                                 nn.Linear(128, 128), nn.Dropout(),
                                                 nn.Linear(128, 1)
                                                 )
        self.Ad_con_fuse = AdaptiveCross()

    def forward(self, xyz, features, npoint,
                idxs=None,
                cls_label=None,
                target_feature=None,
                template_points=None,
                keep_first_half=True):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """
        self.npoint = npoint
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if target_feature is not None:
            assert features is not None
            assert template_points is not None
            # print(features.shape, target_feature.shape, template_points.transpose(1,2).contiguous().shape, xyz.transpose(1,2).contiguous().shape)
            fusion_feature = self.Ad_con_fuse(features, target_feature,
                                                 template_points.transpose(1, 2).contiguous(),
                                                 xyz_flipped)
            fusion_feature = fusion_feature.transpose(1, 2).contiguous()
            score = self.classify_mlp_phase2(fusion_feature)

            max_score, max_top_k = torch.topk(score, self.npoint, largest=True, dim=1)
            target_idx = max_top_k.squeeze(-1).int()
        else:
            npoint = self.npoint

        if self.sample_method is None:
            if keep_first_half:
                idx0 = torch.arange(npoint // 2).repeat(
                    xyz.size(0), 1).int().cuda()
                idx1 = torch.arange(npoint - npoint // 2).repeat(
                    xyz.size(0), 1).int().cuda() + xyz.shape[1] // 2
                idx = torch.cat([idx0, idx1], dim=1)
            else:
                idx = torch.arange(npoint).repeat(xyz.size(0), 1).int().cuda()
            scores = None
        elif self.sample_method == 'fps':
            points_sampler_fps = Points_Sampler(fps_mod_list=['D-FPS'], fps_sample_range_list=[-1], num_point=[npoint])
            idx = points_sampler_fps(xyz, features=None)
            scores = None
        elif self.sample_method == 'ffps':
            points_sampler_ffps = Points_Sampler(fps_mod_list=['F-FPS'], fps_sample_range_list=[-1], num_point=[npoint])
            idx = points_sampler_ffps(xyz, features=None)
            scores = None
        elif self.sample_method == 'rand':
            idx = torch.randint(xyz.size(1), size=(xyz.size(1),)).repeat(xyz.size(0), 1).int().cuda()
            idx = idx[:,50:npoint+50]
            scores = None
        else:
            raise ValueError()


        if target_feature is not None:
            idx = target_idx
            scores = score
            features = fusion_feature.transpose(1, 2).contiguous()
            # print(features.shape)
        new_xyz = gather_points(xyz_flipped,
                                idx).transpose(1, 2).contiguous()

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), idx, scores


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, radii, nsamples, mlps, bn=True,
                 use_xyz=True, vote=False, sample_method=None):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__(sample_method=sample_method)
        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            grouper = QueryAndGroup(radius, nsample, use_xyz=use_xyz)
            if vote is False:
                self.groupers.append(grouper)

            mlp = nn.Sequential()
            for i in range(len(mlps[0]) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlps[0][i],
                        mlps[0][i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),

                        bias=True))
                self.mlps.append(mlp)


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self, mlp, radius=None, nsample=None,
            bn=True, use_xyz=True, sample_method=None
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            sample_method=sample_method,
        )
