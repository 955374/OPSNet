from modules.pointnet2_modules import PointnetSAModule
import torch
from torch import nn as nn
from mmcv.ops import QueryAndGroup
from modules.voxel_utils.voxel.voxelnet import Conv_Middle_layers
from modules.voxel_utils.voxelization import Voxelization
from modules.RPN import RPN
from visual_utils.open3d_vis_utils import plt_scenes


class Pointnet_Backbone(nn.Module):
    def __init__(self, input_channels=3,
                 use_xyz=True, sample_method='fps',
                 first_sample_method=None):
        super(Pointnet_Backbone, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
                sample_method='fps'
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.5,
                nsample=32,
                mlp=[128, 128, 128, 128],
                use_xyz=False,  # False,
                sample_method='fps'
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[128, 128, 128, 128],
                use_xyz=False,  # False,
                sample_method=None,
            )
        )
        self.cov_final = nn.Conv1d(128, 128, kernel_size=1)
        self.sample_method = sample_method


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints,
                target_features=None,
                template_points=None,
                cls_label=None,
                keep_first_half=False):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        l_score = []

        for i in range(len(self.SA_modules)):
            if i > 1 and target_features is not None:
                assert not keep_first_half
                li_xyz, li_features, idx, score = self.SA_modules[i](
                    l_xyz[i], l_features[i], numpoints[i], idxs, cls_label,
                    # target_feature=None, template_points=None)
                    target_feature=target_features[i], template_points=template_points[i])


            else:
                li_xyz, li_features, idx, score = self.SA_modules[i](
                    l_xyz[i], l_features[i], numpoints[i],
                    keep_first_half=keep_first_half)

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_score.append(score)
            idxs.append(idx)
        output_dict = {
            'xyz': l_xyz[-1],
            'feature': self.cov_final(l_features[-1]),
            'idxs': idxs,
            'xyzs': l_xyz,
            'features': l_features,
            'score': l_score
        }

        return output_dict


class Pointnet_Tracking(nn.Module):
    def __init__(self, input_channels=3, use_xyz=True, objective=False):
        super(Pointnet_Tracking, self).__init__()

        self.backbone = Pointnet_Backbone()
        self.vote_aggregation = PointnetSAModule(
            radius=0.3,
            nsample=16,
            mlp=[128, 128, 128, 128],
            use_xyz=False,  # False,
            sample_method='ffps',
        )
        self.FC_layer_cla = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 1))
        self.vote_layer = nn.Sequential(
            nn.Linear(3 + 128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 3 + 128)
        )
        self.group5 = QueryAndGroup(1.0, 8, use_xyz=use_xyz)
        self.xyz_proposal = nn.Sequential(
            nn.Linear(128 + 1 + 128 + 3 + 128 + 3, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128))
        self.voxelize = Voxelization(38, 24, 18,
                                     scene_ground=torch.tensor([-5.6000, -3.6000, -2.4000]), mode=False,
                                     voxel_size=torch.tensor([0.3000, 0.3000, 0.3000]))
        self.cml = Conv_Middle_layers(inplanes=3 + 128)
        self.RPN = RPN()


    def forward(self, input_dict):
        _, input_size, _ = input_dict['template'].shape
        output_dict = {}
        # print(input_dict['reg_label'].shape)
        template_output_dict = \
            self.backbone(pointcloud=input_dict['template'], numpoints=[input_size // 2,
                                                                        input_size // 4,
                                                                        input_size // 8],
                          keep_first_half=False)

        template_idx = template_output_dict['idxs']
        template_features = template_output_dict['features']
        template_points = template_output_dict['xyzs']

        _, input_size, _ = input_dict['search'].shape
        search_output_dict = \
            self.backbone(pointcloud=input_dict['search'], numpoints=[input_size // 2,
                                                                      input_size // 4,
                                                                      input_size // 8],
                          target_features=template_features,
                          template_points=template_points,
                          cls_label=input_dict['cls_label'] if 'cls_label' in input_dict.keys() else None,
                          keep_first_half=False)

        batch_size, _, _ = input_dict['search'].shape

        fusion_features = search_output_dict['features'][-1]
        search_xyz = search_output_dict['xyz']



        estimation_cla = self.FC_layer_cla(fusion_features.permute(0, 2, 1).contiguous())
        score = estimation_cla.sigmoid().permute(0, 2, 1).contiguous()

        fusion_xyz_feature = torch.cat(
            (search_xyz.transpose(1, 2).contiguous(), fusion_features),
            dim=1).permute(0, 2, 1).contiguous()
        #
        offset_feature = self.vote_layer(fusion_xyz_feature)
        offset = offset_feature[:, :, :3]
        fusion_features = fusion_features + offset_feature[:, :, 3:].permute(0, 2, 1).contiguous()
        temp_selection = search_output_dict['xyz'] - offset

        temp_pooling_feature = self.group5(
            template_output_dict['xyzs'][-1], temp_selection, template_output_dict['features'][-1])
        temp_pooling_feature, _ = torch.max(temp_pooling_feature, dim=-1, keepdim=False)
        pooling_feature = temp_pooling_feature

        search_pooling_feature = self.group5(
            search_output_dict['xyzs'][-1], search_output_dict['xyz'], search_output_dict['features'][-1])
        search_pooling_feature, _ = torch.max(search_pooling_feature, dim=-1, keepdim=False)
        search_pooling_feature = search_pooling_feature

        pooling_feature = torch.cat([pooling_feature, search_pooling_feature], dim=1)
        proposal_features = torch.cat([score, pooling_feature, fusion_features], dim=1).permute(0, 2, 1).contiguous()
        proposal_offsets = self.xyz_proposal(proposal_features)

        proposal_offsets = torch.cat((proposal_offsets, search_xyz), dim=2)

        proposal_offsets_voxel = self.voxelize(proposal_offsets.transpose(1,2), search_xyz)
        proposal_offsets_voxel = proposal_offsets_voxel.permute(0, 1, 4, 3, 2).contiguous()
        cml_out = self.cml(proposal_offsets_voxel)
        pred_hm, pred_loc, pred_z_axis = self.RPN(cml_out)
        return pred_hm, pred_loc, pred_z_axis, search_output_dict

