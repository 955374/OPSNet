import torch
import torch.nn as nn
import torch.nn.functional as F

import etw_pytorch_utils as pt_utils
from .transformer import TransformerDecoder, TransformerEncoder, MultiheadAttention

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=128):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        # xyz : BxNx3
        # xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

class AdaptiveCross(nn.Module):
    def __init__(self):
        super(AdaptiveCross, self).__init__()

        self.cosine = nn.CosineSimilarity(dim=-1)
        self.coarse_pre_mlp = (pt_utils.Seq(128)
                               .conv1d(128, bn=True)
                               .conv1d(1, activation=None))
        self.con_mlp = (
            pt_utils.Seq(4 + 128 + 128)
            .conv1d(128, bn=True)
            .conv1d(128, bn=True)
            .conv1d(128, bn=True))
        self.dis_weight = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), bias=True),
            nn.Sigmoid()
        )
        self.dis_mlp = pt_utils.SharedMLP([3 + 128, 128, 128], bn=True)
        self.final_mlp = (pt_utils.Seq(128)
                          .conv1d(128, bn=True)
                          .conv1d(128, bn=True)
                          .conv1d(128, activation=None))
        self.sub_mlp = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True)
            .conv1d(128, bn=True)
            .conv1d(128, bn=True)
            .conv1d(128, activation=None))

        self.con_trans_mlp = (
            pt_utils.Seq(256)
            .conv1d(256, bn=True)
            .conv1d(256, bn=True)
            .conv1d(256, bn=True)
            .conv1d(128, activation=None))
        self.dis_trans_mlp = (
            pt_utils.Seq(256)
            .conv1d(256, bn=True)
            .conv1d(256, bn=True)
            .conv1d(256, bn=True)
            .conv1d(128, activation=None))
        self.fea_layer = (pt_utils.Seq(128)
                          .conv1d(128, bn=True)
                          .conv1d(128, activation=None))

        d_model = 128
        num_layers = 1
        multihead_attn = MultiheadAttention(
            feature_dim=d_model, n_head=1, key_feature_dim=128)
        encoder_pos_embed = PositionEmbeddingLearned(3, d_model)
        decoder_pos_embed = PositionEmbeddingLearned(3, d_model)

        # if self.with_pos_embed:
        #     encoder_pos_embed = PositionEmbeddingLearned(3, d_model)
        #     decoder_pos_embed = PositionEmbeddingLearned(3, d_model)
        # else:
        #     encoder_pos_embed = None
        #     decoder_pos_embed = None

        self.encoder = TransformerEncoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_encoder_layers=num_layers,
            self_posembed=encoder_pos_embed)
        self.decoder = TransformerDecoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_decoder_layers=num_layers,
            self_posembed=decoder_pos_embed)

    def transform_fuse(self, search_feature, search_xyz,
                       template_feature, template_xyz):
        """Use transformer to fuse feature.

        template_feature : BxCxN
        template_xyz : BxNx3
        """
        # BxCxN -> NxBxC
        search_feature = search_feature.permute(2, 0, 1)
        template_feature = template_feature.permute(2, 0, 1)

        num_img_train = search_feature.shape[0]
        num_img_template = template_feature.shape[0]

        ## encoder
        encoded_memory = self.encoder(template_feature,
            query_pos=template_xyz)

        encoded_feat = self.decoder(search_feature,
                                    memory=encoded_memory,
                                    query_pos=search_xyz)

        # NxBxC -> BxCxN
        encoded_feat = encoded_feat.permute(1, 2, 0)
        encoded_feat = self.fea_layer(encoded_feat)

        return encoded_feat

    def con_fea_fuse(self, temp_fea, con_fea, temp_xyz):
        # x_object template_fea
        # x_label search_fea
        B = temp_fea.size(0)
        f = temp_fea.size(1)
        n1 = temp_fea.size(2)
        n2 = con_fea.size(2)
        final_out_cla = self.cosine(temp_fea.unsqueeze(-1).expand(B, f, n1, n2),
                                    con_fea.unsqueeze(2).expand(B, f, n1, n2))
        final_out_cla_de = final_out_cla.detach()
        # print(temp_xyz.shape, temp_fea.shape)
        template_xyz_fea = torch.cat((temp_xyz, temp_fea), dim=1)
        max_ind = torch.argmax(final_out_cla_de, dim=1, keepdim=True).expand(-1, template_xyz_fea.size(1), -1)

        template_fea = template_xyz_fea.gather(dim=1, index=max_ind)
        max_cla = F.max_pool2d(final_out_cla.unsqueeze(dim=1), kernel_size=[final_out_cla.size(1), 1])
        max_cla = max_cla.squeeze(2)
        fusion_feature = torch.cat((max_cla, template_fea, con_fea), dim=1)
        con_feature = self.con_mlp(fusion_feature)
        return con_feature

    def dis_fea_fuse(self, temp_fea, dis_fea, temp_xyz):
        # if dis_fea.shape[0] == 0:
        # x_object template_fea
        # x_label search_fea

        B = temp_fea.size(0)
        f = temp_fea.size(1)
        n1 = temp_fea.size(2)
        n2 = dis_fea.size(2)
        diff_fea = dis_fea.unsqueeze(2).expand(B, f, n1, n2) - temp_fea.unsqueeze(-1).expand(B, f, n1, n2)
        dis_weight = self.dis_weight(diff_fea)
        dis_feature = (dis_weight * temp_fea.unsqueeze(-1).expand(B, f, n1, n2))
        dis_feature = torch.cat(
            (temp_xyz.unsqueeze(-1).expand(B, 3, n1, n2), dis_feature), dim=1)
        dis_feature = self.dis_mlp(dis_feature)
        dis_feature = F.max_pool2d(dis_feature, kernel_size=[dis_feature.size(2), 1])
        dis_feature = dis_feature.squeeze(2)
        return dis_feature

    def forward(self, search_fea, temp_fea, temp_xyz, search_xyz, isTrain=True):
        con_fea = self.con_trans_mlp(search_fea.permute(0, 2, 1)).permute(0, 2, 1)
        dis_fea = self.dis_trans_mlp(search_fea.permute(0, 2, 1)).permute(0, 2, 1)
        con_fuse_fea = self.con_fea_fuse(temp_fea, con_fea, temp_xyz)
        dis_fuse_fea = self.dis_fea_fuse(temp_fea, dis_fea, temp_xyz)
        fused_fea_out = torch.cat((con_fuse_fea, dis_fuse_fea), dim=2)
        fused_fea_out = self.final_mlp(fused_fea_out)
        diff_fea = self.sub_mlp(search_fea-fused_fea_out)
        fused_fea_out = self.transform_fuse(fused_fea_out, search_xyz, diff_fea, search_xyz)

        return fused_fea_out


if __name__ == '__main__':
    temp_xyz = torch.randn((32, 3, 128)).cuda()
    temp_fea = torch.randn((32, 128, 128)).cuda()
    search_xyz = torch.randn((32, 3, 256)).cuda()
    search_fea = torch.randn((32, 128, 256)).cuda()
    cls_label = torch.randint(0, 2, (32, 256)).cuda()
    contrast = AdaptiveCross()
    contrast.cuda()
    fused_fea_out, info_nce_loss = contrast(search_fea, temp_fea, cls_label, temp_xyz, search_xyz)
