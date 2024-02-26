import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Parameter Initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RankNetJapan(torch.nn.Module):
    def __init__(self):
        super(RankNetJapan, self).__init__()
        # Visual_FC
        self.fc_visual = torch.nn.Linear(2048, 128)
        self.bn_visual = torch.nn.BatchNorm1d(num_features=128)

        # Audio_FC
        self.fc_audio = torch.nn.Linear(128, 128)

        # Visual_Attention
        self.fc_attention_visual = torch.nn.Linear(128, 1)

        # Audio_Attention
        self.fc_attention_audio = torch.nn.Linear(128, 1)
        self.attention_sigmoid = torch.nn.Sigmoid()

        # Dropout
        self.dropout = torch.nn.Dropout(p=0.5)

        # LSTM
        self.lstm_audio = torch.nn.LSTM(128, 128, 1, batch_first=True)

        # MutilModel
        self.fc_mutil_attention_visual = torch.nn.Linear(128, 1)
        self.fc_mutil_attention_audio = torch.nn.Linear(128, 1)

        # Predict_CTR
        self.fc1_predict_ctr = torch.nn.Linear(128, 64)
        self.bn_predict_ctr = torch.nn.BatchNorm1d(num_features=64)
        self.fc2_predict_ctr = torch.nn.Linear(64, 1)
        self.calculate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x_visual1, x_audio1, x_visual2, x_audio2):
        """
        :param x_visual1:
        :param x_audio1:
        :param x_visual2:
        :param x_audio2:
        :return:
        """

        '''
        AD1
        '''
        # input_visual_embedding1
        y_visual_fc1 = self.fc_visual(x_visual1)
        y_visual_bn1 = (self.bn_visual(y_visual_fc1.permute(0, 2, 1))).permute(0, 2, 1)

        # Visual-Attention
        y_attention1_visual = F.softmax(self.fc_attention_visual(y_visual_bn1), dim=1)
        y_weight_emb1_visual = torch.sum(y_visual_bn1 * y_attention1_visual, dim=1)

        # input_audio_embedding1
        y_audio_fc1 = self.fc_audio(x_audio1)
        y_audio_drop1 = F.relu(self.dropout(y_audio_fc1))

        # Audio_LSTM1
        y_audio_lstm1, y_audio_ch1 = self.lstm_audio(y_audio_drop1)

        # Attention-Audio
        y_attention1_audio = self.attention_sigmoid(self.fc_attention_audio(x_audio1))
        y_weight1_audio = F.softmax(y_attention1_audio, dim=1)
        y_weight_emb1_audio = torch.bmm(y_audio_lstm1.transpose(1, 2), y_weight1_audio).squeeze(2)

        # MutilModel-Attention
        y_mutil_attention1_visual = self.fc_mutil_attention_visual(y_weight_emb1_visual)
        y_mutil_attention1_audio = self.fc_mutil_attention_audio(y_weight_emb1_audio)
        y_connect_emb1 = torch.cat([y_mutil_attention1_visual, y_mutil_attention1_audio], dim=1)
        y_mutil_weight1 = F.softmax(y_connect_emb1, dim=1)

        # Calculate norm
        y_visual1_norm_value = torch.norm(y_weight_emb1_visual, dim=1)
        y_audio1_norm_value = torch.norm(y_weight_emb1_audio, dim=1)

        # Divide By Norm
        y_visual1_norm = torch.div(y_weight_emb1_visual, y_visual1_norm_value.unsqueeze(1))
        y_audio1_norm = torch.div(y_weight_emb1_audio, y_audio1_norm_value.unsqueeze(1))

        # Multiply By Weight
        y_mutil_weight_emb1 = torch.add(y_visual1_norm * y_mutil_weight1[:, 0].unsqueeze(1),
                                        y_audio1_norm * y_mutil_weight1[:, 1].unsqueeze(1))

        # Predict_CTR
        y_ctr1_l1 = F.relu(self.fc1_predict_ctr(y_mutil_weight_emb1))
        y_ctr1_bn = self.bn_predict_ctr(y_ctr1_l1)
        y_ctr1_dropout = self.dropout(y_ctr1_bn)
        y_ctr1 = self.fc2_predict_ctr(y_ctr1_dropout)

        '''
        AD2
        '''
        # input_visual_embedding1
        y_visual_fc2 = self.fc_visual(x_visual2)
        y_visual_bn2 = (self.bn_visual(y_visual_fc2.permute(0, 2, 1))).permute(0, 2, 1)

        # Visual-Attention
        y_attention2_visual = F.softmax(self.fc_attention_visual(y_visual_bn2), dim=1)
        y_weight_emb2_visual = torch.sum(y_visual_bn2 * y_attention2_visual, dim=1)

        # input_audio_embedding1
        y_audio_fc2 = self.fc_audio(x_audio2)
        y_audio_drop2 = F.relu(self.dropout(y_audio_fc2))

        # Audio_LSTM1
        y_audio_lstm2, y_audio_ch2 = self.lstm_audio(y_audio_drop2)

        # Attention-Audio
        y_attention2_audio = self.attention_sigmoid(self.fc_attention_audio(x_audio2))
        y_weight2_audio = F.softmax(y_attention2_audio, dim=1)
        y_weight_emb2_audio = torch.bmm(y_audio_lstm2.transpose(1, 2), y_weight2_audio).squeeze(2)

        # MutilModel-Attention
        y_mutil_attention2_visual = self.fc_mutil_attention_visual(y_weight_emb2_visual)
        y_mutil_attention2_audio = self.fc_mutil_attention_audio(y_weight_emb2_audio)
        y_connect_emb2 = torch.cat([y_mutil_attention2_visual, y_mutil_attention2_audio], dim=1)
        y_mutil_weight2 = F.softmax(y_connect_emb2, dim=1)

        # Calculate norm
        y_visual2_norm_value = torch.norm(y_weight_emb2_visual, dim=1)
        y_audio2_norm_value = torch.norm(y_weight_emb2_audio, dim=1)

        # Divide By Norm
        y_visual2_norm = torch.div(y_weight_emb2_visual, y_visual2_norm_value.unsqueeze(1))
        y_audio2_norm = torch.div(y_weight_emb2_audio, y_audio2_norm_value.unsqueeze(1))

        # Multiply By Weight
        y_mutil_weight_emb2 = torch.add(y_visual2_norm * y_mutil_weight2[:, 0].unsqueeze(1),
                                        y_audio2_norm * y_mutil_weight2[:, 1].unsqueeze(1))

        # Predict_CTR
        y_ctr2_l1 = F.relu(self.fc1_predict_ctr(y_mutil_weight_emb2))
        y_ctr2_bn = self.bn_predict_ctr(y_ctr2_l1)
        y_ctr2_dropout = self.dropout(y_ctr2_bn)
        y_ctr2 = self.fc2_predict_ctr(y_ctr2_dropout)

        # Predict CTR
        y_predict_value = self.calculate_sigmoid(y_ctr1 - y_ctr2)

        return y_predict_value
        # return y_predict_value, y_ctr1, y_ctr2

# '''
# Test
# '''
# x_visual_embedding1 = torch.randn(3, 5, 2048)
# x_visual_embedding2 = torch.randn(3, 5, 2048)
# x_audio_embedding1 = torch.randn(3, 5, 128)
# x_audio_embedding2 = torch.randn(3, 5, 128)
#
# model = RankNetJapan()
# model.apply(init_weights)
# output = model(x_visual_embedding1, x_audio_embedding1, x_visual_embedding2, x_audio_embedding2)
#
# # print(output[0])
# # print(output[0].shape)
# #
# # print('=============')
# # print(output[1])
# # print(output[1].shape)
# #
# # print('+++++++++++++')
# # print(output[2])
# # print(output[2].shape)
# #
# # print('-------------')
# # print(output[3])
# # print(output[3].shape)
#
# print(output)
# print(output.shape)
