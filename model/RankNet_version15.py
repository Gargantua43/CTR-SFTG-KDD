import torch
import torch.nn as nn
import torch.nn.functional as F


# import random
# import numpy as np
# from Dataset_pair_train import MyDatasetTrain
# from torch.utils.data import DataLoader


# 参数初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RankNet15(torch.nn.Module):
    def __init__(self):
        super(RankNet15, self).__init__()
        # Visual_FC
        self.fc_visual = torch.nn.Linear(1280, 128)  # ResNet50:2048 MobileNetV2:1280
        # Audio_FC
        self.fc_audio = torch.nn.Linear(128, 128)
        # Text_FC
        self.fc_text = torch.nn.Linear(768, 128)

        # LSTM
        self.lstm_visual = torch.nn.LSTM(128, 128, 1, batch_first=True)
        self.lstm_audio = torch.nn.LSTM(128, 128, 1, batch_first=True)
        self.lstm_text = torch.nn.LSTM(128, 128, 1, batch_first=True)

        # # Self-Attention
        # self.self_attention_visual = torch.nn.MultiheadAttention(embed_dim=128, num_heads=1, batch_first=True)
        # self.self_attention_audio = torch.nn.MultiheadAttention(embed_dim=128, num_heads=1, batch_first=True)

        # Dropout
        self.dropout = torch.nn.Dropout(p=0.5)

        # Attention
        self.fc_attention_visual = torch.nn.Linear(128, 1)  # ResNet50:2048 MobileNetV2:1280
        self.fc_attention_audio = torch.nn.Linear(128, 1)  # Same-Layer:128
        self.fc_attention_text = torch.nn.Linear(128, 1)
        # self.attention_sigmoid = torch.nn.Sigmoid()

        # Fusion_FC
        self.fusion_fc_ly1 = torch.nn.Linear(128, 64)
        self.fusion_fc_ly2 = torch.nn.Linear(64, 32)
        self.fusion_fc_ly3 = torch.nn.Linear(32, 16)
        self.fusion_fc_ly4 = torch.nn.Linear(16, 1)

        # Sigmoid
        self.calculate_sigmoid = torch.nn.Sigmoid()

        # # BatchNorm
        # self.batch_norm = nn.BatchNorm1d(num_features=128)

    def forward(self, x_visual1, x_audio1, x_text1, x_visual2, x_audio2, x_text2):
        """
        :param x_visual1:
        :param x_audio1:
        :param x_text1:
        :param x_visual2:
        :param x_audio2:
        :param x_text2:
        :return:
        """
        # # input_visual_embedding1
        # y_visual_fc1 = self.fc_visual(x_visual1)
        # y_visual_drop1 = F.relu(self.dropout(y_visual_fc1))
        #
        # # Visual_LSTM1
        # y_visual_lstm1, y_visual_ch1 = self.lstm_visual(y_visual_drop1)
        #
        # # Visual-Attention
        # # y_attention1_visual = self.attention_sigmoid(self.fc_attention_visual(x_visual1))
        # y_attention1_visual = self.fc_attention_visual(y_visual_lstm1)
        # y_weight1_visual = F.softmax(y_attention1_visual, dim=1)
        # y_weight_emb1_visual = torch.bmm(y_visual_lstm1.transpose(1, 2), y_weight1_visual).squeeze(2)
        #
        # # input_audio_embedding1
        # y_audio_fc1 = self.fc_audio(x_audio1)
        # y_audio_drop1 = F.relu(self.dropout(y_audio_fc1))
        #
        # # Audio_LSTM1
        # y_audio_lstm1, y_audio_ch1 = self.lstm_audio(y_audio_drop1)
        #
        # # Attention-Audio
        # # y_attention1_audio = self.attention_sigmoid(self.fc_attention_audio(x_audio1))
        # y_attention1_audio = self.fc_attention_audio(y_audio_lstm1)
        # y_weight1_audio = F.softmax(y_attention1_audio, dim=1)
        # y_weight_emb1_audio = torch.bmm(y_audio_lstm1.transpose(1, 2), y_weight1_audio).squeeze(2)

        # input_text_embedding1
        y_text_fc1 = self.fc_text(x_text1)
        y_text_drop1 = F.relu(self.dropout(y_text_fc1))

        # Text_LSTM1
        y_text_lstm1, y_text_ch1 = self.lstm_text(y_text_drop1)

        # Text-Attention
        y_attention1_text = self.fc_attention_text(y_text_lstm1)
        y_weight1_text = F.softmax(y_attention1_text, dim=1)
        y_weight_emb1_text = torch.bmm(y_text_lstm1.transpose(1, 2), y_weight1_text).squeeze(2)

        # Fusion
        y_ctr1_l1 = F.relu(self.fusion_fc_ly1(y_weight_emb1_text))
        y_ctr1_l2 = F.relu(self.fusion_fc_ly2(y_ctr1_l1))
        y_ctr1_l3 = F.relu(self.fusion_fc_ly3(y_ctr1_l2))
        y_ctr1 = self.fusion_fc_ly4(y_ctr1_l3)

        '''
        AD2
        '''
        # # input_visual_embedding2
        # y_visual_fc2 = self.fc_visual(x_visual2)
        # y_visual_drop2 = F.relu(self.dropout(y_visual_fc2))
        #
        # # Visual_LSTM2
        # y_visual_lstm2, y_visual_ch2 = self.lstm_visual(y_visual_drop2)
        #
        # # Visual-Attention
        # # y_attention2_visual = self.attention_sigmoid(self.fc_attention_visual(x_visual2))
        # y_attention2_visual = self.fc_attention_visual(y_visual_lstm2)
        # y_weight2_visual = F.softmax(y_attention2_visual, dim=1)
        # y_weight_emb2_visual = torch.bmm(y_visual_lstm2.transpose(1, 2), y_weight2_visual).squeeze(2)
        #
        # # input_audio_embedding2
        # y_audio_fc2 = self.fc_audio(x_audio2)
        # y_audio_drop2 = F.relu(self.dropout(y_audio_fc2))
        #
        # # Audio_LSTM2
        # y_audio_lstm2, y_audio_ch2 = self.lstm_audio(y_audio_drop2)
        #
        # # Audio-Attention
        # # y_attention2_audio = self.attention_sigmoid(self.fc_attention_audio(x_audio2))
        # y_attention2_audio = self.fc_attention_audio(y_audio_lstm2)
        # y_weight2_audio = F.softmax(y_attention2_audio, dim=1)
        # y_weight_emb2_audio = torch.bmm(y_audio_lstm2.transpose(1, 2), y_weight2_audio).squeeze(2)

        # input_text_embedding1
        y_text_fc2 = self.fc_text(x_text2)
        y_text_drop2 = F.relu(self.dropout(y_text_fc2))

        # Text_LSTM1
        y_text_lstm2, y_text_ch2 = self.lstm_text(y_text_drop2)

        # Text-Attention
        y_attention2_text = self.fc_attention_text(y_text_lstm2)
        y_weight2_text = F.softmax(y_attention2_text, dim=1)
        y_weight_emb2_text = torch.bmm(y_text_lstm2.transpose(1, 2), y_weight2_text).squeeze(2)

        # Fusion
        y_ctr2_l1 = F.relu(self.fusion_fc_ly1(y_weight_emb2_text))
        y_ctr2_l2 = F.relu(self.fusion_fc_ly2(y_ctr2_l1))
        y_ctr2_l3 = F.relu(self.fusion_fc_ly3(y_ctr2_l2))
        y_ctr2 = self.fusion_fc_ly4(y_ctr2_l3)

        # Predict CTR
        y_predict_value = self.calculate_sigmoid(y_ctr1 - y_ctr2)

        return y_predict_value
        # return y_predict_value, y_ctr1, y_ctr2

# '''
# Test
# '''
# x_visual_embedding1 = torch.randn(1, 40, 1280)
# x_visual_embedding2 = torch.randn(1, 40, 1280)
#
# x_audio_embedding1 = torch.randn(1, 40, 128)
# x_audio_embedding2 = torch.randn(1, 40, 128)
#
# x_text_embedding1 = torch.randn(1, 24, 768)
# x_text_embedding2 = torch.randn(1, 24, 768)
#
# model = RankNet()
# output = model(x_visual_embedding1, x_audio_embedding1, x_text_embedding1,
#                x_visual_embedding2, x_audio_embedding2, x_text_embedding2)
#
# print(output[0])
# print(output[0].shape)
# #
# # print(output[1])
# # print(output[1].shape)
# #
# # print(output[2])
# # print(output[2].shape)
#
# # print('--------')
# # print(output[0])
# # print(output[1])
# # print(output[2])
# # print(output.shape)
