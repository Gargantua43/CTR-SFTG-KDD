import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# from torch.utils.data import DataLoader
# from tqdm import tqdm


class MyDatasetTest(Dataset):
    def __init__(self, pair_csv_path):
        self.Pair_csv_path = pair_csv_path
        self.df_Pair = pd.read_csv(self.Pair_csv_path, encoding='utf-8')

    def __getitem__(self, index):
        # Visual1
        visual_csv_dir1 = self.df_Pair['Test_CTR_visual_pair1'][index]
        visual_data1 = np.loadtxt(visual_csv_dir1, delimiter=',')
        visual_train1 = torch.from_numpy(visual_data1[:, :])

        # Audio1
        audio_csv_dir1 = self.df_Pair['Test_CTR_audio_pair1'][index]
        audio_data1 = np.loadtxt(audio_csv_dir1, delimiter=',')
        audio_train1 = torch.from_numpy(audio_data1[:, :])

        # Visual2
        visual_csv_dir2 = self.df_Pair['Test_CTR_visual_pair2'][index]
        visual_data2 = np.loadtxt(visual_csv_dir2, delimiter=',')
        visual_train2 = torch.from_numpy(visual_data2[:, :])

        # Audio2
        audio_csv_dir2 = self.df_Pair['Test_CTR_audio_pair2'][index]
        audio_data2 = np.loadtxt(audio_csv_dir2, delimiter=',')
        audio_train2 = torch.from_numpy(audio_data2[:, :])

        # label
        pair_label = self.df_Pair['Test_CTR_Label'][index]
        label_data = np.array(pair_label)
        label_train = torch.from_numpy(label_data)

        return visual_train1, audio_train1, visual_train2, audio_train2, label_train
        # return self.train_data[index]

    def __len__(self):
        # audio_len = len(self.df_Audio)
        # visual_len = len(self.df_Visual)
        pair_len = len(self.df_Pair)
        return pair_len
