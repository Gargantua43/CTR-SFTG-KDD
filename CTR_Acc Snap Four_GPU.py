import os
import shutil
import torch
import random
import numpy as np

from Dataset.Dataset_pair_test import MyDatasetTest
from torch.utils.data import DataLoader

from model.JapanModel_version0 import RankNetJapan
from model.RankNet_version9_5 import RankNet


def seed_torch(seed=1):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch()


def copy_files(source_folder, destination_folder):
    for filename in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        shutil.copy2(source_file, destination_file)
    print('Test-Data Get Ready')


def main():
    choose_model = 'Bimodal'  # POVAE or Bimodal

    # 数据集准备
    if choose_model == 'POVAE':
        path_resnet = './Snap_data/Partition Dataset/test/visual_embedding ResNet50'
        path_visual = './Snap_data/Partition Dataset/test/visual_embedding'
        copy_files(path_resnet, path_visual)
    elif choose_model == 'Bimodal':
        path_mobilenet = './Snap_data/Partition Dataset/test/visual_embedding MobileNetv2'
        path_visual = './Snap_data/Partition Dataset/test/visual_embedding'
        copy_files(path_mobilenet, path_visual)

    # Available_GPU_List
    device_ids = [0, 1, 2, 3]
    one_gpu_batch = 1

    # Test Data
    my_dataset_test = MyDatasetTest("./Dataset-CSV/Snap_Test_CTR_Label_Pair Without_Train.csv")
    test_loader = DataLoader(dataset=my_dataset_test, batch_size=one_gpu_batch * len(device_ids), drop_last=True,
                             num_workers=8)
    test_num = len(my_dataset_test)

    if choose_model == 'POVAE':
        model_rank = RankNetJapan()
        model_weight_path = './model-weight/RankNet_JapanModel+Snap+ResNet50+Epoch=100.pth'
    elif choose_model == 'Bimodal':
        model_rank = RankNet()
        model_weight_path = './model-weight/RankNet_Version 9_5+Snap+MobileNetV2+Epoch=45.pth'

    model_rank = model_rank.double()
    model_rank = torch.nn.DataParallel(model_rank, device_ids=device_ids)

    # CUDA
    if torch.cuda.is_available():
        print("CUDA Is_Available:", torch.cuda.is_available())
        model_rank = model_rank.cuda(device=device_ids[0])

    # test
    pre_right = 0
    test_steps = 0
    model_rank.load_state_dict(torch.load(model_weight_path))
    model_rank = model_rank.eval()
    with torch.no_grad():
        for v1_test, a1_test, v2_test, a2_test, ctr_label_test in iter(test_loader):
            # Note Batch
            test_steps += 1
            print("Batch-Num:{}".format(test_steps))

            # CUDA-Data
            if torch.cuda.is_available():
                v1_test = v1_test.cuda(device=device_ids[0])
                a1_test = a1_test.cuda(device=device_ids[0])
                v2_test = v2_test.cuda(device=device_ids[0])
                a2_test = a2_test.cuda(device=device_ids[0])
                ctr_label_test = ctr_label_test.cuda(device=device_ids[0])

            # Model Output
            output_test = model_rank(v1_test, a1_test, v2_test, a2_test)

            # Label-Compare
            for i in range(4):
                if output_test[i] >= 1 - output_test[i]:
                    p = 1
                else:
                    p = 0

                if p == ctr_label_test[i]:
                    pre_right += 1

            print("{} Numbers-Batch:COMPARE_FINISH!".format(test_steps))
            print("Predict the Correct Number:{}".format(pre_right))

        # Calculation Accuracy
        acc = pre_right / test_num  # Snap

        print("Visual&Audio_Embedding+RankNet-ACC:{}".format(acc))


if __name__ == '__main__':
    main()
