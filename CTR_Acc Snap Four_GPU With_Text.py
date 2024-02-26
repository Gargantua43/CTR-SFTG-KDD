import os
import torch
import shutil
import random
import numpy as np
from torch.utils.data import DataLoader
from Dataset.Dataset_pair_test_text import MyDatasetTest

from model.RankNet_version18 import RankNet18, init_weights
from model.RankNet_version13 import RankNet13, init_weights
from model.RankNet_version15 import RankNet15, init_weights
from model.RankNet_version15_1 import RankNet151, init_weights
from model.RankNet_version15_2 import RankNet152, init_weights


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
    # Choose-Model
    choose_model = 'Proposed+Hall'

    # Text-Data Preparation
    text_path = './Snap_data/Partition Dataset/test/text_embedding'
    if choose_model == 'Proposed' or choose_model == 'Concat' or choose_model == 'LLM':
        llava_text_path = './Snap_data/Partition Dataset/test/text_embedding_llava_w'
        copy_files(llava_text_path, text_path)
    elif choose_model == 'Bimodal+BLIP':
        blip_text_path = './Snap_data/Partition Dataset/test/text_embedding_blip'
        copy_files(blip_text_path, text_path)
    elif choose_model == 'Proposed+Hall':
        llava_hall_text_path = './Snap_data/Partition Dataset/test/text_embedding_llava_w delete=0.20'
        copy_files(llava_hall_text_path, text_path)

    # Available_GPU_List
    device_ids = [0, 1, 2, 3]
    one_gpu_batch = 1

    # Test Data
    my_dataset_test = MyDatasetTest("./Dataset-CSV/Snap_Test_CTR_Label_Pair Without_Train With_Text.csv")
    test_loader = DataLoader(dataset=my_dataset_test, batch_size=one_gpu_batch * len(device_ids), drop_last=True,
                             num_workers=8)
    test_num = len(my_dataset_test)

    if choose_model == 'Proposed':
        model_rank = RankNet18()
        model_weight_path = './model-weight/RankNet_Version 18+Snap+MobileNetV2&LLaVA_W&BERT+epoch=66.pth'
    elif choose_model == 'Concat':
        model_rank = RankNet13()
        model_weight_path = './model-weight/RankNet_Version 13+Snap+MobileNetV2&LLaVA_W&BERT+epoch=96.pth'
    elif choose_model == 'Bimodal+BLIP':
        model_rank = RankNet13()
        model_weight_path = './model-weight/RankNet_Version 13+Snap+MobileNetV2&BLIP&BERT+epoch=36.pth'
    elif choose_model == 'LLM':
        model_rank = RankNet15()
        model_weight_path = './model-weight/RankNet_Version 15+Snap+LLaVA_W&BERT+epoch=81.pth'
    elif choose_model == 'Visual':
        model_rank = RankNet151()
        model_weight_path = './model-weight/RankNet_Version 15_1+Snap+MobileNetV2+epoch=9.pth'
    elif choose_model == 'Audio':
        model_rank = RankNet152()
        model_weight_path = './model-weight/RankNet_Version 15_2+Snap+VGGish+epoch=100.pth'
    elif choose_model == 'Proposed+Hall':
        model_rank = RankNet18()
        model_weight_path = './model-weight/RankNet_Version 18+Snap+MobileNetV2&LLaVA_W&BERT&TotalData_Delete=0.2+epoch=93.pth'

    model_rank = model_rank.double()
    model_rank = torch.nn.DataParallel(model_rank, device_ids=device_ids)

    # CUDA-Model
    if torch.cuda.is_available():
        print("CUDA Is_available:", torch.cuda.is_available())
        model_rank = model_rank.cuda(device=device_ids[0])

    # test
    pre_right = 0
    test_steps = 0
    model_rank.load_state_dict(torch.load(model_weight_path))
    model_rank = model_rank.eval()
    with torch.no_grad():
        for v1_test, a1_test, t1_test, v2_test, a2_test, t2_test, ctr_label_test in iter(test_loader):
            # Note Batch
            test_steps += 1
            print("{} Number-Batch".format(test_steps))

            # CUDA-Data
            if torch.cuda.is_available():
                v1_test = v1_test.cuda(device=device_ids[0])
                a1_test = a1_test.cuda(device=device_ids[0])
                t1_test = t1_test.cuda(device=device_ids[0])

                v2_test = v2_test.cuda(device=device_ids[0])
                a2_test = a2_test.cuda(device=device_ids[0])
                t2_test = t2_test.cuda(device=device_ids[0])

                ctr_label_test = ctr_label_test.cuda(device=device_ids[0])

            # Model Output
            output_test = model_rank(v1_test, a1_test, t1_test, v2_test, a2_test, t2_test)

            # Label-Compare
            for i in range(4):
                if output_test[i] >= 1 - output_test[i]:
                    p = 1
                else:
                    p = 0

                print('Pre_value:{}'.format(p))
                print('True_value:{}'.format(ctr_label_test[i]))

                if p == ctr_label_test[i]:
                    pre_right += 1

            print("{} Numbers-Batch:COMPARE_FINISH!".format(test_steps))
            print("Predict the Correct Number:{}".format(pre_right))

        # Calculation Accuracy
        acc = pre_right / test_num  # Snap

        print("Visual&Audio-Embedding With-Text+RankNet-Acc:{}".format(acc))


if __name__ == '__main__':
    main()
