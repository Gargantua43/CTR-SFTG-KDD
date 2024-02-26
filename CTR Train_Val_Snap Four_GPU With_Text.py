import torch
import time
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset.Dataset_pair_train_text import MyDatasetTrain
from Dataset.Dataset_pair_val_text import MyDatasetVal

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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


def main():
    global train_loss_list
    global train_acc_list
    global val_loss_list
    global val_acc_list

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Epochs
    epochs = 100

    # Available_GPU_List
    device_ids = [0, 1, 2, 3]
    one_gpu_batch = 256

    # Train-Data
    my_dataset = MyDatasetTrain("./Dataset-CSV/Snap_Train_CTR_label_pair With_Text.csv")
    train_loader = torch.utils.data.DataLoader(dataset=my_dataset, batch_size=one_gpu_batch * len(device_ids),
                                               drop_last=True, shuffle=True, num_workers=8)
    train_num = len(my_dataset)

    # Val-Data
    my_dataset_val = MyDatasetVal("./Dataset-CSV/Snap_Val_Pair_Label New With_Text.csv")
    val_loader = torch.utils.data.DataLoader(dataset=my_dataset_val, batch_size=one_gpu_batch * len(device_ids),
                                             drop_last=True, shuffle=False, num_workers=8)
    val_num = len(my_dataset_val)

    # Choose-Model
    choose_model = 'Proposed'
    if choose_model == 'Proposed' or choose_model == 'Proposed+Hall':
        model_rank = RankNet18()
    elif choose_model == 'Concat' or choose_model == 'Bimodal+BLIP':
        model_rank = RankNet13()
    elif choose_model == 'LLM':
        model_rank = RankNet15()
    elif choose_model == 'Visual':
        model_rank = RankNet151()
    elif choose_model == 'Audio':
        model_rank = RankNet152()

    model_rank = model_rank.double()
    model_rank = model_rank.train()

    # Parameter Initialization
    model_rank.apply(init_weights)

    # Copy Model
    model_rank = torch.nn.DataParallel(model_rank, device_ids=device_ids)

    # CUDA
    if torch.cuda.is_available():
        print("CUDA Is_Available:", torch.cuda.is_available())
        model_rank = model_rank.cuda(device=device_ids[0])

    # BCE_loss
    criterion = torch.nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model_rank.parameters(), lr=0.01)

    # Train
    for epoch in range(1, epochs + 1):
        # Train_start_time
        train_start_time = time.perf_counter() / 60
        print("Training_start_time:{}Hour".format(train_start_time / 60))

        # train
        pre_right = 0  # Count Right-Pairs
        train_steps = 0  # Count Batch
        running_loss_train = 0.0
        model_rank = model_rank.train()  # Model Train Again

        # train
        for v1_train, a1_train, t1_train, v2_train, a2_train, t2_train, ctr_label in iter(train_loader):
            # Count Batch
            train_steps += 1
            print("Running Train Epoch-Num:{},Batch-Num:{}".format(epoch, train_steps))
            optimizer.zero_grad()

            # CUDA
            if torch.cuda.is_available():
                v1_train = v1_train.cuda(device=device_ids[0])
                a1_train = a1_train.cuda(device=device_ids[0])
                t1_train = t1_train.cuda(device=device_ids[0])

                v2_train = v2_train.cuda(device=device_ids[0])
                a2_train = a2_train.cuda(device=device_ids[0])
                t2_train = t2_train.cuda(device=device_ids[0])

                ctr_label = ctr_label.cuda(device=device_ids[0])

            output_train = model_rank(v1_train, a1_train, t1_train, v2_train, a2_train, t2_train)

            # Train-ACC
            for i in range(1024):
                if output_train[i] >= 1 - output_train[i]:
                    p = 1
                else:
                    p = 0

                # Label-Compare
                if p == ctr_label[i]:
                    pre_right += 1
            print("Train Batch-Number {}:COMPARE_FINISH!".format(train_steps))

            # Add Dimension
            real_probability = torch.unsqueeze(ctr_label.to(torch.double), dim=1)

            # Calculate-Loss
            loss = criterion(output_train, real_probability)
            loss.backward()
            optimizer.step()

            # Calculate-Loss_Mean
            running_loss_train += loss.item()
        epoch_loss_train = running_loss_train / train_steps
        train_loss_list.append(epoch_loss_train)
        print("**Train Epoch-Num:{},Loss-Mean:{}**".format(epoch, epoch_loss_train))

        # Calculate Train-ACC
        acc_train = pre_right / train_num
        train_acc_list.append(acc_train)
        print("**Train Epoch-Num:{},Acc:{}**".format(epoch, acc_train))

        # Train_end_time
        train_end_time = time.perf_counter() / 60
        print("Training_end_time:{}Hour".format(train_end_time / 60))
        print("Running One Train Epoch Time(Minute):{}Minute".format(train_end_time - train_start_time))

        # validation
        if epoch % 3 == 0 or epoch == 100:
            acc_val = 0
            pre_right_val = 0
            val_steps = 0
            model_rank = model_rank.eval()
            running_loss_val = 0.0
            with torch.no_grad():
                # val_start_time
                val_start_time = time.perf_counter() / 60
                print("Val_start_time:{}Hour".format(val_start_time / 60))

                for v1_val, a1_val, t1_val, v2_val, a2_val, t2_val, ctr_label_val in iter(val_loader):
                    # Note Val-Batch
                    val_steps += 1
                    print("Running Val Epoch-Num:{},Batch-Num:{}".format(epoch, val_steps))

                    # CUDA
                    if torch.cuda.is_available():
                        v1_val = v1_val.cuda(device=device_ids[0])
                        a1_val = a1_val.cuda(device=device_ids[0])
                        t1_val = t1_val.cuda(device=device_ids[0])

                        v2_val = v2_val.cuda(device=device_ids[0])
                        a2_val = a2_val.cuda(device=device_ids[0])
                        t2_val = t2_val.cuda(device=device_ids[0])

                        ctr_label_val = ctr_label_val.cuda(device=device_ids[0])

                    output_val = model_rank(v1_val, a1_val, t1_val, v2_val, a2_val, t2_val)

                    # Val-ACC
                    for i in range(1024):
                        if output_val[i] >= 1 - output_val[i]:
                            p = 1
                        else:
                            p = 0

                        # Label-Compare
                        if p == ctr_label_val[i]:
                            pre_right_val += 1
                    print("Val Batch-NUmber {}:COMPARE_FINISH!".format(val_steps))

                    # Add Dimension
                    real_probability_val = torch.unsqueeze(ctr_label_val.to(torch.double), dim=1)

                    # Calculate-Loss
                    loss_val = criterion(output_val, real_probability_val)

                    # Calculate-Loss_Mean
                    running_loss_val += loss_val.item()
                epoch_loss_val = running_loss_val / val_steps
                val_loss_list.append(epoch_loss_val)
                print("！！Val Epoch-Num:{},Loss-Mean:{}！！".format(epoch, epoch_loss_val))

                # Calculate Val-ACC
                acc_val = pre_right_val / val_num
                val_acc_list.append(acc_val)
                print("！！Val Epoch-Num:{},Acc:{}！！".format(epoch, acc_val))

                # Val_end_time
                val_end_time = time.perf_counter() / 60
                print("Val_end_time:{}Hour".format(val_end_time / 60))
                print("Running One Val Epoch Time(Second):{}Minute".format(val_end_time - val_start_time))

        # Save Model-Weight
        weight_folder_path = './Experiment/RankNet_Version 18&MobileNetV2&LLaVA&Epochs=100 Snap'
        save_path = weight_folder_path + '/RankNet_Version 18&MobileNetV2&LLaVA+epoch=' + str(epoch) + '.pth'
        torch.save(model_rank.state_dict(), save_path)

    # Snap_train_loss
    col_train_loss = ['train_loss_epoch']
    info_array_train_loss = np.array(train_loss_list)
    df = pd.DataFrame(info_array_train_loss, columns=col_train_loss)
    train_loss_path = './Experiment/RankNet_Version 18&MobileNetV2&LLaVA&Epochs=100 Snap/' \
                      'Train-Snap Loss+Snap+MobileNetV2&LLaVA+RankNet_version 18.csv'

    df.to_csv(train_loss_path, encoding='utf-8')

    # Snap_train_acc
    col_train_acc = ['train_acc_epoch']
    info_array_train_acc = np.array(train_acc_list)
    df = pd.DataFrame(info_array_train_acc, columns=col_train_acc)
    train_acc_path = './Experiment/RankNet_Version 18&MobileNetV2&LLaVA&Epochs=100 Snap/' \
                     'Train-Snap Acc+Snap+MobileNetV2&LLaVA+RankNet_version 18.csv'
    df.to_csv(train_acc_path, encoding='utf-8')

    # Snap_val_loss
    col_val_loss = ['val_loss_epoch']
    info_array_val_loss = np.array(val_loss_list)
    df = pd.DataFrame(info_array_val_loss, columns=col_val_loss)
    val_loss_path = './Experiment/RankNet_Version 18&MobileNetV2&LLaVA&Epochs=100 Snap/' \
                    'Val-Snap Loss+Snap+MobileNetV2&LLaVA+RankNet_version 18.csv'
    df.to_csv(val_loss_path, encoding='utf-8')

    # Snap_val_acc
    col_val_acc = ['val_acc_epoch']
    info_array_val_acc = np.array(val_acc_list)
    df = pd.DataFrame(info_array_val_acc, columns=col_val_acc)
    val_acc_path = './Experiment/RankNet_Version 18&MobileNetV2&LLaVA&Epochs=100 Snap/' \
                   'Val-Snap Acc+Snap+MobileNetV2&LLaVA+RankNet_version 18.csv'
    df.to_csv(val_acc_path, encoding='utf-8')

    # Finish
    print("----Train&Val&ModelWeight_Save Finish----")


if __name__ == '__main__':
    main()
