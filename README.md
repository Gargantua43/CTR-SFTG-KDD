# CTR-SFTG
According to the following steps, the experimental results in the paper can be reproduced.
## Requirements
- CUDA 11.7
- Python 3.8.16
- Pytorch 2.0.0
## Data preparation
1. We have stored the processed dataset in the following link [[link](https://pan.baidu.com/s/1kj78uu7XM08Udwc3LLpbJw?pwd=p4nh) code: p4nh ]
2. Copy the downloaded folder to the project address.
   ```
   CTR-SFTG-KDD
   |--Dataset-CSV
   |--Dataset
   |--Snap_data
   |--Experiment
   |...
   ```
## Get start
You can run the program to directly obtain the accuracy results of different models.
1. CTR_Acc Snap Four_GPU.py
2. CTR_Acc Snap Four_GPU With_Text.py

You can modify the 'choose_model' parameter in the program to select different models.
## Model train
If you want to retrain the model, please execute the following program.
1. CTR Train_Val_Snap Four_GPU.py
2. CTR Train_Val_Snap Four_GPU With_Text.py

You can still choose different models by modifying the 'chose_model' parameter. But before training the model, you need to manually copy the dataset to the specified folder.

| choose_model | model | visual_embedding | text_embedding |
|:-:|:-:|:-:|:-:|
| 'Audio' |  RankNet152 | - | - | 
| 'Visual' | RankNet151 | visual_embedding MobileNetv2 | - |
| 'OPVAE' |RankNetJapan | visual_embedding RestNet50 | - |
| 'Bimodal+BLIP' | RankNet13 | visual_embedding MobileNetv2 | text_embedding_blip |
| 'LLM' | RankNet15 | - | text_embedding_llava_w |
| 'Bimodal' | RankNet| visual_embedding MobileNetv2 | - | 
| 'Concat' | RankNet13 | visual_embedding MobileNetv2 | text_embedding_llava_w |
| 'Proposed' | RankNet18 | visual_embedding MobileNetv2 | text_embedding_llava_w |
| 'Proposed+Hall' | RankNet18 | visual_embedding MobileNetv2 |text_embedding_llava_w delete=0.20 |

In addition, it is necessary to modify the parameters of 'weight_folder_path', 'train_loss_path', 'train-acc_path', 'val_loss_path', and 'val-acc_path' in the program to ensure that the file is stored in the desired location. There are some examples stored in the 'Experiment' folder for your reference.






