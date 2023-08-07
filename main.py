import os,json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os,time
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from trainer import T5Trainer



model_params = {
    "MODEL": "ClueAI/ChatYuan-large-v1",  # model_type
    "TRAIN_BATCH_SIZE": 8,  # training batch size, 8
    "VALID_BATCH_SIZE": 8,  # validation batch size,8 
    "TRAIN_EPOCHS": 1,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text, 512
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text,64
    "SEED": 42,  # set seed for reproducibility
}
print("end...")
path = 'pCLUE_train_2.csv'
# 训练模型
# 使用 pCLUE:1200000+多任务提示学习数据集 的部分数据
# dataframe必须有2列:
#   - input: 文本输入
#   - target: 目标输出
df = pd.read_csv(path)  # 数据量：1200k数据。
df = df.sample(frac=0.01) # TODO  取消本行代码，如果你需要更多数据训练
print("df.head:",df.head(n=5))
print("df.shape:",df.shape)
# 显存占用说明：如果运行现在显存不足，请使用nvidia-smi查看显存；如果显卡多数被占用了，请重启colab程序
T5Trainer(
    dataframe=df,
    source_text="input",
    target_text="target",
    model_params=model_params,
    output_dir="outputs",
)
print("end..")

