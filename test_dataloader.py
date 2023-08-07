from data_set import YourDataSetClass
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_params = {
    "MODEL": "/data/clue/ChatYuan-large-v1",  # model_type
    "TRAIN_BATCH_SIZE": 8,  # training batch size, 8
    "VALID_BATCH_SIZE": 8,  # validation batch size,8
    "TRAIN_EPOCHS": 1,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text, 512
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text,64
    "SEED": 42,  # set seed for reproducibility
}



path = 'pCLUE_train.csv'

tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])



df = pd.read_csv(path)  # 数据量：1200k数据。
# df = df.sample(frac=0.01) # TODO  取消本行代码，如果你需要更多数据训练
print("df.head:",df.head(n=5))
print("df.shape:",df.shape)

train_dataset = df[["input", "target"]]
train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text="input",
        target_text="target"
    )
training_loader = DataLoader(training_set, **train_params)

for _, data in enumerate(training_loader, 0):
    y = data["target_ids"].to(device, dtype=torch.long)
    y_ids = y[:, :-1].contiguous()  # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
    lm_labels = y[:, 1:].clone().detach()  # target, for second to end.e.g."好吗？<EOS>"
    lm_labels[y[:,
              1:] == tokenizer.pad_token_id] = -100  # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
    ids = data["source_ids"].to(device, dtype=torch.long)  # input. e.g. "how are you?"
    mask = data["source_mask"].to(device, dtype=torch.long)



    print(y.type)
    print(y_ids.type)
    print(lm_labels.type)
    print(ids.type)
    print(mask.type)