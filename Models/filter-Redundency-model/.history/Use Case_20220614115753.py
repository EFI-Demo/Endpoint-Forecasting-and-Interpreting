import torch.nn as nn
from transformers import AdamW
from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os

class SentimentDataset(Dataset):
        def __init__(self, path_to_file):
            self.dataset = pd.read_csv(path_to_file, sep="\t", names=["label", "text"])
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            # 根据 idx 分别找到 text 和 label
            text = self.dataset.loc[idx, "text"]
            label = self.dataset.loc[idx, "label"]
            sample = {"text": text, "label": label}
            # 返回一个 dict
            return sample

# 定义训练的函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(dataloader):
        # 标签形状为 (batch_size, 1)
        label = batch["label"]
        label = label.to(device)
        text = batch["text"]

        # tokenized_text 包括 input_ids， token_type_ids， attention_mask
        tokenized_text = tokenizer(text, max_length=100, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")
        tokenized_text = tokenized_text.to(device)
        # 梯度清零
        optimizer.zero_grad()

        #output: (loss), logits, (hidden_states), (attentions)
        output = model(**tokenized_text, labels=label)

        # y_pred_prob = logits : [batch_size, num_labels]
        y_pred_prob = output[1]
        y_pred_prob = y_pred_prob.to(device)
        y_pred_label = y_pred_prob.argmax(dim=1)
        y_pred_label = y_pred_label.to(device)

        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        loss = criterion(y_pred_prob.view(-1, 2), label.view(-1))

        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # epoch 中的 loss 和 acc 累加
        # loss 每次是一个 batch 的平均 loss
        epoch_loss += loss.item()
        # acc 是一个 batch 的 acc 总和
        epoch_acc += acc
        if i % 200 == 0:
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset.dataset)

def evaluate(model, iterator, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            label = batch["label"]
            label = label.to(device)
            text = batch["text"]
            tokenized_text = tokenizer(text, max_length=100, add_special_tokens=True, truncation=True, padding=True,
                                       return_tensors="pt")
            tokenized_text = tokenized_text.to(device)

            output = model(**tokenized_text, labels=label)
            y_pred_label = output[1].argmax(dim=1)
            y_pred_label = y_pred_label.to(device)
            loss = output[0]
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            # epoch 中的 loss 和 acc 累加
            # loss 每次是一个 batch 的平均 loss
            epoch_loss += loss.item()
            # acc 是一个 batch 的 acc 总和
            epoch_acc += acc

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)

if __name__ == '__main__':

    # 超参数
    hidden_dropout_prob = 0.3
    num_labels = 2
    learning_rate = 1e-5
    weight_decay = 1e-2
    epochs = 5
    batch_size = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 文件路径
    data_path = "./data/"
    vocab_file = "vocab.txt"               # 词汇表
    train_data = data_path + "train+dev.csv"  # 训练数据集
    valid_data = data_path + "test.csv"  # 验证数据集
    output_dir = './models/'
    
        
    # 加载训练集
    sentiment_train_set = SentimentDataset(train_data)
    sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    # 加载验证集
    sentiment_valid_set = SentimentDataset(valid_data)
    sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # 定义 tokenizer，传入词汇表
    tokenizer = BertTokenizer(data_path+vocab_file)


    # 加载模型
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained("./pytorch_model.bin", config=config)
    model.to(device)

    # 定义优化器和损失函数
    # Prepare optimizer and schedule (linear warmup and decay)
    # 设置 bias 和 LayerNorm.weight 不使用 weight_decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    #optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 开始训练和验证
    for i in range(epochs):
        train_loss, train_acc = train(model, sentiment_train_loader, optimizer, criterion, device)
        print("train loss: ", train_loss, "\t", "train acc:", train_acc)
        valid_loss, valid_acc = evaluate(model, sentiment_valid_loader, device)
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
    
    #它包装在PyTorch DistributedDataParallel或DataParallel中
    model_to_save = model.module if hasattr(model, 'module') else model
    #如果使用预定义的名称保存，则可以使用`from_pretrained`加载
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)
    '''
    
    # 步骤2: 重新加载保存的模型

    #Bert模型示例
    model = BertForQuestionAnswering.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)  # Add specific options if needed
    #GPT模型示例
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(output_dir)
    tokenizer = OpenAIGPTTokenizer.from_pretrained(output_dir)

    '''

