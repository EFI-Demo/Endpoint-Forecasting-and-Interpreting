'''
Author: Ying Jie
Date: 2022-06-30 11:44:32
LastEditors: Ying Jie
LastEditTime: 2022-10-13 11:20:30
FilePath: \filter-Redundency-model\Use Case.py
Description: 过滤CTI中不相关句子，使用模型为transformers的pretrains模型架构
'''
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os
from nltk import tokenize

# 构建文件地址列表
file_doc_list = []

def traverse(f):
    global file_num
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            # 如果是压缩包或者是Pdf就跳过
            if tmp_path.lower().endswith('.rar') or tmp_path.lower().endswith('.pdf'):continue
            file_doc_list.append(tmp_path)
            # print('文件: %s'%tmp_path)
        else:
            # print('文件夹：%s'%tmp_path)
            traverse(tmp_path)

if __name__ == '__main__':
    # 加载保存的模型
    output_dir = "./models/"
    #如果使用预定义的名称保存，则可以使用`from_pretrained`加载
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    #Bert模型示例
    config = BertConfig.from_pretrained("./models/config.json")
    model = BertForSequenceClassification.from_pretrained(output_dir, config=config)
    tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed

    text_dir = r'D:\\Learning\\Paper\\apt\\基于CTI的攻击预警\\Dataset\\文本过滤\\多源CTI'

    traverse(text_dir)

    file_newdoc_list = []
    new_start_dir = 'D:\\Learning\\Paper\\apt\\基于CTI的攻击预警\\数据集构建\\文本过滤\\过滤后多源CTI'
    for doc in file_doc_list:
        # print(os.path.join(new_start_dir,doc.split('\\')[-1]))
        file_newdoc_list.append(os.path.join(new_start_dir,doc.split('\\多源CTI\\')[-1]))
        # print(doc.split('\\')[-1])

    #计算平均字数和句子个数
    totalAlphaLen = 0
    totalSenlen = 0
    for index, text in enumerate(file_doc_list):
        print(index, text.split('\\')[-2],text.split('\\')[-1])
        new_text = '' # 过滤后的新文本
        with open(text, encoding="ISO-8859-1", mode='r') as f:
            tokens = tokenize.sent_tokenize(f.read()) # 读入，并使用nltk的分句
            for token in tokens : 
                new_text += token
        totalAlphaLen += len(new_text)
        totalSenlen += len(tokens)
    print('average alpha len:', totalAlphaLen/index)
    print('average sen len:', totalSenlen/index)



    # 对newdoc写入过滤后文本
    # for index, text in enumerate(file_doc_list):
    #     print(index, text.split('\\')[-2],text.split('\\')[-1])
    #     new_text = '' # 过滤后的新文本
    #     with open(text, encoding="ISO-8859-1", mode='r') as f:
    #         tokens = tokenize.sent_tokenize(f.read()) # 读入，并使用nltk的分句
    #         for token in tokens : 
    #             if len(token) >= 512: token = token[:512] # 截取前512的长度
    #             encoding = tokenizer(token, return_tensors='pt')
    #             output = model(**encoding)
    #             y_pred_prob = output[0]
    #             y_pred_label = y_pred_prob.argmax(dim=1) # 此句的标签
    #             if y_pred_label == 1 : new_text += token
    #         # print(new_text)
    #     if len(new_text) < 200 : 
    #         print('the fltered text is too short.')
    #         continue # 如果太短就跳过
    #     new_doc = file_newdoc_list[index]
    #     new_docdir = new_doc.replace(new_doc.split('\\')[-1], '')
    #     if not os.path.exists(new_docdir): os.makedirs(new_docdir) # os.mkdir只能创建一级文件夹，如果要创建多级，需要makedirs
    #     with open(file_newdoc_list[index], mode='w', encoding="ISO-8859-1") as f2: # 在新地址写入
    #         f2.write(new_text)