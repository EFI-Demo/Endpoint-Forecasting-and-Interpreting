'''
Author: Ying Jie
LastEditors: Ying Jie
Description: Filtering redundant text in CTI, using pre-trained model of bert
'''
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os
from nltk import tokenize

file_doc_list = []

def traverse(f):
    global file_num
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.lower().endswith('.rar') or tmp_path.lower().endswith('.pdf'):continue
            file_doc_list.append(tmp_path)
        else:
            traverse(tmp_path)

if __name__ == '__main__':

    output_dir = "./models/"

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    config = BertConfig.from_pretrained("./models/config.json")
    model = BertForSequenceClassification.from_pretrained(output_dir, config=config)
    tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed

    text_dir = r'' # dir of CTI reports

    traverse(text_dir)

    file_newdoc_list = []
    new_start_dir = '' # new dir of CTI reports
    for doc in file_doc_list:
        file_newdoc_list.append(os.path.join(new_start_dir,doc.split('\\CTI reports\\')[-1]))

    for index, text in enumerate(file_doc_list):
        print(index, text.split('\\')[-2],text.split('\\')[-1])
        new_text = '' 
        with open(text, encoding="ISO-8859-1", mode='r') as f:
            tokens = tokenize.sent_tokenize(f.read()) 
            for token in tokens : 
                if len(token) >= 512: token = token[:512] 
                encoding = tokenizer(token, return_tensors='pt')
                output = model(**encoding)
                y_pred_prob = output[0]
                y_pred_label = y_pred_prob.argmax(dim=1) 
                if y_pred_label == 1 : new_text += token
            # print(new_text)
        if len(new_text) < 200 : 
            print('the fltered text is too short.')
            continue 
        new_doc = file_newdoc_list[index]
        new_docdir = new_doc.replace(new_doc.split('\\')[-1], '')
        if not os.path.exists(new_docdir): os.makedirs(new_docdir) 
        with open(file_newdoc_list[index], mode='w', encoding="ISO-8859-1") as f2: # write filtered CTI in new dir
            f2.write(new_text)