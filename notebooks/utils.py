import torch
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, load_dataset
from transformers import RobertaForSequenceClassification,AutoModelForSequenceClassification, RobertaTokenizerFast,TrainingArguments, Trainer
import torch
import os
import regex as re
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.metrics import top_k_accuracy_score, f1_score
from tqdm import tqdm
nltk.download('punkt')
import random, torch, os, numpy as np

def seed_everything(seed=42,training=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not training:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def encode(examples,tokenizer):
     return tokenizer(examples["input"], truncation=True, padding='max_length')
    

def fine_tune(tokenizer,model,dataset,model_name,log_dir,output_dir,num_labels):
    dataset_ = dataset.map(lambda x:encode(x, tokenizer), batched=True, batch_size=500)
    training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    evaluation_strategy="steps",
    eval_steps=2000,             
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=5,  # batch size per device during training
    per_device_eval_batch_size=5,   # batch size for evaluation
    load_best_model_at_end=True,
    learning_rate=3e-5,
    warmup_steps=50,                # number of warmup steps for learning rate scheduler
    logging_steps=100,
    weight_decay=0.01,               # strength of weight decay
    save_total_limit=1,   
    report_to=["tensorboard"],   # limit the total amount of checkpoints. Deletes the older checkpoints.  
    fp16=True,
    logging_dir=log_dir,
    metric_for_best_model="eval_loss",
    greater_is_better=False
    )

    trainer = Trainer(
        model= model,                                
        args= training_args,                           # training arguments, defined above
        train_dataset=dataset_["train"],         # training dataset
        eval_dataset=dataset_["valid"],  
    )
    if len(os.listdir(output_dir)) == 0:
        trainer.train()
    
    model=RobertaForSequenceClassification.from_pretrained(output_dir+os.listdir(output_dir)[0],num_labels=num_labels)
    preds=trainer.predict(dataset_["test"])
    write_results(model_name, preds[0],10, dataset_["test"])
    return trainer, preds

def write_results(method:str, preds, to_k:int,data):
    acc_to_k=[f"f1:{f1_score(data['labels'],[np.argmax(i) for i in preds],average='macro')}"]
    try:
        df=pd.read_csv("../evaluation_results/journal_finder_output.csv")
    except:
        df=pd.DataFrame()

    for k in range(1,to_k+1):
        acc_to_k.append(top_k_accuracy_score(data["labels"],preds,k=k,labels=list(range(len(preds[0])))))
    
    df[method]=pd.Series(acc_to_k)
    
    df.to_csv("../evaluation_results/journal_finder_output.csv",index=False)
    return acc_to_k


# There is a problem with the tokenizer it does not tokenize <mask> as the special mask token, ...

def replace_masks(sentance,tokenizer):
    inputs = tokenizer(sentance)
    new_inputs=[]
    new_mask=[]
    mask_index=-1
    for i,tok in enumerate(inputs["input_ids"]):
        if tok==34 and inputs["input_ids"][i-1]==44174 and inputs["input_ids"][i-2]==1388:
            new_inputs=new_inputs[:-2]
            new_mask=new_mask[:-2]
            new_inputs.append(4)
            new_mask.append(0)
            mask_index=len(new_inputs)-1
        else:  
            new_inputs.append(tok)
            new_mask.append(1)
    return {"input_ids":torch.IntTensor(new_inputs).view(-1,len(new_inputs)), "attention_mask":torch.IntTensor(new_mask).view(-1,len(new_mask))}, mask_index

def softmax(logits: list)-> list:
    proba = [np.exp(i) for i in logits]
    total = sum(proba)
    proba = [i / total for i in proba]
    return proba

def top_predict_masked_token(sentance, nb_pred,tokenizer,model):
    inputs, mask_index=replace_masks(sentance,tokenizer)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    logits=softmax(logits[0][mask_index])
    max_indexes=sorted(range(len(logits)), key=lambda i: logits[i])[-nb_pred:]
    max_indexes.reverse()
    proba=[logits[i] for i in max_indexes]
    res=tokenizer.convert_ids_to_tokens(max_indexes)
    return res , proba

def display_output(sentance, nb_pred,tokenizer,model):
    words,proba=top_predict_masked_token(sentance,nb_pred,tokenizer,model)
    df=pd.DataFrame({"word":words,"proba":proba})
    print(f"completions for sentance:{sentance}")
    display(df.round(2))
    

