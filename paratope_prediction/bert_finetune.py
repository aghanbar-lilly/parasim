from transformers import BertModel, BertTokenizer
import re
import pandas as pd
import numpy as np
from utils import *
from  CustomizedTokenClassification import CustomizedTokenClassification
from transformers import  BertConfig, AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd

addArgument()
args = getArgs()

train_dataset = None
dev_dataset = None
val_dataset = None
train_texts, dev_texts, val_texts, train_labels, dev_labels, val_labels = None, None, None, None, None, None

if args.predict_only == 1:
    model = CustomizedTokenClassification.from_pretrained(args.predict_model, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False )
    val_texts, val_labels = prepare_data(args.predict_data)
    val_encodings = tokenizer(val_texts,padding=True)
    val_l= add_labels(val_encodings,val_labels)
    val_dataset = Dataset(val_encodings,val_l)
    #print(val_l)
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        compute_metrics=compute_metrics            # evaluation dataset
    )
    predictions_val, lb_val, _ = trainer.predict(val_dataset)


    import torch
    import torch.nn.functional as F
    soft_val = F.softmax(torch.tensor(predictions_val),dim=-1).numpy()
    
    
    m = ([''.join(m) for m in get_tokens(val_encodings['input_ids'])])
    out = {'seq':[],'cdrs':[],'predictions':[]}
    probs = soft_val[...,1]
    for i in range(len(m)):
        seq = val_texts[i]
        f =np.where(np.array(val_l[i])!=-100)[0]
        cdrs = ' '.join(np.array(list(m[i]))[f])
        p = probs[i][f]
        p = ["%.2f" % q for q in p]
        p = ' '.join(p)
        out['seq'] += [seq]
        out['cdrs'] += [cdrs]
        out['predictions'] += [p]
    outdf = pd.DataFrame.from_dict(out)
    outdf.to_csv('predictions.csv')
    
    pred_val = soft_val[...,1][np.where(lb_val!=-100)].flatten()
    
    true_val = lb_val[np.where(lb_val!=-100)].flatten()

    from sklearn.metrics import roc_auc_score
    print ("")
    print ("TEST AUC:", roc_auc_score(true_val, pred_val))

    from sklearn.metrics import f1_score,matthews_corrcoef

    pred_labels_val = np.argmax(predictions_val, axis=2)[np.where(lb_val!=-100)]
    print ("TEST F1:", f1_score(true_val, pred_labels_val))
    print("TEST MCC:", matthews_corrcoef(true_val,pred_labels_val))

else:
    model = CustomizedTokenClassification.from_pretrained("Rostlab/prot_bert", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    if args.train_predict == 1:
        train_dev_seq, train_dev_label = prepare_data(args.data) #("parapred_dataset_large.csv")
        train_texts, dev_texts, train_labels, dev_labels = train_test_split(train_dev_seq, train_dev_label, test_size=.1,random_state=10)
        val_texts, val_labels = prepare_data(args.predict_data)

    else:
        sequences, labels = prepare_data(args.data)
        train_texts, dev_val_texts, train_labels, dev_val_labels = train_test_split(sequences, labels, test_size=.2,random_state=10)
        dev_texts, val_texts,  dev_labels, val_labels = train_test_split(dev_val_texts, dev_val_labels, test_size=.5,random_state=20)

    train_encodings = tokenizer(train_texts,padding=True)
    train_encodings['labels']= train_labels

    dev_encodings = tokenizer(dev_texts,padding=True)
    val_encodings = tokenizer(val_texts,padding=True)

    train_l= add_labels(train_encodings,train_labels)
    dev_l = add_labels(dev_encodings,dev_labels)
    val_l= add_labels(val_encodings,val_labels)

    train_dataset = Dataset(train_encodings,train_l)
    dev_dataset = Dataset(dev_encodings, dev_l)
    val_dataset = Dataset(val_encodings,val_l)
    print (len(train_dataset), len(dev_dataset), len(val_dataset))

    metric = load_metric("seqeval")

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=args.num_epochs,
        evaluation_strategy = "epoch",
        learning_rate=args.learning_rate,             # total number of training epochs
        per_device_train_batch_size=3,  # batch size per device during training
        per_device_eval_batch_size=3,   # batch size for evaluation
        warmup_steps=1000,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1000,
        load_best_model_at_end=True,
        save_total_limit = 1,
        metric_for_best_model='f1',
    )


    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics            # evaluation dataset
    )

    trainer.train()

    trainer.evaluate()

    predictions_dev, lb_dev, _ = trainer.predict(dev_dataset)
    predictions_val, lb_val, _ = trainer.predict(val_dataset)


    import torch
    import torch.nn.functional as F 
    soft_dev = F.softmax(torch.tensor(predictions_dev),dim=-1).numpy()
    soft_val = F.softmax(torch.tensor(predictions_val),dim=-1).numpy()


    pred_dev  = soft_dev[...,1][np.where(lb_dev!=-100)].flatten()
    pred_val = soft_val[...,1][np.where(lb_val!=-100)].flatten()
    true_dev = lb_dev[np.where(lb_dev!=-100)].flatten()
    true_val = lb_val[np.where(lb_val!=-100)].flatten()

    from sklearn.metrics import roc_auc_score
    print ("")
    print ("DEV AUC:", roc_auc_score(true_dev,pred_dev))
    print ("TEST AUC:", roc_auc_score(true_val, pred_val))

    from sklearn.metrics import f1_score,matthews_corrcoef

    pred_labels_dev = np.argmax(predictions_dev, axis=2)[np.where(lb_dev!=-100)]
    pred_labels_val = np.argmax(predictions_val, axis=2)[np.where(lb_val!=-100)]
    print("DEV F1:",f1_score(true_dev,pred_labels_dev))
    print ("TEST AUC:", f1_score(true_val, pred_labels_val))
    print("DEV MCC:", matthews_corrcoef(true_dev,pred_labels_dev))
    print("TEST MCC:", matthews_corrcoef(true_val,pred_labels_val))
