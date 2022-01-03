import os

import torch
import torchmetrics
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import transformers
from transformers import BertModel, AdamW


# device = torch.device("cuda")

# https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
# https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/electra#transformers.ElectraForSequenceClassification
# https://www.koreascience.or.kr/article/CFKO202130060662823.pdf

class MultiClassification(pl.LightningModule):
    def __init__(self, learning_rate, dropout_p=0.5, hidden_size=768, num_classes=2):
        super().__init__()
        # hyperparameters
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        print(learning_rate, dropout_p, hidden_size, num_classes)
        self.save_hyperparameters()

        # set models
        # BERT
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.linear_1 = nn.Linear(self.hidden_size, 128)
        self.linear_2 = nn.Linear(128, self.num_classes)

        # metrics functions
        self.metric_acc = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro')
        self.metric_f1 = torchmetrics.F1(num_classes=self.num_classes, average='macro')
        self.metric_rec = torchmetrics.Recall(num_classes=self.num_classes, average='macro')
        self.metric_pre = torchmetrics.Precision(num_classes=self.num_classes, average='macro')

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids
                           )
        output = self.dropout(output.pooler_output)
        output = F.leaky_relu(self.linear_1(output), negative_slope=0.1)
        output = self.linear_2(output)
        return output

    def training_step(self, batch, batch_idx):
        '''
        ##########################################################
        bert forward input shape information
        * input_ids.shape (batch_size, max_length)
        * attention_mask.shape (batch_size, max_length)
        * label.shape (batch_size,)
        ##########################################################
        '''

        # change label shape (list -> torch.Tensor((batch_size, 1)))
        label = batch['label'].view([-1, 1])

        output = self(input_ids=batch['input_ids'].to(self.device),
                      attention_mask=batch['attention_mask'].to(self.device),
                      token_type_ids=batch['token_type_ids'].to(self.device)
                      )
        '''
        ##########################################################
        bert forward output shape information
        * loss.shape (1,)
        * logits.shape (batch_size, config.num_labels=2)
        ##########################################################
        '''
        # loss = output.loss
        loss = self.loss_func(output.to(self.device), batch['label'].to(self.device))

        softmax = nn.functional.softmax(output, dim=1)
        preds = softmax.argmax(dim=1)

        self.log("train_loss", loss, prog_bar=True)

        return {
            'loss': loss,
            'pred': preds,
            'label': batch['label']
        }

    def training_epoch_end(self, outputs, state='train'):
        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['label'].tolist()
            y_pred += i['pred'].tolist()
        y_true = torch.tensor(y_true).to(self.device)
        y_pred = torch.tensor(y_pred).to(self.device)
        print(y_true, y_pred)
        print(y_true.shape, y_pred.shape)
        acc = self.metric_acc(y_true, y_pred)
        prec = self.metric_pre(y_true, y_pred)
        rec = self.metric_rec(y_true, y_pred)
        f1 = self.metric_f1(y_true, y_pred)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')

    def validation_step(self, batch, batch_idx):
        '''
        ##########################################################
        bert forward input shape information
        * input_ids.shape (batch_size, max_length)
        * attention_mask.shape (batch_size, max_length)
        ##########################################################
        '''
        output = self(input_ids=batch['input_ids'].to(self.device),
                      attention_mask=batch['attention_mask'].to(self.device),
                      token_type_ids=batch['token_type_ids'].to(self.device))
        preds = nn.functional.softmax(output, dim=1).argmax(dim=1)

        labels = batch['label']
        accuracy = self.metric_acc(preds, labels)
        f1 = self.metric_f1(preds, labels)
        recall = self.metric_rec(preds, labels)
        precision = self.metric_pre(preds, labels)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, on_epoch=True, prog_bar=True)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision
        }

    def validation_epoch_end(self, outputs):
        val_acc = torch.stack([i['accuracy'] for i in outputs]).mean()
        val_f1 = torch.stack([i['f1'] for i in outputs]).mean()
        val_rec = torch.stack([i['recall'] for i in outputs]).mean()
        val_pre = torch.stack([i['precision'] for i in outputs]).mean()
        # self.log('val_f1', val_f1, on_epoch=True, prog_bar=True)
        # self.log('val_acc', val_acc, on_epoch=True, prog_bar=True)
        print(f'val_accuracy : {val_acc}, val_f1 : {val_f1}, val_recall : {val_rec}, val_precision : {val_pre}')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }