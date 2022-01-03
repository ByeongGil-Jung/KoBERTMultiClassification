import os
import re

import emoji
import numpy as np
import pandas as pd
from soynlp.normalizer import repeat_normalize

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from kobert_tokenizer import KoBERTTokenizer

import transformers
from transformers import AdamW

class ClassificationDataset(Dataset) :
    def __init__(self, path, sep, doc_col, label_col, max_length, 
                num_workers=1, labels_dict=None) :
        self.dataset = pd.read_csv(path, sep=sep)
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

        self.max_length = max_length

        self.doc_col = doc_col
        self.label_col = label_col

        # labels
        # None : label이 num으로 되어 있음
        # dict : label이 num이 아닌 것으로 되어 있음
        # ex : {True : 1, False : 0}
        self.labels_dict = labels_dict

        # nan 제거
        self.dataset = self.dataset.dropna(axis=0)
        # 중복제거
        self.dataset.drop_duplicates(subset=[self.doc_col], inplace=True)

    def __len__(self) :
        return len(self.dataset)

    def cleanse(self, text) :
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())  
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        )
        processed = pattern.sub(' ', text)
        processed = url_pattern.sub(' ', processed)
        processed = processed.strip()
        processed = repeat_normalize(processed, num_repeats=2)
      
        return processed

    def __getitem__(self, idx) :
        document = self.cleanse(self.dataset[self.doc_col].iloc[idx])
        inputs = self.tokenizer(
            document,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True
        )

        if self.labels_dict :
            label = self.labels_dict[self.dataset[self.label_col].iloc[idx]]
        else :
            label = self.dataset[self.label_col].iloc[idx]

        return {
            'input_ids' : inputs['input_ids'][0],
            'attention_mask' : inputs['attention_mask'][0],
            'token_type_ids' : inputs['token_type_ids'][0],
            'label' : int(label)
        }

class ClassificationDataModule(pl.LightningDataModule) :
    def __init__(self, train_path, valid_path, max_length, batch_size, sep,
                doc_col, label_col, num_workers=1, labels_dict=None) :
        super().__init__()
        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.max_length = max_length
        self.doc_col = doc_col
        self.label_col = label_col
        self.sep = sep
        self.num_workers = num_workers
        self.labels_dict = labels_dict

    def _get_sample_weights(self, dataset):
        class_count_list = [0, 0, 0]

        for data in dataset:
            label = data["label"]
            class_count_list[label] += 1

        class_weight_list = 1. / torch.Tensor(class_count_list)

        sample_weights = [class_weight_list[class_idx] for class_idx in [int(data["label"]) for data in dataset]]

        return sample_weights

    def setup(self, stage=None) :
        self.set_train = ClassificationDataset(self.train_path, sep=self.sep,
                                            doc_col=self.doc_col, label_col=self.label_col,
                                            max_length = self.max_length, labels_dict=self.labels_dict)
        self.set_valid = ClassificationDataset(self.valid_path, sep=self.sep,
                                            doc_col=self.doc_col, label_col=self.label_col,
                                            max_length = self.max_length, labels_dict=self.labels_dict)

    def train_dataloader(self) :
        train_dataset = self.set_train
        train_sample_weights = self._get_sample_weights(dataset=train_dataset)

        train = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=WeightedRandomSampler(train_sample_weights, len(train_dataset))
        )
        return train
    
    def val_dataloader(self) :
        val = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return val
    
    def test_dataloader(self) :
        test = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test