import re

import emoji
from soynlp.normalizer import repeat_normalize

from ClassificationModule import *
from MultiClassification import *


def infer(x, path) :
    model = MultiClassification.load_from_checkpoint(path)
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())  
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    )
    processed = pattern.sub(' ', x)
    processed = url_pattern.sub(' ', processed)
    processed = processed.strip()
    processed = repeat_normalize(processed, num_repeats=2)

    tokenized = tokenizer(processed, return_tensors='pt')
    print(tokenized)

    output = model(input_ids=tokenized.input_ids[0], attention_mask=tokenized.attention_mask[0], token_type_ids=tokenized.token_type_ids[0])
    return nn.functional.softmax(output.logits, dim=-1)

text = '송중기 시대극은 믿고본다. 첫회 신선하고 좋았다.'
print(infer(text,'./bert_hate_multi_chpt/KoBERT/epoch=05-val_accuracy=0.441.ckpt'))


