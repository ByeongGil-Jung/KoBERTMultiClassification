from ClassificationModule import *
from MultiClassification import *

from torchmetrics import Precision

prec_score = Precision(num_classes=3)

model = MultiClassification(learning_rate=0.0001, dropout_p=0.5, hidden_size=768, num_classes=3)
dm = ClassificationDataModule(batch_size=32, train_path='./train.tsv', valid_path='./dev.tsv',
                                    max_length=256, sep='\t', doc_col='comments', label_col='hate', num_workers=1,
                                    labels_dict={'none' : 0, 'hate' : 1, 'offensive' : 2})

dm.setup()

t = dm.train_dataloader()

idx, data = next(enumerate(t))

output = model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids'])
print('output :', output)

preds = torch.nn.functional.softmax(output,dim=1).argmax(dim=1)

print("prec : ", prec_score(preds, data['label']))
