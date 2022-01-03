import random

# from ClassificationModule import *
from ClassificationModule_weighted_sampling import *
from MultiClassification import *
# from CPUsimplemulti import *


# Fix random seed
def set_random_seed(random_seed=777):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")


if __name__ == "__main__" :
    set_random_seed(random_seed=777)

    model = MultiClassification(learning_rate=1e-5, dropout_p=0.3, hidden_size=768, num_classes=3)

    dm = ClassificationDataModule(batch_size=16, train_path='./train.tsv', valid_path='./dev.tsv',
                                    max_length=256, sep='\t', doc_col='comments', label_col='hate', num_workers=1,
                                    labels_dict={'none' : 0, 'hate' : 1, 'offensive' : 2})

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_accuracy',
                                                    dirpath='./bert_hate_multi_chpt',
                                                    filename='KoBERT/{epoch:02d}-{val_accuracy:.3f}',
                                                    verbose=True,
                                                    save_last=True,
                                                    mode='max',
                                                    save_top_k=-1,
                                                    )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join('./bert_hate_multi_chpt', 'tb_logs'))

    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        default_root_dir='./bert_hate_multi_chpt/checkpoints',
        logger = tb_logger,
        callbacks = [checkpoint_callback, lr_logger],
        max_epochs=2,
        devices=[0],  # Set device numbers
        accelerator="auto"
        # gpus=1
    )

    trainer.fit(model, dm)

# if __name__ == "__main__" :
#     model = MultiClassification(learning_rate=0.0001, dropout_p=0.3, hidden_size=768, num_classes=2)

#     dm = ClassificationDataModule(batch_size=8, train_path='./ratings_train_pre.txt', valid_path='./ratings_test_pre.txt',
#                                     max_length=256, sep='\t', doc_col='document', label_col='label', num_workers=1
#                                     )

#     checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_accuracy',
#                                                     dirpath='./bert_nsmc_multi_chpt',
#                                                     filename='KoBERT/{epoch:02d}-{val_accuracy:.3f}',
#                                                     verbose=True,
#                                                     save_last=True,
#                                                     mode='max',
#                                                     save_top_k=-1,
#                                                     )

#     tb_logger = pl_loggers.TensorBoardLogger(os.path.join('./bert_nsmc_multi_chpt', 'tb_logs'))

#     lr_logger = pl.callbacks.LearningRateMonitor()

#     trainer = pl.Trainer(
#         default_root_dir='./bert_nsmc_multi_chpt/checkpoints',
#         logger = tb_logger,
#         callbacks = [checkpoint_callback, lr_logger],
#         max_epochs=1,
#         gpus=1
#     )
#     trainer.fit(model, dm)