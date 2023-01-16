"""
Trains a otto_session article buy and article_type predictor.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.custom_model import CustomGPT
from mingpt.custom_trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from otto_dataset import OttoTrainDataset
# -----------------------------------------------------------------------------
def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/otto_gpt'

    # data
    C.data = OttoTrainDataset.get_default_config()

    # model
    C.model = CustomGPT.get_default_config()
    C.n_layer = 8
    C.n_head = 16
    C.n_embd =  1024
    # these options must be filled in externally
    C.vocab_size = None
    C.block_size = None

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.batch_size = 32
    C.max_iters = None
    return C

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # import os
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    # text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    # train_dataset = CharDataset(config.data, text)
    from otto_datasplit import OttoDataSplit
    import pandas as pd
    import gc
    df_loc = '/kaggle/input/otto-full-optimized-memory-footprint/train.parquet'
    train_df = pd.read_parquet(df_loc)
    otto_split = OttoDataSplit(train_df, min_session_len=3, max_session_len=100)
    # use any type of split to get session distribution for training
    train_session_ids, test_session_ids = otto_split.get_split_session_ids(train_size=0.7)
    
    del otto_split; del train_df; gc.collect()
    train_dataset = OttoTrainDataset(config.data, df_loc, train_session_ids)
    

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = CustomGPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            # with torch.no_grad():
            #     # sample from the model...
            #     context = "O God, O God!"
            #     x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
            #     y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
            #     completion = ''.join([train_dataset.itos[int(i)] for i in y])
            #     print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
