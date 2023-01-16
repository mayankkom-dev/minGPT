"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict
import torchtext
import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
from torch.nn.utils.rnn import pad_sequence 

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        aid = [item[0] for item in batch]
        aid_type = [item[1] for item in batch]
        session_type_target = [item[2] for item in batch]
        session_aid_target = [item[3] for item in batch]
        
        aid = pad_sequence(aid, batch_first=True, padding_value=self.pad_idx)
        aid_type = pad_sequence(aid_type, batch_first=True, padding_value=self.pad_idx)
        session_type_target = pad_sequence(session_type_target, batch_first=True, padding_value=self.pad_idx)
        session_aid_target = pad_sequence(session_aid_target, batch_first=True, padding_value=self.pad_idx)

        return aid, aid_type, session_type_target, session_aid_target

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 8
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            # sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=True,
            pin_memory=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=MyCollate(pad_idx=0)
        )
        # train_loader = torchtext.data.BucketIterator(
        #      self.train_dataset,
        #      batch_size=config.batch_size,
        #      sort_key=lambda x: len(x[0]),
        #      repeat=True,
        #      sort=False, 
        #      shuffle=True,
        #      sort_within_batch=True, 
        # )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y, y_target, x_target = batch

            # forward the model
            logits, logits_a_type, self.loss, self.loss_a_type = model(x, y,  y_target, x_target)

            self.loss_final = self.loss + self.loss_a_type # TODO
            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss_final.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
