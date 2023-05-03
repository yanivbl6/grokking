import torch
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import IterableDataset
from datasets import AbstractDataset
from utils import combine_logs
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item

import os

import argparse 

class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'val':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)

def train(config):
    print('using config:', config)
    train_cfg = config['train']
    wandb_cfg = config['wandb']
    if wandb_cfg['use_wandb']:
        wandb.init(project=wandb_cfg['wandb_project'], entity = 'dl-projects' , config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_item(config['dataset'])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    model = load_item(config['model'], dataset.n_vocab, dataset.n_out, device)
    model.train()
    train_dataloader = DataLoader(train_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], 
                              weight_decay=train_cfg['weight_decay'], 
                              betas=train_cfg['betas'])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: min(s / train_cfg['warmup_steps'], 1))
    step = 0
    for x, y in tqdm(train_dataloader):
        loss, logs = model.get_loss(x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schedule.step()
        if (step+1) % train_cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                all_val_logs = []
                for i, (val_x, val_y) in tqdm(enumerate(val_dataloader)):
                    if i >= train_cfg['eval_batches']:
                        break
                    _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
            out_log = {'val': combine_logs(all_val_logs), 'train': combine_logs([logs]), 'step': (step+1), 
                       'lr': float(lr_schedule.get_last_lr()[0])}
            print(out_log)
            if wandb_cfg['use_wandb']:
                wandb.log(out_log)
            model.train()
        step += 1
        if train_cfg['max_steps'] is not None and step >= train_cfg['max_steps']:
            break



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/train_grokk.yaml')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--bsize', type=int, default=512)
parser.add_argument('--max_steps', type=int, default=100000)
parser.add_argument('--eval_every', type=int, default=10)
parser.add_argument('--eval_batches', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--betas', type=float, nargs='+', default=[0.9, 0.98])
parser.add_argument('--warmup_steps', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--misc', dest='use_wandb' , action='store_false')
parser.add_argument('--wandb_project', type=str, default='grokking_replica')
parser.add_argument('--dataset_name', type=str, default='mod_subtract_dataset')
parser.add_argument('--p', type=int, default=96)
parser.add_argument('--frac_train', type=float, default=0.4)

parser.add_argument('--max_length', type=int, default=5)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--attn_dim', type=int, default=32)
parser.add_argument('--intermediate_dim', type=int, default=512)
parser.add_argument('--num_blocks', type=int, default=2)
parser.add_argument('--block_repeats', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--no_pre_norm', dest='pre_norm' , action='store_false')

parser.add_argument('--device', type=int, default=0)




#@hydra.main(config_path="../config", config_name="train_grokk")
#def main(cfg : DictConfig):
    ##cfg = OmegaConf.to_container(cfg)
    
def main():
    ##parse args
    args = parser.parse_args()



    cfg = {'dataset': {'name': args.dataset_name , 'p': args.p, 'frac_train': args.frac_train}, 
           'model': {'name': 'grokk_model', 
                     'transformer_config': {'max_length': args.max_length, 'heads': args.heads, 'hidden_dim': args.hidden_dim, 'attn_dim': args.attn_dim, 'intermediate_dim': args.intermediate_dim, 'num_blocks': args.num_blocks, 'block_repeats': args.block_repeats, 'dropout': args.dropout, 'pre_norm': args.pre_norm}, 
                    'checkpoint_path': None, 'strict_load': True}, 
            'train': {'num_workers': args.num_workers , 'bsize': args.bsize , 'lr': args.lr , 'weight_decay': args.weight_decay , 'betas': args.betas, 'warmup_steps': args.warmup_steps, 'eval_every': args.eval_every , 'eval_batches': args.eval_batches, 'max_steps': args.max_steps}, 
            'wandb': {'use_wandb': args.use_wandb , 'wandb_project': args.wandb_project}}


    ##change device to args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    train(cfg)

if __name__ == "__main__":
    main()

