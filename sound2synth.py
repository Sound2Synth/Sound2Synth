
from utils import *
import pytorch_lightning as pl
from pyheaven.torch_utils import HeavenDataset, HeavenDataLoader
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

import numpy as np

class Sound2SynthModel(pl.LightningModule):
    def __init__(self, net, interface, args):
        super().__init__()
        self.net = net; self.interface = interface
        self.args = args; self.criteria = interface.criteria
        self.learning_rate = args.learning_rate

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.net.parameters(),
            lr = self.args.learning_rate,
            weight_decay = self.args.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
            warmup_epochs = self.args.warmup_epochs,
            max_epochs = self.args.num_epochs,
            warmup_start_lr = self.args.warmup_start_lr_ratio*self.args.learning_rate,
            eta_min = self.args.eta_min,
        )
        return [optimizer], [scheduler]
         
    def train_dataloader(self):
        return HeavenDataLoader(DataLoader(self.args.datasets.train,batch_size=self.args.batch_size,num_workers=8,shuffle=True), self.args.datasets.train)
    
    def val_dataloader(self):
        return HeavenDataLoader(DataLoader(self.args.datasets.val,batch_size=self.args.batch_size,num_workers=8,shuffle=False), self.args.datasets.val)
    
    def test_dataloader(self):
        return HeavenDataLoader(DataLoader(self.args.datasets.test,batch_size=1,num_workers=8,shuffle=False), self.args.datasets.test)
    
    def run_batch(self, batch, split='train', batch_idx=-1):
        (specs, sample_rates), (parameters, gradients) = batch; pred = self(specs)
        parameters = [self.interface.from_dict(LoadJson(par)) for par in parameters]
        gradients = [LoadJson(g) if g!="" else {key:1 for key in self.interface.parameters()} for g in gradients]
        weights = {key:torch.tensor([g[key] for g in gradients], dtype=torch.float).type_as(pred) for key in self.interface.parameters()}
        true = torch.stack([p.to_tensor() for p in parameters]).type_as(pred).detach(); w = {k:w.detach() for k,w in weights.items()}
        loss = self.criteria(pred, true, w)
        result = {
            'src': batch,
            'prd': pred,
            'tgt': true,
            'loss': loss,
            'stats': {
                self.criteria.default_loss_type: float(loss.item()),
            }
        }
        if split!='train':
            for loss_type in LOSSES_MAPPING:
                result['stats'][loss_type] = float(
                    self.criteria(pred, true, w, loss_type).item()
                )
        for stat in result['stats']:
            self.log(f"{split}_{stat}", result['stats'][stat], sync_dist=True, prog_bar=(stat==self.criteria.default_loss_type), on_epoch=True, logger=True)
        return result

    def training_step(self, train_batch, batch_idx):
        result = self.run_batch(train_batch, split='train', batch_idx=batch_idx)
        if torch.any(torch.isnan(result['loss'])):
            return None
        else:
            return result['loss']
        
    def validation_step(self, val_batch, batch_idx):
        result = self.run_batch(val_batch, split='valid', batch_idx=batch_idx)
        if batch_idx*self.args.batch_size < self.args.examples:
            inf_epoch = pjoin("examples", "%05d"%self.current_epoch)
            CreateFolder(inf_epoch)
            for i, (prd, tgt) in enumerate(zip(result['prd'], result['tgt'])):
                instance = batch_idx * len(val_batch) + i
                preset_prd = self.interface.from_tensor(prd).to_dict()
                preset_tgt = self.interface.from_tensor(tgt).to_dict()
                SaveJson(preset_prd, pjoin(inf_epoch,"%03d_pd.json"%instance),indent=4)
                SaveJson(preset_tgt, pjoin(inf_epoch,"%03d_gt.json"%instance),indent=4)
    
    def test_step(self, test_batch, batch_idx):
        result = self.run_batch(test_batch, split='test', batch_idx=batch_idx)
        for i, (prd, tgt) in enumerate(zip(result['prd'], result['tgt'])):
            inf_epoch = pjoin("inference", self.args.identifier)
            CreateFolder(inf_epoch)
            for i, (prd, tgt) in enumerate(zip(result['prd'], result['tgt'])):
                instance = batch_idx * len(test_batch) + i
                preset_prd = self.interface.from_tensor(prd).to_dict()
                preset_tgt = self.interface.from_tensor(tgt).to_dict()
                SaveJson(preset_prd, pjoin(inf_epoch,"%03d_pd.json"%instance),indent=4)
                SaveJson(preset_tgt, pjoin(inf_epoch,"%03d_gt.json"%instance),indent=4)

def SplitDatasets(dataset, train=0.7, val=0.2, test=0.1, cut=-1):
    assert( 0<=train<=1 and 0<=val<=1 and 0<=test<=1 and abs((train+val+test)-1)<=1e-6 )
    n = len(dataset); indices = [i for i in range(n)]; np.random.shuffle(indices); datasets = {}
    num_train = int(train*n); datasets['train'] = Subset(dataset, indices[:num_train])
    num_val = int(val*n); datasets['val'] = Subset(dataset, indices[num_train:num_train+num_val])
    datasets['test'] = Subset(dataset, indices[num_train+num_val:])
    for name, dataset in datasets.items():
        datasets[name] = HeavenDataset(dataset)
    if cut >= 0:
        for name, dataset in datasets.items():
            datasets[name] = dataset[:cut]
    return MemberDict(datasets)

def Identifier(args):
    return f"{args.backbone}_{args.model}_{FORMATTED_TIME('%Y-%m-%d_%H.%M')}" if args.identifier is None else args.identifier