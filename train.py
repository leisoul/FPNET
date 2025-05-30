import argparse
import torch
import yaml
import tqdm
from tqdm import tqdm
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from models.FPNet import FPNet
from models.losses import L1Loss

class Trainer:
    def __init__(self, config_path):
        self.setup(config_path)
        self.initialize_components()
        
    def setup(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        
            
    def initialize_components(self):
        """Initialize model, optimizer, datasets and other components"""
        from utils.metrics import create_metrics
        from AIO_dataset import TrainDataset, ValDataset
        
        self.model = FPNet(enc_blk_nums = [2, 2, 4, 8],middle_blk_num = 12,dec_blk_nums = [2, 2, 2, 2],FGM_nums = 1,backbone = 'NAFNet').to('cuda')
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.config['train']['optim_g']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            **self.config['train']['scheduler']
        )

        self.criterion = L1Loss()
        self.metrics_cls = create_metrics(self.config)
        
        self.train_dataset = TrainDataset(self.config['datasets_config'])
        self.val_dataset = ValDataset(self.config['datasets_config'])

        self.train_loader = self.create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self.create_dataloader(self.val_dataset, shuffle=False)
        
        self.scaler = GradScaler()
        
        self.current_iter = 0

        self.total_iters = int(self.config['train']['total_iter'])
            
    def create_dataloader(self, dataset, shuffle):
        """Create dataloader with optimal settings"""
        return DataLoader(
            dataset=dataset,
            batch_size=self.config['datasets_config']['batch_size_per_gpu'],
            shuffle=shuffle,
            num_workers=self.config['datasets_config']['num_worker_per_gpu'],
            pin_memory=True,
        )


    def train_step(self, input_data):
        self.optimizer.zero_grad(set_to_none=True)
        
        target, input_ = (d.to('cuda', non_blocking=True) for d in input_data[:2])
        
        with torch.amp.autocast('cuda'):
            output = self.model(input_)
            loss = self.criterion(output, target)
            
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        metrics = {dtype: {met.__class__.__name__: [] for met in self.metrics_cls} 
                  for dtype in self.config['datasets_config']['de_type']}
        
        for data_val in (self.val_loader):
            target = data_val[0].to('cuda', non_blocking=True)
            input_ = data_val[1].to('cuda', non_blocking=True)
            data_types = data_val[3]
            
            with torch.amp.autocast('cuda'):
                output = self.model(input_)
            
            for dtype in self.config['datasets_config']['de_type']:
                for idx, data_type in enumerate(data_types):
                    if data_type == dtype:
                        for met in self.metrics_cls:
                            metric_name = met.__class__.__name__
                            metrics[dtype][metric_name].append(
                                met(output[idx], target[idx]).item()
                            )
                            
        return {dtype: {k: sum(v)/len(v) if v else 0 
                      for k, v in dtype_metrics.items()}
               for dtype, dtype_metrics in metrics.items()}
        
    def train(self):
        pbar = tqdm(total=self.total_iters, desc='Training')
        while self.current_iter <= self.total_iters:

            self.model.train()
            
            for batch_idx, train_data in enumerate(self.train_loader):
                self.current_iter += 1
                pbar.update(1)
                if self.current_iter > self.total_iters:
                    break
                    
                self.train_step(train_data)


                if self.current_iter % self.config['val']['val_freq'] == 0 or self.current_iter == 1000:
                    metrics = self.validate()
                    for dtype, dtype_metrics in metrics.items():

                        metrics_str = ', '.join([
                            f"{k}: {v:.4f}" for k, v in dtype_metrics.items()
                        ])
                        print(f"\n metrics [{dtype}] {metrics_str}")

        pbar.close() 
        Path("checkpoint").mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': self.model.state_dict()}, "checkpoint/model_latest.pth")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-yml', type=str, default='./option/test.yml',help='Path to option YAML file.')
    # parser.add_argument('-yml', type=str, default='./option/AIO_32.yml',help='Path to option YAML file.')
    args = parser.parse_args()
    
    trainer = Trainer(args.yml)
    trainer.train()
    print("====== Training Complete ==========")

if __name__ == '__main__':
    main()