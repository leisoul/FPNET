import os
import argparse
import torch
import yaml
import math
import time
from tqdm import tqdm
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from utils.util import  CheckpointHandler
from utils.logger import Logger

class Trainer:
    def __init__(self, config_path):
        self.setup(config_path)
        self.initialize_components()
        
    def setup(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu_id']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = Logger(self.config)
        self.logger.log_config()
        
        if torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Using CPU")
            
    def initialize_components(self):
        """Initialize model, optimizer, datasets and other components"""
        from models import create_model, create_loss
        from utils.metrics import create_metrics
        from AIO_dataset import TrainDataset, ValDataset
        
        self.model = create_model(self.config, self.logger).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.config['train']['optim_g']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            **self.config['train']['scheduler']
        )

        self.criterion = create_loss(self.config, self.logger)
        self.metrics_cls = create_metrics(self.config)
        
        self.train_dataset = TrainDataset(self.config['datasets_config'])
        self.val_dataset = ValDataset(self.config['datasets_config'])

        self.train_loader = self.create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self.create_dataloader(self.val_dataset, shuffle=False)
        
        self.logger.log_dataset_info(len(self.train_dataset), len(self.val_dataset))


        self.current_iter = 0
        self.epoch = 1
        self.scaler = GradScaler()
        
        self.num_iter_per_epoch = math.ceil(len(self.train_dataset) / 
                                          self.config['datasets_config']['batch_size_per_gpu'])
        self.total_iters = int(self.config['train']['total_iter'])
        self.total_epochs = math.ceil(self.total_iters / self.num_iter_per_epoch)
        

        
        self.checkpoint_handler = CheckpointHandler(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            save_dir="checkpoint",
            name=self.logger.result_name, 
            logger=self.logger
        )
        
        self.logger.log_training_info(self.num_iter_per_epoch, self.total_epochs, self.total_iters)
            
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
        
        target, input_ = (d.to(self.device, non_blocking=True) for d in input_data[:2])
        
        if self.current_iter < self.config['train']['warmup_iter']:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_warmup_lr()
        
        with torch.amp.autocast('cuda'):
            output = self.model(input_)
            loss = self.criterion(output, target)
            
        self.scaler.scale(loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.scheduler.step()
        
        return loss.item()
        
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        metrics = {dtype: {met.__class__.__name__: [] for met in self.metrics_cls} 
                  for dtype in self.config['datasets_config']['de_type']}
        
        for data_val in tqdm(self.val_loader, desc='Validating'):
            target = data_val[0].to(self.device, non_blocking=True)
            input_ = data_val[1].to(self.device, non_blocking=True)
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
                            
        # Average metrics
        return {dtype: {k: sum(v)/len(v) if v else 0 
                      for k, v in dtype_metrics.items()}
               for dtype, dtype_metrics in metrics.items()}
        
    def train(self):
        self.start_time = time.time()
        train_loss = 0
        
        try:
            while self.current_iter <= self.total_iters:
                self.model.train()
                
                for batch_idx, train_data in enumerate(self.train_loader):
                    self.current_iter += 1
                    if self.current_iter > self.total_iters:
                        break
                        
                    loss = self.train_step(train_data)
                    train_loss += loss
                    
                    if self.current_iter % self.config['logger']['print_freq'] == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        self.logger.log_training_status(
                            self.current_iter,
                            self.start_time,
                            lr,
                            train_loss
                        )
                        train_loss = 0


                    if self.current_iter % self.config['val']['val_freq'] == 0 or self.current_iter == 1000:
                        metrics = self.validate()
                        is_best = self.logger.log_validation_results(metrics, self.current_iter)

                        self.checkpoint_handler.save(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            current_iter=self.current_iter,
                            epoch=self.epoch,
                            is_best=is_best
                        )

                self.epoch += 1
        finally:
            self.logger.close()

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