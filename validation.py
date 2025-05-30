import argparse
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, config_path):
        self.setup(config_path)
        self.initialize_components()
        
    def setup(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
            
    def initialize_components(self):
        from utils.metrics import create_metrics
        from AIO_dataset import  ValDataset
        
        self.model = create_model(self.config, self.logger).to('cuda')
        self.metrics_cls = create_metrics(self.config)
        self.val_dataset = ValDataset(self.config['datasets'])
        self.val_loader = self.create_dataloader(self.val_dataset, shuffle=False)


        weight = r'C:\model\mymodel\checkpoint\GOPRO_TEST_0201\model_latest.pth'

        checkpoint = torch.load(weight, weights_only=True)
        print("Checkpoint keys:", checkpoint.keys())  

        self.model.load_state_dict(checkpoint['state_dict'])






            
    def create_dataloader(self, dataset, shuffle):
        """Create dataloader with optimal settings"""
        return DataLoader(
            dataset=dataset,
            batch_size=self.config['datasets']['batch_size_per_gpu'],
            shuffle=shuffle,
            num_workers=self.config['datasets']['num_worker_per_gpu'],
            pin_memory=True,
        )



    @torch.no_grad()
    def validate(self):
        """Run validation"""
        self.model.eval()
        metrics = {dtype: {met.__class__.__name__: [] for met in self.metrics_cls} 
                  for dtype in self.config['datasets']['de_type']}
        
        for data_val in tqdm(self.val_loader, desc='Validating'):
            target = data_val[0].to('cuda')
            input_ = data_val[1].to('cuda')
            data_types = data_val[3]
            
            with torch.amp.autocast('cuda'):
                output = self.model(input_)
            
            for dtype in self.config['datasets']['de_type']:
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
        """Enhanced training loop with early stopping"""
        
        metrics = self.validate()
        for dtype, dtype_metrics in metrics.items():
            metrics_str = ', '.join([
                f"{k}: {v:.4f}" for k, v in dtype_metrics.items()
            ])
            self.logger.info(f"Validation [{dtype}] {metrics_str}")         

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-yml', type=str, default='./option/test.yml',help='Path to option YAML file.')
    args = parser.parse_args()
    
    trainer = Trainer(args.yml)
    trainer.train()
    print("====== Training Complete ==========")

if __name__ == '__main__':
    main()