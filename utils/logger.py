import logging
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import timedelta, datetime

class Logger:
    def __init__(self, config):
        self.config = config
        self.dataset_name = '_'.join(self.config['datasets_config']['de_type'])
        mmdd = datetime.now().strftime('%m%d')

        self.result_name = f"{self.dataset_name}_{self.config['name']}_{mmdd}"
        self.log_dir = Path(self.result_name)

        self.log_dir.mkdir(exist_ok=True)
        
        self._logger = self._init_logger()
        self.tb_logger = SummaryWriter(log_dir=str(self.log_dir))

        self.best_metrics = {}
        
        
    def _init_logger(self):
        logger = logging.getLogger('train')
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s INFO: %(message)s', 
                                           datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        log_file = self.log_dir / f"{self.config['name']}_train.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
        return logger

    def info(self, msg):
        """Log info level message"""
        self._logger.info(msg)
        
        


    def log_dataset_info(self, train_dataset_lens, val_dataset_lens):
        self.info(
            f"[{self.config['name']}..] Datasets:"
            f"\n - Degradation types: {self.dataset_name}"
            f"\n - Train dataset size: {train_dataset_lens}"
            f"\n - Validation dataset size: {val_dataset_lens}"
        )
        return  
    
    def log_training_info(self, num_iter_per_epoch, total_epochs, total_iters):
        self.info(
            f"[{self.config['name']}..] Training details:"
            f"\n - Batch size per GPU: {self.config['datasets_config']['batch_size_per_gpu']}"
            f"\n - Total epochs: {total_epochs}"
            f"\n - Total iterations: {total_iters}"
        )
        
    def log_training_status(self, current_iter, start_time, lr, loss):
        elapsed = time.time() - start_time
        iter_time = elapsed / current_iter
        total_iters = self.config['train']['total_iter']
        eta_seconds = (total_iters - current_iter) * iter_time


        eta = str(timedelta(seconds=eta_seconds)).split('.')[0]


        self.info(
            f"[{self.config['name']}..][iter:{current_iter:,}, "
            f"lr:({lr:.3e}), "
            f"loss:({loss/self.config['logger']['print_freq']:.4f})] "
            f"[eta: {eta}]"
        )
        

    def log_validation_results(self, metrics, current_iter):
        """Log validation metrics"""
        if not self.best_metrics:
            self.best_metrics = metrics
        
        is_best = False

        for dtype, dtype_metrics in metrics.items():
            if dtype_metrics['PSNR'] > self.best_metrics[dtype]['PSNR']:
                self.best_metrics[dtype] = dtype_metrics
                is_best = True
            metrics_str = ', '.join([
                f"{k}: {v:.4f}" for k, v in dtype_metrics.items()
            ])
            best_metrics_str = ', '.join([
                f"{k}: {v:.4f}" for k, v in self.best_metrics[dtype].items()
            ])
            self.info(f"[{self.config['name']}..] metrics [{dtype}] {metrics_str}")

            self.info(f"[{self.config['name']}..] best metrics [{dtype}] {best_metrics_str}")
            
        return is_best
    
    def close(self):
        """Close tensorboard writer"""
        self.tb_logger.close()
