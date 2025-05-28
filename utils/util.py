import torch
from pathlib import Path



class CheckpointHandler:
    def __init__(self, model, optimizer, scheduler, device, save_dir, name, logger):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.name = name
        self.logger = logger

    def save(self,model, optimizer, scheduler, current_iter, epoch, is_best=False):
        """Save model checkpoint."""
        # 確保存檔目錄存在
        save_dir = Path(self.save_dir) / f"{self.name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iter': current_iter,
            'epoch': epoch,
        }
        
        # 保存最新的檢查點
        save_path = save_dir / "model_latest.pth"
        torch.save(state, save_path)
        self.logger.info(f"Saved latest checkpoint to {save_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = save_dir / "model_best.pth"
            torch.save(state, best_path)
            self.logger.info(f"Saved best model to {best_path}")
