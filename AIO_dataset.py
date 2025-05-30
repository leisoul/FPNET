import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import yaml

class TrainDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        with open(config['datasets_root'], 'r', encoding='utf-8') as f:
            self.root = yaml.safe_load(f)['train']
        self.patch_size = config['patch_size']
        self.de_type = config['de_type']
       
        self.data = {}

        self._load_datasets(self.de_type)


    def _load_datasets(self, de_type):
        for dataset_name in de_type:
        
            if dataset_name not in self.root:
                continue

            dataset = self.root.get(dataset_name)

            image_pairs = self._get_image_pairs(
                dataset['gt_path'], 
                dataset['deg_path']
            )
            
            self.data= {
                'ground_truth': image_pairs[0],
                'degradation': image_pairs[1],
            }

    def _get_image_pairs(self, gt_path, deg_path):
        VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        pairs = tuple(zip(*(
            (os.path.join(gt_path, f), os.path.join(deg_path, f))
            for f in os.listdir(gt_path)
            if f.lower().endswith(VALID_EXTENSIONS) and os.path.exists(os.path.join(deg_path, f))
        )))
        
        return pairs or ((), ())
    
    def _extract_patch(self, img1, img2):
        height, width = img1.shape[:2]
        top = np.random.randint(0, height - self.patch_size + 1)
        left = np.random.randint(0, width - self.patch_size + 1)
        
        patch1 = img1[top:top + self.patch_size, 
                     left:left + self.patch_size]
        patch2 = img2[top:top + self.patch_size, 
                     left:left + self.patch_size]
        return patch1, patch2

    def _load_and_process_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):

   
        clean_root = self.data['ground_truth'][idx]
        degrade_root = self.data['degradation'][idx]

        clean_img = self._load_and_process_image(clean_root)
        degrade_img = self._load_and_process_image(degrade_root)

        clean_patch, degrade_patch = self._extract_patch(clean_img, degrade_img)
        
        mode = random.randint(0, 7)
        clean_patch = data_augmentation(clean_patch, mode).copy()
        degrade_patch = data_augmentation(degrade_patch, mode).copy()



        
        return (transforms.ToTensor()(clean_patch),
                transforms.ToTensor()(degrade_patch),
                )

    def __len__(self):
      
        return len(self.data['ground_truth'])

class ValDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        with open(config['datasets_root'], 'r', encoding='utf-8') as f:
            self.root = yaml.safe_load(f)['val']
        self.patch_size = config['patch_size']
        self.de_type = config['de_type']

        self.ground_truth = []
        self.degradation = []
        self.labels = []
        self.dataset_types = []
        
        self._load_datasets(self.de_type)

    def _load_datasets(self, de_type):
        for dataset_name in de_type:
            if dataset_name not in self.root:
                continue

            dataset = self.root.get(dataset_name)

            image_pairs = self._get_image_pairs(
                dataset['gt_path'], 
                dataset['deg_path']
            )
            
            self.ground_truth.extend(image_pairs[0])
            self.degradation.extend(image_pairs[1])
            self.labels.extend([dataset['label']] * len(image_pairs[0]))
            self.dataset_types.extend([dataset_name] * len(image_pairs[0]))

    def _get_image_pairs(self, gt_path, deg_path):
        gt_images = {
            f: os.path.join(gt_path, f)
            for f in os.listdir(gt_path)
            if f.endswith(('.jpg', '.png'))
        }
        
        paired_gt = []
        paired_deg = []
        
        for img_name in gt_images:
            deg_path_full = os.path.join(deg_path, img_name)
            if os.path.exists(deg_path_full):
                paired_gt.append(gt_images[img_name])
                paired_deg.append(deg_path_full)
                
        return paired_gt, paired_deg

    def _extract_patch(self, img1, img2):
        height, width = img1.shape[:2]
        top = np.random.randint(0, height - self.patch_size + 1)
        left = np.random.randint(0, width - self.patch_size + 1)
        
        patch1 = img1[top:top + self.patch_size, 
                     left:left + self.patch_size]
        patch2 = img2[top:top + self.patch_size, 
                     left:left + self.patch_size]
        return patch1, patch2

    def _load_and_process_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        clean_img = self._load_and_process_image(self.ground_truth[idx])
        degrade_img = self._load_and_process_image(self.degradation[idx])
        
        clean_patch, degrade_patch = self._extract_patch(clean_img, degrade_img)

        return (transforms.ToTensor()(clean_patch),
                transforms.ToTensor()(degrade_patch),
                self.labels[idx],
                self.dataset_types[idx]
                )

    def __len__(self):
        return len(self.ground_truth)

def data_augmentation(image, mode):
    augmentations = {
        0: lambda img: img,                                 # original
        1: lambda img: np.flipud(img),                      # flip up and down
        2: lambda img: np.rot90(img),                       # rotate counterclockwise 90 degrees
        3: lambda img: np.flipud(np.rot90(img)),            # rotate 90 degrees and flip up and down
        4: lambda img: np.rot90(img, k=2),                  # rotate 180 degrees
        5: lambda img: np.flipud(np.rot90(img, k=2)),       # rotate 180 degrees and flip
        6: lambda img: np.rot90(img, k=3),                  # rotate 270 degrees
        7: lambda img: np.flipud(np.rot90(img, k=3)),      # rotate 270 degrees and flip
    }
    return augmentations[mode](image)

