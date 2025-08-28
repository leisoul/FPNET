Quick Run
You can simply demo on my space of Hugging Face

Dehazing
Deraindrop
Deblurring
Or test on local environment:
To test the pre-trained models of Deraindrop, Dehaze, Deblurring on your own images, run

python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
Here is an example to perform Deraindrop:

python demo.py --input_dir './demo_samples/deraindrop' --result_dir './demo_results' --weights './pretrained_model/deraindrop_model.pth'
All pre-trained models can be downloaded at pretrained_model/README.md or here

Train
To train the restoration models of Deraindrop, Dehaze and Deblurring. You should check the following components:

training.yaml:

# Training configuration
GPU: [0,1,2,3]

VERBOSE: False

MODEL:
  MODE: 'Deblur'

# Optimization arguments.
OPTIM:
  BATCH: 2
  EPOCHS: 150
  # NEPOCH_DECAY: [10]
  LR_INITIAL: 2e-4
  LR_MIN: 1e-6
  # BETA1: 0.9

TRAINING:
  VAL_AFTER_EVERY: 1
  RESUME: False
  TRAIN_PS: 256
  VAL_PS: 256
  TRAIN_DIR: './datasets/deraindrop/train'       # path to training data
  VAL_DIR: './datasets/deraindrop/test' # path to validation data
  SAVE_DIR: './checkpoints'           # path to save models and images

Details of Hyperparameters

 -------------------------------------------------
 GoPro dataset:
 Training patches: 33648 (2103 x 16)
 Validation: 1111
 Initial learning rate: 2e-4
 Final learning rate: 1e-6
 Training epochs: 150 (120 is enough)
Training time (on single 2080ti): about 10 days

 Raindrop dataset:
 Training patches: 6888 (861 x 8)
 Validation: 1228 (307 x 4)
 Initial learning rate: 2e-4
 Final learning rate: 1e-6
 Training epochs: 150 (100 is enough)
Training time (on single 1080ti): about 2.5 days

 IO-Haze dataset:
 Training patches: 15000 (75 x 200)
 Validation: 55
 Initial learning rate: 1e-4
 Final learning rate: 1e-6
 Training epochs: 150 (50 is enough)
Training time (on single 1080ti): about 3 days
 -------------------------------------------------
Dataset:
The preparation of dataset in more detail, see datasets/README.md.

Train:
If the above path and data are all correctly setting, just simply run:

python train.py
Test (Evaluation)
To test the models of Deraindrop, Dehaze, Deblurring with ground truth, see the test.py and run

python test.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models --dataset type_of_task --gpus CUDA_VISIBLE_DEVICES
Here is an example to perform Deraindrop:

python test.py --input_dir './datasets/' --result_dir './test_results/' --weights './pretrained_model/deraindrop_model.pth' --dataset deraindrop --gpus '0'
To test the PSNR and SSIM of Deraindrop, see the evaluation_Y.py and run

python evaluation_Y.py --input_dir path_to_restored_images --gt_dir path_to_gt_images
Here is an example:

python valuation_Y.py --input_dir './test_results/deraindrop' --gt_dir './demo_samples/deraindrop'
And to test the PSNR and SSIM of Dehaze and Deblur, see the evaluation_RGB.m

Results
Result Tables (Click to expand)
Visual Comparison
Visual Comparison Figures (Click to expand)
More visual results can be downloaded at here.
