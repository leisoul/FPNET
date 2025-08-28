

# Training configuration
GPU: [0,1,2,3]

python train.py

# Optimization arguments.
OPTIM:
  BATCH: 2
  EPOCHS: 150
  # NEPOCH_DECAY: [10]
  LR_INITIAL: 2e-4
  LR_MIN: 1e-6
  # BETA1: 0.9


 -------------------------------------------------
 GoPro dataset:
 Training patches: 33648 (2103 x 16)
 Validation: 1111
 Initial learning rate: 2e-4
 Final learning rate: 1e-6
 Training epochs: 150 (120 is enough)
Training time (on single 2080ti): about 10 days



Train:
If the above path and data are all correctly setting, just simply run:

python train.py
Test (Evaluation)
To test the models of , Deblurring with ground truth, see the test.py and run

python test.py -S
Here is an example to perform Deraindrop:



python valuation_Y.py --input_dir './test_results/deraindrop' --gt_dir './demo_samples/deraindrop'


