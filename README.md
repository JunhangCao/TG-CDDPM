# TG-CDDPM: Text-Guided Antimicrobial Peptides Generation Based on Conditional Denoising Diffusion Probabilistic Model



## installation

#### pull project

git clone https://github.com/JunhangCao/TG-CDDP.git

#### install requirments

pip install -r requirements.txt



## Data and Checkpoints

the required data and checkpoints can be downloaded: 

https://drive.google.com/drive/folders/1aN3cScePnxq368pL6ymeKFj8EFenPBj0?usp=drive_link

## train
if you want re-train the whole framework. <br/>

#### stage 1
run TexPepAlignment.py for training peptide encoder and text encoder.

#### stage 2
adjust forward_backward in utils.train_utils.py and loss function in gaussian_diffusion.py for adapter.  
run diffusion_train.py for training adapter.

#### stage 3
pretrained DDPM has been prepared.  
adjust forward_backward in utils.train_utils.py and loss function in diffusion for text-guided fine-tuning DDPM.  
run diffusion_train.py for fine-training text-guided DDPM.

## sampling
set text description in 'test_text = ['xxx']' in sample.sampling.py  
run sampling.py for sampling.
