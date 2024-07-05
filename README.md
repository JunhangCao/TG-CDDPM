# TG-CDDPM: Text-Guided Antimicrobial Peptides Generation Based on Conditional Denoising Diffusion Probabilistic Model



## Installation

#### pull project

git clone https://github.com/JunhangCao/TG-CDDP.git

#### install requirments

pip install -r requirements.txt



## Data and Checkpoints

the required data and checkpoints can be downloaded: 

https://drive.google.com/drive/folders/1aN3cScePnxq368pL6ymeKFj8EFenPBj0?usp=drive_link

## File Structure  
└─TG-CDDP-main  
    │  README.md  
    │  requirements.txt  
    │  
    ├─config  
    │  backen_config.py  
    │  base_config.py  
    │  config.json  
    │  
    ├─mapping  
    │  vocab.txt  
    │  
    ├─model  
    │  backend.py  
    │  diffusion.py  
    │  diffusion_sample.py  
    │  fp16_utils.py  
    │  gaussian_diffusion.py  
    │  losses.py  
    │  nn.py  
    │  resample.py  
    │  respace.py  
    │  rounding.py  
    │  sampling.py  
    │  
    ├─train  
    │  diffusion_train.py  
    │  TexPepAlignment.py  
    │  
    └─utils  
    │  script_utils.py  
    │  tokenizer.py  
    │  train_util.py  
  
## Train
if you intend to re-train the whole framework. <br/>

#### stage 1
run TexPepAlignment.py for training peptide encoder and text encoder.

#### stage 2
adjust the 'forward_backward' function in 'utils.train_utils.py' and the loss function in 'gaussian_diffusion.py' for adapter.  
def forward_backward():  
>>...  
>>if k == 'input_ids':  
>>>>micro_cond[k] = v[i: i + self.microbatch].to(self.device)
  
>>...  

def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):  
>>...  
>>terms["loss"] = terms["mse"]  
>>...  
run diffusion_train.py for training adapter.  

#### stage 3
pretrained DDPM has been prepared.  
adjust the 'forward_backward' function in 'utils.train_utils.py' and the loss function in 'gaussian_diffusion.py' for text-guided fine-tuning DDPM.  
def forward_backward():  
>>...  
>>if k == 'input_ids':  
>>>>micro_cond[k] = v[i: i + self.microbatch].to(self.device)
  
>>else:  
>>>>with torch.no_grad():  
>>>>>>text_features = self.text_encoder(v[i: i + self.microbatch].to(self.device))  
>>>>>>text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)  
>>>>>>text_features_norm = text_features_norm.unsqueeze(1).repeat(1, 50, 1)  
>>>>>>timesteps = torch.tensor([0] * text_features.shape[0], device=self.device)  
>>>>>>fac_text_z = self.facilitator(inputs_embeds=text_features_norm, timesteps=timesteps)  
>>>>>>fac_text_z_norm = fac_text_z / fac_text_z.norm(dim=-1, keepdim=True)  
>>>>>>micro_cond['self_condition'] = text_features_norm
  
>>...  
  
def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):  
>>...  
>>terms["loss"] = terms["mse"] + decoder_nll  
>>...  
run diffusion_train.py for fine-training text-guided DDPM.

## Sampling
set the text description using 'test_text = ['xxx']' in 'model.sampling.py'  
run sampling.py for sampling.
