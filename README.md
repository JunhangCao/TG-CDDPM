# TG-CDDPM: Text-Guided Antimicrobial Peptides Generation Based on Conditional Denoising Diffusion Probabilistic Model

# Contents

- [Installation](#Installation)
- [Data and Checkpoints](#Data and Checkpoints)
- [Train](#Train)
- [Sampling](#Sampling)
- [Result](#Result)
- [Molecular Dynamics Simulation](#Molecular Dynamics Simulation)
- [Reference](#Reference)

## Installation

#### pull project

- git clone https://github.com/JunhangCao/TG-CDDP.git

#### install requirments

- pip install -r requirements.txt

## Data and Checkpoints

- the required data and checkpoints can be downloaded: 

- https://drive.google.com/drive/folders/1aN3cScePnxq368pL6ymeKFj8EFenPBj0?usp=drive_link

## Train

- if you want re-train the whole framework. <br/>

#### stage 1

```
cd train
python TextPepAlignment.py
```

#### stage 2

- Adjusting forward_backward in __utils.train_utils.py__ and loss function in __gaussian_diffusion.py__ for adapter.  

```
def forward_backward():  
    ...  
    if k == 'input_ids':  
        micro_cond[k] = v[i: i + self.microbatch].to(self.device)
    ...  
```

```
def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):  
    ...  
    terms["loss"] = terms["mse"]  
    ...  
```

```
cd train
python diffusion_train.py
```

#### stage 3
- Collecting Checkpoints of pre-trained DDPM.  

- Adjusting forward_backward in __utils.train_utils.py__ and loss function in __gaussian_diffusion.py__ for text-guided fine-tuning DDPM. 

  ```
  - def forward_backward():  
      ...  
      if k == 'input_ids':  
          micro_cond[k] = v[i: i + self.microbatch].to(self.device)
      else:  
          with torch.no_grad():  
              text_features = self.text_encoder(v[i: i + self.microbatch].to(self.device)) 
              text_features_norm = text_features / text_features.norm(dim=-1,keepdim=True) 
              text_features_norm = text_features_norm.unsqueeze(1).repeat(1, 50, 1)  
              timesteps = torch.tensor([0] * text_features.shape[0], device=self.device)  
              fac_text_z = self.facilitator(inputs_embeds=text_features_norm,timesteps=timesteps)  
              fac_text_z_norm = fac_text_z / fac_text_z.norm(dim=-1, keepdim=True)  
              micro_cond['self_condition'] = text_features_norm
  ```

  ```
  def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):  
     terms["loss"] = terms["mse"] + decoder_nll  
  
  ```

  ```
  cd train
  python diffusion_train.py
  ```

## Sampling
```
# -num_samples --- number of generated peptides
cd model
python sampling.py -num_samples 100
```

## Result

- The generated peptides was saved in __/sample/samples.fasta__ like:

```
>1
SWKSMAKKLKEYLKQRA
>2
GLRKRLRKFRNKIKQKLKKIMEKL
>3
GLRKALRKFRNKIKELKKI
>4
WLRRIGKGVKIIGGLDHL
>5
GLRKRLIKEKLKKI
>6
GLRKRLRKARNKIKEKLKKI
>7
SWASMAKKLKEYMEKLKQRAMEKLMEKL
```

## Molecular Dynamics Simulation

- We selected generated peptides to conduct MD simulation, and the process of selection was shown in Paper. 
- Before starting the procedure of MD simulation, 3D structures of generated peptides was predicted by the website of AlphaFold 2 [1]: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
- The secondary structure predicted in first ranking score was saved in __.pdb__ for GROMACS [2] to run MD simulation. 
- We used CHARMM-gui [3], a cyberinfrastructure that aims to simplify and standardize the precess of MD preparation, to prepare field force, temperature and pressure: https://www.charmm-gui.org/
- Runing 100 ns MD simulation in GROMACS, the simple process in Linux system are as follows, the tutorial of simulation are shown in official website in GROMACS.

```
mkdir {path}/run
cd {path}/run

cp {path}/gromacs/step5_input.gro .       
cp {path}/gromacs/step5_input.pdb .        
cp {path}/gromacs/topol.top .        
cp {path}/gromacs/index.ndx .           
cp -r {path}/gromacs/toppar .

# step 1
gmx grompp -f step6.0_minimization.mdp -o minimization.tpr -c step5_input.gro -r step5_input.gro -p topol.top
gmx mdrun -v deffnm minimization

# step 2
gmx grompp -f step6.1_equilibration.mdp -o step6.1_equilibration.tpr -c minimization.gro -r step5_input.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step6.1_equilibration

# step 3
gmx grompp -f step6.2_equilibration.mdp -o step6.2_equilibration.tpr -c step6.1_equilibration.gro -r step5_input.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step6.2_equilibration

# step 4
gmx grompp -f step6.3_equilibration.mdp -o step6.3_equilibration.tpr -c step6.2_equilibration.gro -r step5_input.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step6.3_equilibration

# step 5
gmx grompp -f step6.4_equilibration.mdp -o step6.4_equilibration.tpr -c step6.3_equilibration.gro -r step5_input.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step6.4_equilibration

# step 6
gmx grompp -f step6.5_equilibration.mdp -o step6.5_equilibration.tpr -c step6.4_equilibration.gro -r step5_input.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step6.5_equilibration

# step 7
gmx grompp -f step6.6_equilibration.mdp -o step6.6_equilibration.tpr -c step6.5_equilibration.gro -r step5_input.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step6.6_equilibration

# step 8
gmx grompp -f step7_production.mdp -o step7_production.tpr -c step6.6_equilibration.gro -t step6.6_equilibration.cpt -p topol.top -n index.ndx

# modifying dt and nsteps in step7_production.mdp can adjust the time of simulation e.g. dt * nsteps = time ps = time * 10-3 ns
gmx mdrun -s step7_production -cpi
```

- The trajectory file, __traj.trr__, can be produced. We used this file to validate the effect of simulation.

 ## Reference

```
[1] Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold[J]. nature, 2021, 596(7873): 583-589.
[2] Van Der Spoel D, Lindahl E, Hess B, et al. GROMACS: fast, flexible, and free[J]. Journal of computational chemistry, 2005, 26(16): 1701-1718.
[3] Jo S, Kim T, Iyer V G, et al. CHARMM‐GUI: a web‐based graphical user interface for CHARMM[J]. Journal of computational chemistry, 2008, 29(11): 1859-1865.
```

