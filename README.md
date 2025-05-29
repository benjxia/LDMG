# LDMG - Latent Diffusion for (Continuous Domain) Music Generation

CSE 153/253 Spring 2025

## Environment Setup

### Installing Conda

Follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Installing dependencies

```bash
conda env create --file=environment.yml
```

This will create a local conda environment named "ldmg".

Run the following to activate the conda environment.

```bash
conda activate ldmg
```

## Dataset Setup for Training

```
wget https://os.unil.cloud.switch.ch/fma/fma_medium.zip

unzip fma_medium.zip
```

## Running our code

### Training the VAE-GAN

This will also create an additional directory `lightning_logs`

```bash
python train.py --mode=VAE --checkpoint=<path_to_checkpoint> --data_path=<path_to_data>
```

### Training an unconditioned Latent Diffusion Model
```bash
python train.py --mode=LDM --vae_checkpoint=<path_to_vae_checkpoint> --checkpoint=<path_to_checkpoint> --data_path=<path_to_data>
```
### Open Tensorboard dashboard (Only run after the directory has been created)
```bash
tensorboard --logdir lightning_logs/
```
