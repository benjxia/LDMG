#!/usr/bin/python3
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import os

from ldm.data.fma_dataset import MusicDataModule
from ldm.modules.ldm import AudioLDM
from ldm.modules.vae_gan import AudioVAEGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='train.py',
                    description='Trains the latent diffusion model',
                    epilog='Good luck running my shit code - ben')
    parser.add_argument('-m', '--mode', help='Whether to train VAE, LDM or CLDM, takes values VAE or LDM or CLDM', type=str, required=True) # Whether to train VAE-GAN or LDM <vae>
    parser.add_argument('-vg', '--vae_checkpoint', default=None, type=str)
    parser.add_argument('-c', '--checkpoint', default=None, type=str) # Path to checkpoint
    parser.add_argument('-d', '--data_path', default=None, type=str, required=True)
    parser.add_argument('-e', '--epochs', default=5, type=int) # Number of epochs to train for
    parser.add_argument('-b', '--batch_size', default=2, type=int) # Batch size for training
    parser.add_argument('-s', '--sample_rate', default=16000, type=int)
    parser.add_argument('-a', '--audio_dur', default=10, type=int) # Audio duration to train on, should be <= 30 seconds
    parser.add_argument('-p', '--discriminator_pause', default=0, type=int)

    args = parser.parse_args()

    if args.mode == 'VAE':
        if args.checkpoint is None:
            model = AudioVAEGAN(1, kl_weight=1e-2, sample_rate=args.sample_rate, discriminator_pause=args.discriminator_pause)
        else:
            model = AudioVAEGAN.load_from_checkpoint(args.checkpoint, strict=False, kl_weight=1e-2)

        data_module = MusicDataModule(
            data_dir=args.data_path,
            target_sr=args.sample_rate,
            clip_duration=args.audio_dur,
            batch_size=args.batch_size,
            num_workers=os.cpu_count())
        data_module.setup()

    elif args.mode == 'LDM':
        if args.vae_checkpoint is None:
            raise ValueError
        if args.checkpoint is None:
            model = AudioLDM(
                n_dit_layers=16,
                audiovae_ckpt_path=args.vae_checkpoint
            )
        else:
            model = AudioLDM.load_from_checkpoint(args.checkpoint, audiovae_ckpt_path=args.vae_checkpoint, n_dit_layers=8)

        # TODO: do something else for conditioned generation
        data_module = MusicDataModule(
            data_dir=args.data_path,
            target_sr=args.sample_rate,
            clip_duration=args.audio_dur,
            batch_size=args.batch_size,
            num_workers=os.cpu_count())
        data_module.setup()
    elif args.mode == 'CLDM': # Conditioned LDM
        raise NotImplementedError

    ckpt_callback = ModelCheckpoint(
        every_n_train_steps=100,
        save_top_k=1
    )

    trainer = L.Trainer(callbacks=[ckpt_callback], max_epochs=args.epochs, log_every_n_steps=5)
    trainer.fit(model, datamodule=data_module)
