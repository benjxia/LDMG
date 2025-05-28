# A bunch of default hyperparameters
DEFAULT_INPUT_SR = 16000
DEFAULT_LATENT_SR = 250 # Chosen because 16000 / 2^6 = 250, and we have an even number of 0.5x downsamples
DEFAULT_LATENT_CHANNELS = 32 # Seems to be a pretty standard value for this

DEFAULT_1D_KERNEL_SIZE = 7 # This seems to be standard practice for waveforms
DEFAULT_1D_PADDING = 3 # Padding necessary for kernel size 7 for exact halving of dimensions

DEFAULT_MAX_CHANNELS = 128

DEFAULT_AUDIO_DUR = 10 # In seconds
