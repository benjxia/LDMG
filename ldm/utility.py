from torch import nn
from torch.nn import functional as F

def get_activation(activation='relu'):
    # TOOD: add more - but I'm lazy - ben
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        # I'm too lazy to add more activations, these are probably all we need
        return F.relu

# Useful for nn.sequential and shit
def get_activation_module(activation='relu'):
    # TOOD: add more - but I'm lazy - ben
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        # I'm too lazy to add more activations, these are probably all we need
        return nn.ReLU()
