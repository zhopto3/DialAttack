import torch

class Greedy_Decoder(torch.nn.Module):
    
    def __init__(self):
        pass

    def forward(self):
        pass
        #No softmax necessary

class BeamSearch_Decoder(torch.nn.Module):
    
    def __init__(self, beam_size):
        pass

    def forward(self):
        pass
        #Softmax is necessary...