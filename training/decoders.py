import torch

class Greedy_Decoder(torch.nn.Module):
    """Mostly inspired by the implementation of greedy decoding here, but changed a few things to work with my other modules:
      https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html"""
    def __init__(self):
        super().__init__()
        #Based on how I make my vocab in the data_util script, blank will always be 0
        self.blank=0

    def forward(self, batch_logits):
        #No softmax necessary
        #Transpose to get dimensions BXTimeStepsXVocabSize
        batch_logits = torch.transpose(batch_logits, 0, 1)
        #Get the highest likelihood token for each time step; dim: Bx# Time Steps
        decoded_seqs = torch.argmax(batch_logits, dim=-1)
        #Merge repeated char in each output (repeated characters can still occur in decoded text, but only when blank character "|" is between them)
        out_seq = []
        for seq in decoded_seqs:
            seq = torch.unique_consecutive(seq, dim=-1)

            #get rid of all the blank tokens in each sequence
            seq = seq[seq!=self.blank]

            out_seq.append(seq)
        return out_seq


class BeamSearch_Decoder(torch.nn.Module):
    
    def __init__(self, beam_size):
        super().__init__()
        self.beam_size=beam_size

    def forward(self):
        pass
        #Softmax is necessary...