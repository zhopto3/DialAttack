"""A custom version of the CV dataset class to make the data set for training adversarial attacks"""
import sys
import json
from typing import Dict, Tuple

from torch import Tensor
import torchaudio

#Add customCV modules to path
sys.path.insert(1,"./data_prep")

from data_util import Tokenizer, TextTransform



class CustomCV(torchaudio.datasets.COMMONVOICE):
    
    def __init__(self, model_sr:int , attack_target: str,
                 #CHANGE ATTACK TARGET
                 vocab: str="vocab.json", path: str="../cv-cat-18/ca/"):
        super(CustomCV,self).__init__(
            root=path,
            #access a balanced sample from the eval set
            tsv='attack_samp_balanced.tsv'
        )
        self.tokenizer = Tokenizer(vocab)
        self.text_pipe = TextTransform()
        self.dial = {"central":0,
                     "nord":1,
                     "nord-occidental":2,
                     "balear":3,
                     "valencià":4}
        self.code_2_dial = {0:"central",
                            1:"nord",
                            2:"nord-occidental",
                            3:"balear",
                            4:"valencià"}
        self.model_samp_rate = model_sr

        #Can just do these preprocessing operations one time since we'll only have the one attack target
        self.text = self.text_pipe.preprocess(attack_target)
                #Encode text (Might need to turn these into one-hot encoded vectors at some point)
        self.encoded_text = self.tokenizer.encode(self.text)

    def __getitem__(self, n: int) -> Tuple[Tensor, list, int]:
        waveform, samp_rate, metadata = super().__getitem__(n)
        #Create transform depending on current samples sample rate
        if samp_rate != self.model_samp_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=samp_rate,new_freq=self.model_samp_rate)

        return waveform,self.encoded_text,self.dial.get(metadata["grouped_accents"])