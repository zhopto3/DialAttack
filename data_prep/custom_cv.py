import torchaudio
from text_processing import Tokenizer, TextTransform
import json
from typing import Dict, Tuple
from torch import Tensor



class CustomCV(torchaudio.datasets.COMMONVOICE):
    
    def __init__(self, prop_central: int, split: str, vocab: str="vocab.json", path: str="../../../../../Desktop/cv-cat-18/ca/"):
        super(CustomCV,self).__init__(
            root=path,
            tsv='eval_balanced.tsv' if split == 'test' else f"{split}_{prop_central}.tsv"
        )
        self.tokenizer = Tokenizer(vocab)
        self.text_pipe = TextTransform()
        self.dial = {"central":0,
                     "nord":1,
                     "nord-occidental":2,
                     "balear":3,
                     "valencià":4}

    def __getitem__(self, n: int) -> Tuple[Tensor, list, int]:
        waveform, samp_rate, metadata = super().__getitem__(n)
        #Create transform depending on current samples sample rate
        resamp = torchaudio.transforms.Resample(orig_freq=samp_rate,new_freq=16000)
        #Preproces text
        text = self.text_pipe.preprocess(metadata['sentence'])
        #Encode text (Might need to turn these into one-hot encoded vectors at some point)
        encoded_text = self.tokenizer.encode(text)
        return resamp(waveform),encoded_text,self.dial.get(metadata["grouped_accents"])