import torchaudio
from data_util import Tokenizer, TextTransform
import json
from typing import Dict, Tuple
from torch import Tensor



class CustomCV(torchaudio.datasets.COMMONVOICE):
    
    def __init__(self, prop_central: int, model_sr: int ,split: str, clip_path: str, sample_path: str, vocab: str="vocab.json"):
        super(CustomCV,self).__init__(
            root=path,
            tsv='eval_balanced.tsv' if split == 'test' else f"{split}_{prop_central}.tsv"
        )
        #path: str="../cv-cat-18/ca/"
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

    def __getitem__(self, n: int) -> Tuple[Tensor, list, int]:
        waveform, samp_rate, metadata = super().__getitem__(n)
        #Create transform depending on current samples sample rate
        if samp_rate != self.model_samp_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=samp_rate,new_freq=self.model_samp_rate)
        #Preproces text
        text = self.text_pipe.preprocess(metadata['sentence'])
        #Encode text (Might need to turn these into one-hot encoded vectors at some point)
        encoded_text = self.tokenizer.encode(text)
        return waveform,encoded_text,self.dial.get(metadata["grouped_accents"])