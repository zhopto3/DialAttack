import torchaudio
from text_processing import Tokenizer, TextTransform
import json


class CustomCV(torchaudio.datasets.COMMONVOICE):
    
    def __init__(self, prop_central: float, split: str, vocab: str="vocab.json", path: str="../../../../../Desktop/cv-cat-18/ca/"):
        super(CustomCV,self).__init__(
            root=path,
            tsv='eval_balanced.tsv' if split == 'test' else "train_dev_full.tsv"
        )
        self.tokenizer = Tokenizer(vocab)
        # with open("./macrodial.json",'r',encoding="utf-8") as input:
        #     self.macro_dial = json.loads(input)
        
        #@TODO : get the data w/ relevant proportion of each dialect

        #@TODO : Preprocess text (character tokenization, lowercase, remove punctuation) (in separate class)

        #@TODO : preprocess speech input (just resampling, xlsr takes raw waveform not mel)

        #@TODO : overwrite the __get_item__ method