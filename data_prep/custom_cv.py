import torchaudio
import json


class CustomCV(torchaudio.datasets.COMMONVOICE):
    
    def __init__(self, prop_central: float, path: str, tsv_name:str, split: str):
        super(CustomCV,self).__init__(
            root=path,
            tsv=tsv_name
        )

        with open("./macrodial.json",'r',encoding="utf-8") as input:
            self.macro_dial = json.loads(input)

        self.alphabet = set()
        self.char_2_id = {}
        self.id_2_char = {}
        
        #@TODO : get the data w/ relevant proportion of each dialect

        #@TODO : Preprocess text (character tokenization, lowercase, remove punctuation) (possibly in separate class)

        #@TODO : preprocess speech input

        #@TODO : overwrite the __get_item__ method