from typing import Optional, List
import unicodedata
import os
import json
import re

import pandas as pd
#import torch

class Vocab:

    def __init__(self, path_to_voc:str, path_to_train_val: str = "../../../../../Desktop/cv-cat-18/ca/train_dev_full.tsv"):
        if os.path.isfile(path_to_voc):
            with open(path_to_voc,'r',encoding='utf-8') as voc:
                self.char_2_id = json.load(voc)
        else:
            self.char_2_id = self._make_vocab(path_to_train_val, path_to_voc)
        
        self.id_2_char = {id:char for char,id in list(self.char_2_id.items())}

    def _make_vocab(self, data:str, out_path:str):
        """Make a json containing the correspondance between tokens and their numeric id"""
        preprocesser = TextTransform()
        train_dev_df = pd.read_csv(data,delimiter="\t")
        
        alphabet = []

        preprocessed = train_dev_df.apply(lambda row: preprocesser.preprocess(row.sentence), axis=1)

        for sent in list(preprocessed):
            alphabet.extend(list(sent))
            
        #add unk, pad, and blank (same as pad?)
        print(set(alphabet))

    def encode_char(self, text):
        #If the character isn't in the alphabet, replace w unknown
        pass

    def decode_char(self, ids):
        pass


class TextTransform:

    def __init__(self,specials:Optional[List[str]]=None):
        if specials:
            self.specials=specials
        else:
            self.specials = ["\.","\?","\,","\!","\(","\)",
                             "\t","\n","…","\+","―",'‐','–','-',
                             "\[","\]","\*","/","¿",'�',"\|",
                             ";","_","–","—",">","~","¡",'“','ª','»',
                             '"','‘','’','«','ℂ','•','²',":","ı",'´','`',
                              'ং', 'ঃ', 'ः',"ὑ"]
    
    def preprocess(self,text: str):
        #Original df has mixed types, convert to str
        text = str(text)
        text = text.lower()
        text = unicodedata.normalize("NFC",text)

        #remove special characters
        for char in self.specials:
            text = re.sub(char,'',text)
        text = re.sub("@"," arroba ",text)
        text = re.sub('°',' graus ',text)
        text = re.sub('β',' beta ', text)

        text = re.sub("ň","n", text)
        text = re.sub("ë","e", text)
        text = re.sub("ć","c", text)
        text = re.sub("ž","z", text)
        text = re.sub('č','c', text)
        text = re.sub('ė','e', text)
        text = re.sub('ö','o', text)
        text = re.sub('ş','s', text)
        text = re.sub('ã','a', text)
        text = re.sub('ô','o', text)
        text = re.sub('â','a', text)
        text = re.sub('š','s', text)


        return text
    
    def dynamic_padding(self,batch):
        """Data collator for the data loader that will allow for dynamic padding (i.e., to the longest sent in a batch) rather than absolute"""
        pass