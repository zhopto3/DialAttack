from typing import Optional, List
import unicodedata
import os
import json
import re
import csv

import pandas as pd
import torch

class Tokenizer:

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
        train_dev_df = pd.read_csv(data,delimiter="\t", escapechar="\\",quoting = csv.QUOTE_NONE)
        
        alphabet = []

        preprocessed = train_dev_df.apply(lambda row: preprocesser.preprocess(row.sentence), axis=1)

        for sent in list(preprocessed):
            alphabet.extend(list(sent))
            
        alphabet = list(set(alphabet))
        #Add "|" for the blank 
        alphabet = {"|":0} | {char:i+1 for i,char in enumerate(alphabet)}
        #Add "unk"
        alphabet["[UNK]"]=len(alphabet)

        #Save for future use
        with open(out_path, "w", encoding="utf-8") as output:
            json.dump(alphabet,output,indent=1,ensure_ascii=False)
        return alphabet

    def encode(self, text):
        #If the character isn't in the alphabet, replace w unknown
        return [self.char_2_id.get(char, self.char_2_id["[UNK]"]) for char in list(text)]

    def decode(self, ids):
        return "".join([self.id_2_char.get(num) for num in ids])


class TextTransform:

    def __init__(self,specials:Optional[List[str]]=None):
        if specials:
            self.specials=specials
        else:
            self.specials = ["\.","\?","\,","\!","\(","\)",
                             "…","\+","―",'‐','–','-','\\\\',
                             "\[","\]","\*","/","¿",'�',"\|",
                             ";","_","–","—",">","~","¡",'“','ª','»',
                             '"','‘','’','«','•','²',":","ı",'´','`',
                              'ং', 'ঃ', 'ः',"ὑ","”",'̟','ð',"́"]
    
    def preprocess(self,text: str):
        #Original df has mixed types, convert to str
        text = str(text)
        text = unicodedata.normalize("NFKC",text)
        text = text.lower()

        # #remove special characters
        for char in self.specials:
            text = re.sub(char,' ',text)
        text = re.sub(r"\d",' ', text)
        text = re.sub("@"," arroba ",text)
        text = re.sub('°',' graus ',text)
        text = re.sub('β',' beta ', text)

        text = re.sub(r"\s",' ', text)
        text = re.sub(r"[ň,ℕ]","n", text)
        text = re.sub(r"[ë,ė]","e", text)
        text = re.sub(r"[ć,č,ℂ,с]","c", text)
        text = re.sub(r"[ž,ℤ]","z", text)
        text = re.sub(r'[š,ś,ş]','s', text)
        text = re.sub(r"[ì,î]",'i', text)
        #Replace with catalan digraph corresponding to same sound
        text = re.sub("ñ","ny", text)
        text = re.sub("ū","u", text)
        text = re.sub("ţ","t", text)
        text = re.sub("ř","r", text)
        text = re.sub('ł','l', text)
        text = re.sub('ù','u', text)
        text = re.sub(r"[ō,ŏ,ö,ô,ő,ø]","o", text)
        text = re.sub(r"[å,ā,â,ã,ä,á]","a", text)

        return text.strip()
    

def dynamic_padding(batch):
    X = [wf.reshape(-1,1) for wf,_,_2 in batch]
    T = [torch.Tensor(txt) for _,txt,_2 in batch]
    dial = [dialect for _,_2,dialect in batch]
    X = torch.nn.utils.rnn.pad_sequence(X,batch_first=True).squeeze()
    T = torch.nn.utils.rnn.pad_sequence(T,batch_first=True)
    return X, T, dial