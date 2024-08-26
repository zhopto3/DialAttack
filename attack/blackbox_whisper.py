import argparse
import os
import glob
import sys
import re

import whisper
import pandas as pd

sys.path.insert(1,"./training")
sys.path.insert(1,"./data_prep")

from error_rate import error_rate
from data_util import TextTransform


TEXT_PIPE = TextTransform()


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--experiment',type=str,required=True)
    args.add_argument('--model',type=str,required=True,
                      choices=['tiny','small','medium','large-v2','large-v3'])
    args.add_argument('--adversarial_target',type=str,required=True)
    return args.parse_args()


def get_wer(audio,model_text,gold_data):
    id=re.search(r'_(\d+).wav',audio)[0].lstrip('_').rstrip(".wav")
    gold_path = f'common_voice_ca_{id}.mp3'

    gold = gold_data.loc[gold_data['path']==gold_path,'sentence'].item()
    gold_preprocessed = TEXT_PIPE.preprocess(gold)

    return error_rate(gold_preprocessed,model_text,char=False),gold_preprocessed


def get_dial(audio,gold_data):
    id=re.search(r'_(\d+).wav',audio)[0].lstrip('_').rstrip(".wav")
    gold_path = f'common_voice_ca_{id}.mp3'

    dialect = gold_data.loc[gold_data['path']==gold_path,'grouped_accents'].item() 
    return dialect


def main(args):
    #load model
    model = whisper.load_model(args.model)

    adversarial_preprocessed = TEXT_PIPE.preprocess(args.adversarial_target)

    gold = pd.read_csv(f"./attack_samp_balanced.tsv",delimiter='\t',encoding='utf-8')

    audio_path = f"./experiments/{args.experiment}/cwattacks"
    audio_dir = glob.glob(os.path.join(audio_path,"*.wav"))
    print("Dialect\tGold Standard\tWhisper Output\tTrue Target WER\tAdversarial Target WER")
    for audio in audio_dir:
        out = model.transcribe(audio)
        text = out['text']
        wer,gold_transcript=get_wer(audio,text,gold)
        dial = get_dial(audio, gold)
        print(f'{dial}\t{gold_transcript}\t{text}\t{wer}\t{error_rate(adversarial_preprocessed,text,char=False)}')


if __name__ == "__main__":
    args = get_args()
    main(args)
