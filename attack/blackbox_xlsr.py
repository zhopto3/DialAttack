"""Adapted Hugging Face code from here: https://huggingface.co/PereLluis13/Wav2Vec2-Large-XLSR-53-catalan"""
import argparse
import torch
import torchaudio
import sys

sys.path.insert(1,"./data_prep")
sys.path.insert(1,"./training")

from data_util import attack_collator
from attack_dataset import AttackCV
from inference import Inference
from xlsr_asr import XLSR_ASR



#@TODO: Don't use his. U have everything in place, just run eval with yours now right? Like the actual eval script should work with this data loader!!

def get_args():
    parser = argparse.ArgumentParser("Script to launch CW adversarial attacks on fine-tuned ASR models")

    parser.add_argument("--attack_experiment", type = str, required = True, 
                        help = "Name for the experiment used to train these attacks")
    parser.add_argument("--experiment_subject", type = str, required = True, 
                        help = "Name for the experiment being attacked")
    
    parser.add_argument("--model", choices = ["XLSR53","XLSR_300M","XLSR_1B","XLSR_2B"], required = True, 
                        help="Parameter num for pretrained XLSR model")
    
    parser.add_argument("--vocab_path",default="./data_prep/vocab.json",
                        help="Path to file containing the tokens in vocab and their ids")
    
    parser.add_argument("--data_path",default="../cv-cat-18/ca/",
                        help="Path to directory containing data TSVs and clip directory")
    
    parser.add_argument("--checkpoint_name",default="best_checkpoint.pt",
                        help="Path to file containing model checkpoint")
    
    
    return parser.parse_args()


def get_bundle(model_name: str):
    if model_name =="XLSR53":
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    elif model_name=="XLSR_300M":
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
    elif model_name=="XLSR_1B":
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
    elif model_name=="XLSR_2B":
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
    else:
        raise Exception('Model not found')
    
    return bundle


def main():
    target = "Porta'm a un lloc web malvat."
    attacks = AttackCV(16000, attack_target=target, vocab=args.vocab_path, path=args.data_path,blackbox=True,experiment=args.attack_experiment)

    attack_loader = torch.utils.data.DataLoader(attacks, shuffle=False, batch_size=1, collate_fn = attack_collator)

    bundle = get_bundle(args.model)
    vocab_size = len(attacks.tokenizer)
    network = XLSR_ASR(vocab_size, args.model)

    evaluator = Inference(task="adversarial", model=network, ckpt_path=f"./{args.experiment_subject}/checkpoints/{args.checkpoint_name}",
                          test_loader=attack_loader, experiment_name=args.experiment_subject)
    
    
    #Initiate Inference/Evaluation
    evaluator.evaluate(beam_size=args.beam_size, eval_set=attacks)



if __name__ =="__main__":
    args = get_args()
    main(args)
