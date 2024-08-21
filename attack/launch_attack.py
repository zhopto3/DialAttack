"""A script with CLI to begin training and evaluating CW adversarial attacks"""
import argparse
import sys

from torchaudio import pipelines
from torch.utils.data import DataLoader
from torch.cuda import empty_cache
from torch import load


#Add customCV modules to path
sys.path.insert(1,"./data_prep")

from attack_dataset import AttackCV
from data_util import attack_collator
from xlsr_asr import XLSR_ASR
from adversarial_attacks import CWAttack


def get_args():
    parser = argparse.ArgumentParser("Script to launch CW adversarial attacks on fine-tuned ASR models")

    parser.add_argument("--experiment_name", type = str, required = True, 
                        help = "Name for logging files and attacl output; should be the same as the train experiment name")
    
    parser.add_argument("--model", choices = ["XLSR53","XLSR_300M","XLSR_1B","XLSR_2B"], required = True, 
                        help="Parameter num for pretrained XLSR model")
    
    parser.add_argument("--vocab_path",default="./data_prep/vocab.json",
                        help="Path to file containing the tokens in vocab and their ids")
    
    parser.add_argument("--data_path",default="../cv-cat-18/ca/",
                        help="Path to directory containing data TSVs and clip directory")
    
    parser.add_argument("--checkpoint_name",default="best_checkpoint.pt",
                        help="Path to file containing model checkpoint")

    parser.add_argument("--lr", type=float, default=0.1,
                        help = "learning rate for updating the adversarial perturbance")
    
    parser.add_argument("--regularizing_const", type=float, default=0.25,
                        help = "Regularizing factor by which we multiply the perturbation in the loss calculation in order to balance the desire for a quiet but successful attack")
    
    parser.add_argument("--k", type=int, default=8,
                        help = "Max number of times to attenuate the eta")
    
    parser.add_argument("--alpha", type=float, default=0.7,
                        help = "Attenuating factor to multiply eta by once there is a successful attack")
    
    parser.add_argument("--initial_epsilon", type=float, default=0.1,
                        help = "Maximum change in amplitude allowed in first adversarial perturbance")
    
    parser.add_argument("--number_iterations", type=int, default=2000,
                        help = "Number of updates to make to adversarial noise.")
    
    return parser.parse_args()


def get_bundle(model_name: str):
    if model_name =="XLSR53":
        bundle = pipelines.WAV2VEC2_XLSR53
    elif model_name=="XLSR_300M":
        bundle = pipelines.WAV2VEC2_XLSR_300M
    elif model_name=="XLSR_1B":
        bundle = pipelines.WAV2VEC2_XLSR_1B
    elif model_name=="XLSR_2B":
        bundle = pipelines.WAV2VEC2_XLSR_2B
    else:
        raise Exception('Model not found')
    
    return bundle


def main(args):
    bundle = get_bundle(args.model)

    dataset = AttackCV(bundle.sample_rate, attack_target="Porta'm a un lloc web malvat.", vocab=args.vocab_path, path=args.data_path)

    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn = attack_collator)

    vocab_size = len(dataset.tokenizer)

    empty_cache()
    #Doesn't matter if we freeze the cnn when initializing or not because we will freeze the whole network to do inference
    network = XLSR_ASR(vocab_size, args.model)
    #Load the relevant checkpoint
    checkpoint = load(args.experiment_name+"/checkpoints/"+args.checkpoint_name)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()

    attacker = CWAttack(data_loader=data_loader,model=network,experiment=args.experiment_name,
             tokenizer=dataset.tokenizer,dial_codes=dataset.code_2_dial,learning_rate=args.lr)

    #Launch attack
    attacker.train_attack(args.initial_epsilon,args.alpha,args.regularizing_const,
                          args.number_iterations,args.k)

if __name__=="__main__":
    args = get_args()
    main(args)