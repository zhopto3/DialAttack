"""A script to initiate XLSR inference for ASR."""
import argparse
import sys

from torchaudio import pipelines
from torch.utils.data import DataLoader
from torch.cuda import empty_cache


#Add customCV modules to path
sys.path.insert(1,"./data_prep")

from data_util import dynamic_padding
from custom_cv import CustomCV
from xlsr_asr import XLSR_ASR
from inference import Inference


def get_args():
    parser = argparse.ArgumentParser("Fine tune XLS-R for Catalan ASR")

    parser.add_argument("--experiment_name", type = str, required = True, 
                        help = "Name for logging files and model output; should be the same as the train experiment name")
    
    parser.add_argument("--task", choices=["asr","dial_class","adversarial"], default = 'asr',
                        help="automatic speech recognition (asr), dialect classifcation (dial_class), or evaluate effectiveness of an adversarial attack (adversarial)")
    
    parser.add_argument("--model", choices = ["XLSR53","XLSR_300M","XLSR_1B","XLSR_2B"], required = True, 
                        help="Parameter num for pretrained XLSR model")
    
    parser.add_argument("--vocab_path",default="./data_prep/vocab.json",
                        help="Path to file containing the tokens in vocab and their ids")
    
    parser.add_argument("--data_path",default="../cv-cat-18/ca/",
                        help="Path to directory containing data TSVs and clip directory")
    
    parser.add_argument("--checkpoint_name",default="best_checkpoint.pt",
                        help="Path to file containing model checkpoint")
    
    parser.add_argument("--batch_size", default = 64, type = int,
                        help="Inference batch size")

    parser.add_argument("--beam_size", default = 1, type = int,
                        help="Beam size; default 1 indicates greedy decoding")

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


def main():
    args = get_args()

    bundle = get_bundle(args.model)

    test_set = CustomCV(100, bundle.sample_rate, split="test", vocab=args.vocab_path, path=args.data_path)

    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size, collate_fn = dynamic_padding)

    vocab_size = len(test_set.tokenizer)

    #Doesn't matter if we freeze the cnn when initializing or not because we will freeze the whole network to do inference
    network = XLSR_ASR(vocab_size, args.model)

    evaluator = Inference(task=args.task, model=network, ckpt_path=f"./{args.experiment_name}/checkpoints/{args.checkpoint_name}",
                          test_loader=test_loader, experiment_name=args.experiment_name)

    #Initiate Inference/Evaluation
    evaluator.evaluate(beam_size=args.beam_size, eval_set=test_set)

if __name__ == "__main__":
    main()