"""A script to initiate XLSR fine-tuning for ASR."""
import argparse
import sys
import os

from torchaudio import pipelines
from torch.utils.data import DataLoader
from torch.cuda import empty_cache


#Add customCV modules to path
sys.path.insert(1,"./data_prep")

from data_util import dynamic_padding
from custom_cv import CustomCV
from xlsr_asr import XLSR_ASR
from trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser("Fine tune XLS-R for Catalan ASR")

    parser.add_argument("--experiment_name", type = str, required = True, 
                        help = "Name for logging files and model output")
    
    parser.add_argument("--task", choices=["asr","dialect_classification"], default = 'asr',
                        help="automatic speech recognition (asr) or dialect classifcation (dialect_classification)")
    
    parser.add_argument("--model", choices = ["XLSR53","XLSR_300M","XLSR_1B","XLSR_2B"], required = True, 
                        help="Parameter num for pretrained XLSR model")
    
    parser.add_argument("--freeze_whole_model", action="store_true",
                        help="If passed, all XLSR encoder parameters are frozen (not the ASR head);if not passed, only the param in XLSR's CNN feature extractor are frozen.")
    
    parser.add_argument("--prop_central", choices=['20','50','80','100'], required=True, 
                        help="Percent of Central Catalan in training and dev data")
    
    parser.add_argument("--vocab_path",default="./data_prep/vocab.json",
                        help="Path to file containing the tokens in vocab and their ids")
    
    # parser.add_argument("--data_path",default="../cv-cat-18/ca/",
    #                     help="Path to directory containing data TSVs and clip directory")

    parser.add_argument("--clip_path",default="~/data/zhopto/cv-corpus-18.0-delta-2024-06-14/ca/clips",
                        help="Path to directory containing audio clips")

    parser.add_argument("--sample_path",default="./samples/samp_01",
                        help="Path to directory containing data TSVs about train and dev; should specify which sample")
    
    parser.add_argument("--batch_size", default = 16, type = int,
                        help="Train batch size")
    
    parser.add_argument("--grad_accum",action="store_true",
                        help="If true, will accumulate gradients and only update the weigths every 16 batches. Use to increase effective batch size when GPU storage limited.")
    
    parser.add_argument("--patience_delta", default = -0.05, type=float,
                        help = "Some minimal change to indicate when a model should stop training")
    
    # parser.add_argument("--min_eta", default = 0.00001, type=float,
    #                     help = "When using the Cosine annealing LR scheduler w/ warm restarts, the minimum learning rate")
    
    parser.add_argument("--eta", default = 0.0009, type=float,
                        help = "When using the Cosine annealing LR scheduler w/ warm restarts, the maximum learning rate")
    
    # parser.add_argument("--num_epochs", default = 6, type=int,
    #                     help = "When using the OneCycle LR scheduler, the number epochs")
    # parser.add_argument("--initial_lr", default = 0.00001, type=float,
    #                     help = "Initial learning rate to pass to lr scheduler")
    
    # parser.add_argument("--lr_patience", default=2, type=int,
    #                     help="Number of stagnant epochs (measured by validation loss) before dropping lr")
    
    parser.add_argument("--es_patience", default=3, type=int,
                        help="Number of epochs without a new best validation loss before ending training")

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

    clip_path_full = os.path.expanduser(args.clip_path)
    
    bundle = get_bundle(args.model)
    train_set = CustomCV(args.prop_central, bundle.sample_rate, split="train", vocab=args.vocab_path, path=args.data_path)
    dev_set = CustomCV(args.prop_central, bundle.sample_rate, split="dev", vocab=args.vocab_path, path=args.data_path)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, collate_fn = dynamic_padding)
    dev_loader = DataLoader(dev_set, shuffle=True, batch_size=args.batch_size, collate_fn = dynamic_padding)

    vocab_size = len(train_set.tokenizer)

    empty_cache()
    network = XLSR_ASR(vocab_size, args.model, freeze_model=args.freeze_whole_model)

    trainer = Trainer(args.task, train_loader=train_loader, val_loader=dev_loader, model=network, name=args.experiment_name)

    #Initiate training
    #trainer.train(initial_lr=args.initial_lr, es_patience=args.es_patience,lr_patience=args.lr_patience, delta=args.patience_delta,grad_accum=args.grad_accum)
    # trainer.train(min_eta=args.min_eta,max_eta=args.max_eta, es_patience=args.es_patience, delta=args.patience_delta,grad_accum=args.grad_accum)
    # trainer.train(max_eta=args.max_eta, grad_accum=args.grad_accum, num_epochs=args.num_epochs,batch_size=args.batch_size)
    trainer.train(eta=args.eta, es_patience=args.es_patience, delta=args.patience_delta,grad_accum=args.grad_accum)
if __name__ == "__main__":
    main()