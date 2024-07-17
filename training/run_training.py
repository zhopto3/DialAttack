"""A script to initiate XLSR fine-tuning for ASR."""
import argparse
import sys

from torchaudio import pipelines
from torch.utils.data import DataLoader


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
    
    parser.add_argument("--freeze_feature_extractor", action="store_true",
                        help="If passed, parameters in CNN feature extractor are frozen.")
    
    parser.add_argument("--prop_central", choices=['20','50','80','100'], required=True, 
                        help="Percent of Central Catalan in training and dev data")
    
    parser.add_argument("--vocab_path",default="./data_prep/vocab.json",
                        help="Path to file containing the tokens in vocab and their ids")
    
    parser.add_argument("--data_path",default="../cv-cat-18/ca/",
                        help="Path to directory containing data TSVs and clip directory")
    
    parser.add_argument("--batch_size", default = 256, type = int,
                        help="Train batch size")
    
    parser.add_argument("--intial_lr", default = 0.00001, type=float,
                        help = "Initial learning rate to pass to lr scheduler")
    
    parser.add_argument("--lr_patience", default=2, type=int,
                        help="Number of stagnant epochs (measured by validation loss) before dropping lr")
    
    parser.add_argument("--es_patience", default=5, type=int,
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

    bundle = get_bundle(args.model)

    train_set = CustomCV(args.prop_central, bundle.sample_rate, split="train", vocab=args.vocab_path, path=args.data_path)
    dev_set = CustomCV(args.prop_central, bundle.sample_rate, split="dev", vocab=args.vocab_path, path=args.data_path)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, collate_fn = dynamic_padding)
    dev_loader = DataLoader(dev_set, shuffle=True, batch_size=args.batch_size, collate_fn = dynamic_padding)

    vocab_size = len(train_set.tokenizer)

    network = XLSR_ASR(vocab_size, args.model, freeze_CNN=args.freeze_feature_extractor)

    trainer = Trainer(args.task, train_loader=train_loader, val_loader=dev_loader, model=network, name=args.experiment_name)

    #Initiate training
    #trainer.train(initial_lr=args.initial_lr, es_patience=args.es_patience,lr_patience=args.lr_patience)

if __name__ == "__main__":
    main()