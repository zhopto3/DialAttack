import sys

import torch

from decoders import Greedy_Decoder, BeamSearch_Decoder

class Inference():

    def __init__(self, task, model,ckpt_path, test_loader, experiment_name):
        if task in ["asr","dial_class","adversarial"]:
            self.task = task
        else:
            raise Exception("Task not implemented")
        
        #Load model
        self.model = model
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

        self.eval_loader = test_loader

        self.name = experiment_name

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def evaluate(self, beam_size: int, eval_set):
        if self.task=="asr":
            #@TODO: make decoder object from another module
            if beam_size==1:
                #make a greedy decoder
                decoder = Greedy_Decoder()
            else:
                #make a beam search decoder
                decoder = BeamSearch_Decoder(beam_size=beam_size)
            self.tokenizer = eval_set.tokenizer
            self._eval_asr(decoder)
        elif self.task=="adversarial":
            pass
        else:
            raise Exception("Task not implemented")
        
    def _eval_asr(self, decoder):
        self.network = self.model.to(self.device)

        #Print header
        print(f"Target\tASR Output\tWord Error Rate\tCharacter Error Rate")
        with torch.no_grad():
            for x,t,_,audio_l,txt_l in self.eval_loader:
                x = x.to(self.device)
                t = t.to(self.device)
                audio_l = audio_l.to(self.device)
                txt_l = txt_l.to(self.device)

                y,_ = self.network(x, audio_l)

                out_seqs = decoder(y)

                for gold_seq, model_seq in zip(t,out_seqs):
                    #Get output and target in human readable formats (decode)
                    decoded_gold = self.tokenizer.decode(gold_seq)
                    decoded_model = self.tokenizer.decode(model_seq)

                    wer = self._wer(decoded_gold, decoded_model)
                    cer = self._cer(decoded_gold, decoded_model)

                    print(f"{decoded_gold}\t{decoded_model}\t{wer}\t{cer}")
    
    def _wer(self, gold, model):
        pass

    def _cer(self, gold, model):
        pass

