import torch
import numpy as np

from decoders import Greedy_Decoder, BeamSearch_Decoder

class Inference():

    def __init__(self, task, model,ckpt_path, test_loader, experiment_name):
        if task in ["asr","dial_class","adversarial"]:
            self.task = task
        else:
            raise Exception("Task not implemented")
        
        #Load model
        self.model = model

        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

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
            self.dial_codes = eval_set.code_2_dial
            self._eval_asr(decoder)
        elif self.task=="adversarial":
            pass
        else:
            raise Exception("Task not implemented")
        
    def _eval_asr(self, decoder):
        self.network = self.model.to(self.device)
        
        #Print header
        print(f"Dialect\tTarget\tASR Output\tWord Error Rate\tCharacter Error Rate")
        with torch.no_grad():
            for x,t,dialect,audio_l,txt_l in self.eval_loader:

                x = x.to(self.device)
                t = t.to(self.device)
                audio_l = audio_l.to(self.device)

                y,_ = self.network(x, audio_l)

                out_seqs = decoder(y)
                
                for gold_seq, model_seq, dial in zip(t,out_seqs, dialect):

                    #Get output and target in human readable formats (decode)
                    decoded_gold = self.tokenizer.decode(gold_seq.detach().cpu().type(torch.int64).tolist())
                    decoded_model = self.tokenizer.decode(model_seq.detach().cpu().type(torch.int64).tolist())
                    
                    wer = self._error_rate(decoded_gold, decoded_model, char=False)
                    cer = self._error_rate(decoded_gold, decoded_model, char=True)

                    print(f"{self.dial_codes[dial]}\t{decoded_gold}\t{decoded_model}\t{wer}\t{cer}")

    def _error_rate(self, gold, model, char):
        if char:
            gold_toks = list(gold)
            hyp_toks = list(model)

        else:
            gold_toks = gold.split()
            hyp_toks = model.split()
        
        #set up matrix: 
        mat = np.zeros((len(gold_toks)+1,len(hyp_toks)+1), dtype=int)
        #The first row/column represent the distance between each string and an empty string; fill them to reflect that
        mat[:,0] = np.array(range(len(gold_toks)+1))
        mat[0] = np.array(range(len(hyp_toks)+1))

        #now fill table, starting from second row/column
        for i in range(1,len(gold_toks)+1):
            for j in range(1,len(hyp_toks)+1):
                #Check if the word at the prev indices are equal to determine the local cost at a given cell
                cost = 0 if gold_toks[i-1] == hyp_toks[j-1] else 1
                #Check the prior three cells (horizontally, diagonally, and vertically) to select the minimum global cost for the cell
                mat[i][j]= cost + min(mat[i-1][j-1],mat[i][j-1],mat[i-1][j])
        min_cost = mat[len(gold_toks),len(hyp_toks)]

        #Return the cost normalized by the number of tokens in the reference
        return min_cost/len(gold_toks)


