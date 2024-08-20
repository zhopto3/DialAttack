import sys
import os
from math import isclose

import torch
import torchaudio
from numpy.random import random_sample 

#Add customCV modules to path
sys.path.insert(1,"./data_prep")
sys.path.insert(1,"./training")

from training.decoders import Greedy_Decoder
from training.error_rate import error_rate


class CWAttack:

    def __init__(self,data_loader, model, experiment, tokenizer, learning_rate=0.1):
        self.dl = data_loader
        self.network = model
        self.eta = learning_rate
        self.criterion =  torch.nn.CTCLoss(blank=0,zero_infinity=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.decoder = Greedy_Decoder()
        self.tokenizer = tokenizer

        #Make a folder to save adversarial samples to
        os.makedirs(f"./experiments/{experiment}/cwattacks")

    def train_attack(self,epsilon=0.1, alpha=0.7, reg_c = 0.25, num_iter=2000,k=8):
        """args:
            initial_epsilon: the maximum value in our distortion, i.e., ||delta||_inf <initial_epsilon
            alpha: Attenuating factor to multiply eta by once there is a successful attack
            reg_c: Regularizing factor by which we multiply the perturbation in the loss calculation in order to balance the desire for a quiet but successful attack
            num_iter: maximum number of iterations to carry out 
            k: Max number of times to attenuate the eta
            """
        self.network = self.network.to(self.device)
        self.network.eval()
        for x,t, dial, audio_l,text_l in self.dl:
            self._initialize_metrics()

            #Move data to device
            t = t.to(self.device)
            audio_l = audio_l.to(self.device)
            txt_l = txt_l.to(self.device)

            #Make random noise
            delta = torch.from_numpy((epsilon+epsilon) * random_sample(size=x.shape)-epsilon)
            #Should probably save the 
            ##@TODO: LOG SOMETHING
            #make the delta require a gradient
            delta = torch.autograd.Variable(delta,requires_grad=True)
            #set up our optimizer for this sample
            optimizer = torch.optim.adam([delta],lr=self.eta)

            while self.k_count < k and self.iter_count < num_iter:
                x_prime = x+delta
                #clamp down to the possible range for audio files
                x_prime = torch.clamp(x_prime,min=-1.,max=1.)
                x_prime = x_prime.to(self.device)
                #Get model output
                y, in_lengths = self.network(x_prime,audio_l)
                #Calculate loss and add relevant criteria:
                J = self.criterion(y,t,in_lengths,txt_l)
                #Use default order of two for the norm
                J = J.item() + reg_c * (torch.linalg.vector_norm(delta).item()**2)
                J.backward()
                #Update the perturbation
                optimizer.step()
                optimizer.zero_grad()
                
                #Evaluate attack efficiency
                wer = self._eval_attack(y,t)

                if isclose(0.0,wer):
                    self.k_count +=1
                    self.iter_count +=1
                    epsilon *= alpha
                    ##@TODO: LOG SOME INFORMATION##
                    #break the loop, restart 
                    break
                else: 
                    self.iter_count +=1
            ###@TODO:LOG SOMETHING###
            #What to log at the end of each data point?
                #final perturbations amplitude
                # last wer for the target...
                # The current k_count, iter_count, epsilon


    def _eval_attak(self,model_out, attack_target):
        #Decode the output:
        decoded_y = self.decoder(model_out)

        decoded_model = self.self.tokenizer.decode(decoded_y.detach().cpu().type(torch.int64).tolist())
        decoded_gold = self.tokenizer.decode(attack_target.detach().cpu().type(torch.int64).tolist())

        wer = error_rate(decoded_gold, decoded_model, char=False)

        return wer

    
    def _initialize_metrics(self):
        self.k_count = 0  
        self.iter_count = 0