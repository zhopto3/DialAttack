import sys
import os
import logging
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

    def __init__(self,data_loader, model, experiment, tokenizer,dial_codes, learning_rate=0.1):
        self.dl = data_loader
        self.network = model
        self.eta = learning_rate
        self.dial_codes = dial_codes
        self.criterion =  torch.nn.CTCLoss(blank=0,zero_infinity=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.decoder = Greedy_Decoder()
        self.tokenizer = tokenizer
        self.toDB = torchaudio.transforms.AmplitudeToDB(stype="magnitude")

        #Make a folder to save adversarial samples to
        os.makedirs(f"./experiments/{experiment}/cwattacks",exist_ok=True)
        self.name = experiment

    def train_attack(self,epsilon=0.1, alpha=0.7, reg_c = 0.25, num_iter=2000,k=8):
        """args:
            initial_epsilon: the maximum value in our distortion, i.e., ||delta||_inf <initial_epsilon
            alpha: Attenuating factor to multiply eta by once there is a successful attack
            reg_c: Regularizing factor by which we multiply the perturbation in the loss calculation in order to balance the desire for a quiet but successful attack
            num_iter: maximum number of iterations to carry out 
            k: Max number of times to attenuate the eta
            """
        self._prepare_logger(epsilon, alpha, reg_c, num_iter,k)
        self.network = self.network.to(self.device)
        self.network.eval()
        for x,t, dial, audio_l,text_l,path in self.dl:
            self._initialize_metrics()

            #Move data to device
            t = t.to(self.device)
            audio_l = audio_l.to(self.device)
            txt_l = txt_l.to(self.device)

            #Make random noise
            delta = torch.from_numpy((epsilon+epsilon) * random_sample(size=x.shape)-epsilon)
            #Should probably save the initial amplitude / SNR of the noise
            if self.k_count==0:
                #Only log the name of the data point for the first adversarial made
                self.logger.info(f"Data Point:\t{path}")
                self.logger.info(f"Dialect:\t{self.dial_codes[dial[0]]}")
            self.logger.info("Now beginning a new adversarial sample for this data point.") 
            self.logger.info(f"Initial Epsilon:\t{self.toDB(delta)-self.toDB(x)}")
            self.logger.info(f"Initial SNR:\t{self.toDB(delta)-self.toDB(x)}")
            self.logger.info(f"k_count\tIteration\tSNR\tWER")
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
                
                #Evaluate attack efficiency
                wer, decoded_model= self._eval_attack(y,t)

                if isclose(0.0,wer):
                    epsilon *= alpha
                    self.last_success_db = self.toDB(delta)-self.toDB(x)
                    self.last_success_epsilon = epsilon
                    self.last_success_k = self.k_count
                    self.logger.info(f"{self.k_count}\t{self.num_iter}\t{self.toDB(delta)-self.toDB(x)}\t{wer}")

                    self.k_count +=1
                    self.iter_count +=1
                    #break the loop, restart 
                    break
                else: 
                    self.iter_count +=1
                    #Update the perturbation
                    optimizer.step()
                    #As in the whisper attacks, clamp the updated delta down to epsilon after every optimization step
                    delta = torch.clamp(delta,min=-1*epsilon, max = epsilon)
                    optimizer.zero_grad()
                    #
                    if self.iter_count//25==0:
                        self.logger.info(f"{self.k_count}\t{self.num_iter}\t{self.toDB(delta)-self.toDB(x)}\t{wer}")

            #Log relevant statistics after finishing each data point
            if self.k_count>0:
                self._finish_attack_log(x_prime, epsilon, wer, decoded_model,path[0])
            else:
                last_snr=self.toDB(delta)-self.toDB(x)
                self._finish_attack_log(x_prime, epsilon, wer, decoded_model,path[0],last_snr=last_snr)

    def _prepare_logger(self,epsilon, alpha, reg_c, num_iter,k):
        logging.basicConfig(filename=f'./experiments/{self.name}/logs/cw_attack.log',filemode="w", format='%(asctime)s %(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.info('Now Launching CW Attacks')
        self.logger.info(f'Initial Epsilon (Max value in perturbation):\t{epsilon}')
        self.logger.info(f'Alpha (Factor to multiply by epsilon after successful attack):\t{alpha}')
        self.logger.info(f"Regularizing Constant to multiply by the perturbation's L2 norm in loss calculation:\t{reg_c}")
        self.logger.info(f'Number of perturbation updates before moving to next data point:\t{num_iter}')
        self.logger.info(f'Max number of times to attempt attenuating eta within the max number of iterations:\t{k}')

    def _initialize_metrics(self):
        self.k_count = 0  
        self.iter_count = 0
        #Keep track of the dB_x(delta),k, and epsilon of the last successful attack
        self.last_success_db = 0
        self.last_success_epsilon = 0
        self.last_success_k = 0

    def _eval_attak(self,model_out, attack_target):
        #Decode the output:
        decoded_y = self.decoder(model_out)

        decoded_model = self.self.tokenizer.decode(decoded_y.detach().cpu().type(torch.int64).tolist())
        decoded_gold = self.tokenizer.decode(attack_target.detach().cpu().type(torch.int64).tolist())

        wer = error_rate(decoded_gold, decoded_model, char=False)

        return wer,decoded_model

    def _finish_attack_log(self,x_prime, last_wer, decoded_model_out,path,last_snr=None):
        """Called after every data point in the data set to log final statistics about the attack success"""
        if self.k_count>0:
            self.logger.info('Successful:\tTrue')
            #Final K: k_count
            self.logger.info(f'Final K:\t{self.last_success_k}')
            #Final epsilon (can figure this out with the final k and the alpha but this is easier)
            self.logger.info(f"Final Epsilon:\t{self.last_success_epsilon}")
            #Last Successful SNR: db_x(delta)
            self.logger.info(f"SNR of quietest successful attack: {self.last_success_db.item()}")
            # Final iter count (did it stop bc we reached max iter or max k? )
            self.logger.info(f"Number of iterations:\t{self.iter_count}")
            #Only saving the audio file if there was at least one succesful attack:
            self._save_audio(x_prime,path)
        else:
            #Log: "Successful: 0"
            self.logger.info("Successful:\tFalse")
            # Last Decoded attack output: since it wasn't a success, is it closer to the true target than the attack target? 
            self.logger.info(f"Last Decoded Output:\t{decoded_model_out}")
            # Attack wer: (last wer) (might be useful to see how low the wer was among the attacks that never succeeded)
            self.logger.info(f"Last Attack Target WER:\t{last_wer}")
            # Last SNR: Not sure if it's useful, but might as well
            self.logger.ingo(f"Last SNR:\t{last_snr}") 
    
    def _save_audio(self, x_prime,path):
        #Example in_path: common_voice_ca_20252730.mp3
        out_path = "adversarial_"+path.split("_")[-1]
        #Replace with .wav
        out_path = out_path.split(".")[0]+".wav"
        torchaudio.save(uri=f"./experiments/{self.name}/cwattacks/{out_path}",src=x_prime,
                        sample_rate=16000)