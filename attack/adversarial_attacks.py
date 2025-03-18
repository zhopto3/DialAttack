import sys
import os
import logging
from math import isclose, log10

import torch
import torchaudio
from numpy.random import uniform

# Add customCV modules to path
sys.path.insert(1, "./data_prep")
sys.path.insert(1, "./training")

from decoders import Greedy_Decoder
from error_rate import error_rate


class CWAttack:

    def __init__(
        self, data_loader, model, experiment, tokenizer, dial_codes, learning_rate=0.1
    ):
        self.dl = data_loader
        self.network = model
        self.eta = learning_rate
        self.dial_codes = dial_codes
        self.criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.decoder = Greedy_Decoder()
        self.tokenizer = tokenizer
        self.toDB = torchaudio.transforms.AmplitudeToDB(stype="magnitude")

        # Make a folder to save adversarial samples to
        os.makedirs(f"./experiments/{experiment}/cwattacks", exist_ok=True)
        self.name = experiment

    def train_attack(self, epsilon=0.1, alpha=0.7, reg_c=0.25, num_iter=2000, k=8):
        """args:
        initial_epsilon: the maximum value in our distortion, i.e., ||delta||_inf <initial_epsilon
        alpha: Attenuating factor to multiply eta by once there is a successful attack
        reg_c: Regularizing factor by which we multiply the perturbation in the loss calculation in order to balance the desire for a quiet but successful attack
        num_iter: maximum number of iterations to carry out
        k: Max number of times to attenuate the eta
        """
        self._prepare_logger(epsilon, alpha, reg_c, num_iter, k)
        self.network = self.network.to(self.device)
        self.network.eval()
        for x, t, dial, audio_l, txt_l, path in self.dl:
            self._initialize_metrics(epsilon)

            # Move data to device
            t = t.to(self.device)
            audio_l = audio_l.to(self.device)
            txt_l = txt_l.to(self.device)
            x = x.to(self.device)
            # x_shape: B(1)xTxChannel(1); reshape to remove channel info
            x = x.squeeze(-1)
            # Make random noise
            delta = torch.from_numpy(
                uniform(low=-1 * self.epsilon, high=self.epsilon, size=x.shape)
            ).float()
            delta = delta.to(self.device)
            if self.k_count == 0:
                # Only log the name of the data point for the first adversarial made
                self.logger.info(f"Data Point:\t{path[0]}")
                self.logger.info(f"Dialect:\t{self.dial_codes[dial[0]]}")
            self.logger.info(
                "Now beginning a new adversarial sample for this data point."
            )
            self.logger.info(f"Initial Epsilon:\t{self.epsilon}")
            self.logger.info(f"Initial SNR:\t{self._calculate_snr(delta,x)}")
            self.logger.info(f"k_count\tIteration\tLoss\tSNR\tCurrent Output\tWER")
            # make the noisy sample require a gradient
            delta = torch.autograd.Variable(delta, requires_grad=True)
            # set up our optimizer for this sample
            optimizer = torch.optim.Adam([delta], lr=self.eta)
            while self.k_count < k and self.iter_count < num_iter:
                # Clamp the updated delta down to epsilon after every optimization step
                x_prime = x + torch.clamp(
                    delta.clone(), min=-1 * self.epsilon, max=self.epsilon
                )
                # clamp down to the possible range for audio files
                x_prime = torch.clamp(x_prime.clone(), min=-1.0, max=1.0)
                # Get model output
                y, in_lengths = self.network(x_prime, audio_l)
                # Calculate loss and add relevant criteria:
                J = self.criterion(y, t, in_lengths, txt_l)
                # Use default order of two for the norm
                loss = J + reg_c * (
                    torch.linalg.vector_norm(
                        torch.clamp(
                            delta.clone(), min=-1 * self.epsilon, max=self.epsilon
                        ),
                        dim=1,
                    )
                    ** 2
                )
                loss.backward(retain_graph=True)

                # Evaluate attack efficiency; squeeze middle dimension (batch) away
                wer, decoded_model = self._eval_attack(y, t)

                if isclose(0.0, wer):
                    self.last_success_db = self._calculate_snr(
                        torch.clamp(
                            delta.clone(), min=-1 * self.epsilon, max=self.epsilon
                        ),
                        x,
                    )
                    self.last_success_epsilon = self.epsilon
                    self.last_success_k = self.k_count
                    self.logger.info(
                        f"{self.k_count}\t{self.iter_count}\t{loss.item()}\t{self._calculate_snr(torch.clamp(delta.clone(),min=-1*self.epsilon, max = self.epsilon),x)}\t{decoded_model}\t{wer}"
                    )

                    self.epsilon *= alpha
                    self.k_count += 1
                    self.iter_count += 1
                    if self.k_count < 8:
                        self.logger.info(
                            f"Successful attack; reducing attack radius to {self.epsilon}"
                        )

                else:
                    self.iter_count += 1
                    # Update the perturbation
                    optimizer.step()
                    optimizer.zero_grad()
                    if self.iter_count % 25 == 0:
                        self.logger.info(
                            f"{self.k_count}\t{self.iter_count}\t{loss.item()}\t{self._calculate_snr(torch.clamp(delta.clone(),min=-1*self.epsilon, max = self.epsilon),x)}\t{decoded_model}\t{wer}"
                        )

            # Log relevant statistics after finishing each data point
            if self.k_count > 0:
                self._finish_attack_log(dial, x_prime, wer, decoded_model, path[0])
            else:
                last_snr = self._calculate_snr(
                    torch.clamp(delta.clone(), min=-1 * self.epsilon, max=self.epsilon),
                    x,
                )
                self._finish_attack_log(
                    dial, x_prime, wer, decoded_model, path[0], last_snr=last_snr
                )

    def _prepare_logger(self, epsilon, alpha, reg_c, num_iter, k):
        logging.basicConfig(
            filename=f"./experiments/{self.name}/logs/cw_attack.log",
            filemode="w",
            format="%(asctime)s %(message)s",
        )
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Now Launching CW Attacks")
        self.logger.info(f"Initial Epsilon (Max value in perturbation):\t{epsilon}")
        self.logger.info(
            f"Alpha (Factor to multiply by epsilon after successful attack):\t{alpha}"
        )
        self.logger.info(
            f"Regularizing Constant to multiply by the perturbation's L2 norm in loss calculation:\t{reg_c}"
        )
        self.logger.info(
            f"Number of perturbation updates before moving to next data point:\t{num_iter}"
        )
        self.logger.info(
            f"Max number of times to attempt attenuating eta within the max number of iterations:\t{k}"
        )
        self.logger.info(f"Adam Optimizer Eta:\t{self.eta}")

        # Prepare header for printed tsv information
        print(
            f"path\tdialect\tsuccess\tfinal_k\tfinal_epsilon\tquietest_attack_snr\tnum_iter\tlast_wer\tlast_transcription"
        )

    def _initialize_metrics(self, epsilon):
        self.epsilon = epsilon

        self.k_count = 0
        self.iter_count = 0
        # Keep track of the dB_x(delta),k, and epsilon of the last successful attack
        self.last_success_db = 0
        self.last_success_epsilon = 0
        self.last_success_k = 0

    def _calculate_snr(self, delta, x):
        # Carloni wagner def to get magnitude of dif in decibels
        # Larger difference indicates a quieter adversarial perturbance
        return 20 * (
            log10(torch.linalg.vector_norm(x, ord=torch.inf, dim=1).item())
            - log10(torch.linalg.vector_norm(delta, ord=torch.inf, dim=1).item())
        )

    def _eval_attack(self, model_out, attack_target):
        # Decode the output:
        decoded_y = self.decoder(model_out)

        decoded_model = self.tokenizer.decode(
            decoded_y[0].detach().cpu().type(torch.int64).tolist()
        )
        decoded_gold = self.tokenizer.decode(
            attack_target[0].detach().cpu().type(torch.int64).tolist()
        )
        wer = error_rate(decoded_gold, decoded_model, char=False)

        return wer, decoded_model

    def _finish_attack_log(
        self, dialect, x_prime, last_wer, decoded_model_out, path, last_snr=None
    ):
        """Called after every data point in the data set to log final statistics about the attack success"""
        if self.k_count > 0:
            self.logger.info("Successful:\tTrue")
            # Final K: k_count
            self.logger.info(f"Final K:\t{self.last_success_k+1}")
            # Final epsilon
            self.logger.info(f"Final Epsilon:\t{self.last_success_epsilon}")
            # Last Successful SNR: db_x(delta)
            self.logger.info(
                f"SNR of quietest successful attack: {self.last_success_db}"
            )
            # Final iter count (did it stop bc we reached max iter or max k? )
            self.logger.info(f"Number of iterations:\t{self.iter_count}")
            # Only save audio file if there was at least one succesful attack:
            self._save_audio(x_prime, path)
            # print relevant stats for tsv
            print(
                f"{path}\t{self.dial_codes[dialect[0]]}\t1\t{self.last_success_k+1}\t{self.last_success_epsilon}\t{self.last_success_db}\t{self.iter_count}\tNONE\tNONE"
            )
        else:
            # Log: "Successful: 0"
            self.logger.info("Successful:\tFalse")
            # Last Decoded attack output: since it wasn't a success, is it closer to the true target than the attack target?
            self.logger.info(f"Last Decoded Output:\t{decoded_model_out}")
            # Attack wer: (last wer) (might be useful to see how low the wer was among the attacks that never succeeded)
            self.logger.info(f"Last Attack Target WER:\t{last_wer}")
            # Final SNR
            self.logger.info(f"Last SNR:\t{last_snr}")
            # print relevant stats for tsv
            print(
                f"{path}\t{self.dial_codes[dialect[0]]}\t0\tNONE\tNONE\tNONE\tNONE\t{last_wer}\t{decoded_model_out}"
            )

    def _save_audio(self, x_prime, path):
        # Example in_path: common_voice_ca_20252730.mp3
        out_path = "adversarial_" + path.split("_")[-1]
        # Replace with .wav
        out_path = out_path.split(".")[0] + ".wav"
        torchaudio.save(
            uri=f"./experiments/{self.name}/cwattacks/{out_path}",
            src=x_prime.detach().cpu(),
            sample_rate=16000,
        )
