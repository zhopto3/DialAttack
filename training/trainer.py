from collections import deque
import os
import logging

import torch


class Trainer:

    def __init__(self,task,train_loader,val_loader, model, name):
        if task=="asr":
            self.task = 'asr'
            self.criterion = torch.nn.CTCLoss(blank=0,zero_infinity=True)
        elif task=="dialect_classification":
            self.task = "dial_class"
            self.criterion = torch.nn.CrossEntropyLoss()
        else: 
            raise Exception("Task not implemented")
        self.network = model
        self.train_dl = train_loader
        self.val_dl = val_loader

        self.name = name

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def train(self, initial_lr, es_patience, lr_patience, logger_level = logging.INFO):
        os.makedirs(f'./{self.name}',exist_ok=True)
        os.makedirs(f'./{self.name}/logs',exist_ok=True)
        logging.basicConfig(filename=f'./{self.name}/logs/{self.name}.log',filemode="w", format='%(asctime)s %(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logger_level)

        if self.task=="asr":
            self.optimizer = torch.optim.Adam(self.network.parameters(),lr = initial_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=lr_patience)
            self.logger.info(f"ASR training:")
            self._train_asr(es_patience)
        elif self.task=="dial_class":
            #@TODO implement optimizer and scheduler for dialect classification if necessary
            self._train_dial_class(es_patience)
        else:
            raise Exception("Task not implemented")
        
    def _train_asr(self,patience):
        self.network = self.network.to(self.device)

        self._initialize_metrics(patience)

        while self._terminate(patience):
            self.logger.info(f"Epoch {len(self.train_loss_epoch)+1}:")
            self.logger.info(f"Learning Rate: {self.scheduler.get_last_lr()}")
            self.logger.info("Train Step\tBatch Loss")
            self.network.train()
            for x,t,_,audio_l,txt_l in self.train_dl:
                x = x.to(self.device)
                t = t.to(self.device)
                audio_l = audio_l.to(self.device)
                txt_l = txt_l.to(self.device)

                self.optimizer.zero_grad()

                #log_probs of each class at each at each frame; num frames w/out padding in each batch item
                y, in_lengths = self.network(x, audio_l)

                #Calculate the loss
                J = self.criterion(y,t,in_lengths,txt_l)
                #Calculate gradient of loss and make update 
                J.backward()
                self.optimizer.step()

                #Update running batch loss & dict
                self.running_train_loss+=J.item()
                self.train_loss_batch[len(self.train_loss_batch)]=J.item()
                self.logger.info(f"{len(self.train_loss_batch)}\t{J.item()}")
            self.logger.info(f"End of Epoch {len(self.train_loss_epoch)+1}")
            #Update the train epoch loss; reset running train loss to 0
            self.train_loss_epoch[len(self.train_loss_batch)]=self.running_train_loss/len(self.train_dl)
            self.logger.info(f"Average Train loss over epoch:\t{self.train_loss_epoch[len(self.train_loss_batch)+1]}")
            self.running_train_loss=0.
            #Run validation and update relevant metrics
            J_val = self._vaildation_step_asr()
            self.logger.info(f"Average Validation loss over epoch:\t{J_val}")
            self.logger.info(f"Best Validation Epoch: {self.best_val_epoch}")
            self.logger.info(f"Best Validation Loss: {self.best_val_loss}")
            self.logger.info(f"Delta Validation loss: {self.delta_queue[-1]}")
            #Adjust learning rate
            self.scheduler.step(J_val)

    def _validation_step_asr(self):
        self.network.eval()
        with torch.no_grad():
            for x,t,_,audio_l,txt_l in self.val_dl:
                x = x.to(self.device)
                t = t.to(self.device)
                audio_l = audio_l.to(self.device)
                txt_l = txt_l.to(self.device)

                y, in_lengths = self.network(x, audio_l)

                J = self.criterion(y,t,in_lengths,txt_l)
                self.running_val_loss+=J.item()
        #Update validation metrics
        J_val=self.running_val_loss/len(self.val_dl)
        self.val_loss_epoch[len(self.val_loss_epoch)] = J_val
        self.running_val_loss = 0.
        prior_best = self.best_val_loss
        if J_val <=self.best_val_loss:
            self.best_val_loss = J_val
            self.best_val_epoch = len(self.val_loss_epoch)-1
            self._save_best_ckpt()
        #Update the delta queue
        self.delta_queue.append(self.best_val_loss-prior_best)
        return J_val
        
    def _train_dial_class(self,optimizer,patience,lr_scheduler):
        #@TODO: If necessary, implement this
        raise Exception("Task not implemented")
    
    def _initialize_metrics(self,patience):
        #Accumulate train loss over all batches epoch, divide by num batches, and reset to 0
        self.running_train_loss = 0.
        self.running_val_loss = 0.

        #Epoch:train_loss
        self.train_loss_epoch={}
        #Update:train_loss
        self.train_loss_batch={}
        #Epoch:val_loss
        self.val_loss_epoch = {}

        self.best_val_epoch = 0
        self.best_val_loss = torch.inf

        #queue of length of patience; store the change in the best validation loss after each epoch
        self.delta_queue = deque(maxlen=patience)
    
    def _terminate(self,patience):
        #Keep going until all the items in the queue are 0 (x epochs with no change in best validation loss)
        return False if (any(self.delta_queue)&len(self.delta_queue)==patience) else True
    
    def _save_best_ckpt(self):
        os.makedirs(f'./{self.name}/checkpoints',exist_ok=True)
        if os.path.isfile(f'./{self.name}/checkpoints/best_checkpoint.pt'):
            os.remove(f'./{self.name}/checkpoints/best_checkpoint.pt')

        torch.save({
            'epoch':len(self.best_val_epoch),
            'loss':self.train_loss_epoch[self.best_val_epoch],
            'model_state_dict':self.network.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scheduler_state_dict':self.scheduler.state_dict()
        },f'./{self.name}/checkpoints/best_checkpoint.pt')