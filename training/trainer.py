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

    # def train(self, initial_lr, es_patience, lr_patience, grad_accum, delta=-0.1, logger_level = logging.INFO):
    def train(self, min_eta, max_eta, es_patience, grad_accum, delta=-0.1, logger_level = logging.INFO):
        os.makedirs(f'./{self.name}',exist_ok=True)
        os.makedirs(f'./{self.name}/logs',exist_ok=True)
        logging.basicConfig(filename=f'./{self.name}/logs/{self.name}.log',filemode="w", format='%(asctime)s %(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logger_level)

        if self.task=="asr":
            #self.optimizer = torch.optim.Adam(self.network.parameters(),lr = initial_lr)
            #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=lr_patience)
            #If using cosine annealing warm restarts, then the optimizer lr gets set to the max lr
            self.optimizer = torch.optim.Adam(self.network.parameters(),lr = max_eta)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=500,eta_min=min_eta)
            self.logger.info(f"ASR training:")
            self._train_asr(es_patience, grad_accum, delta)
        elif self.task=="dial_class":
            #@TODO implement optimizer and scheduler for dialect classification if necessary
            self._train_dial_class(es_patience)
        else:
            raise Exception("Task not implemented")
        
    def _train_asr(self,patience, accum, delta):
        self.network = self.network.to(self.device)

        self._initialize_metrics()

        if accum:
            #accumulate gradients for 16 batches before updting weights
            accum_steps = 16
            if len(self.train_dl)%16==0:
                self.iters=len(self.train_dl)/16
            else:
                self.iters=(len(self.train_dl)//16)+1
        else:
            #reduce this to one so updates are done after every batch
            accum_steps = 1
            self.iters=len(self.train_dl)

        while self._terminate(patience):
            self.logger.info(f"Epoch {len(self.train_loss_epoch)+1}:")
            self.logger.info(f"Learning Rate: {self.scheduler.get_last_lr()}")
            self.logger.info("Train Step\tBatch Loss\tLearning Rate")
            self.network.train()
            for i, (x,t,_,audio_l,txt_l) in enumerate(self.train_dl):
                x = x.to(self.device)
                t = t.to(self.device)
                audio_l = audio_l.to(self.device)
                txt_l = txt_l.to(self.device)

                #log_probs of each class at each at each frame; num frames w/out padding in each batch item
                y, in_lengths = self.network(x, audio_l)

                #Calculate the loss
                J = self.criterion(y,t,in_lengths,txt_l)
                #Calculate gradient of loss and make update 
                J = J/accum_steps
                J.backward()

                if ((i+1)%accum_steps==0) or (i+1 == len(self.train_dl)):
                    #If the necessary number of batches' grads have been accumulated (or we reach the last batch) update
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    #Update running batch loss & dict
                    self.running_train_loss+=J.item()
                    self.train_loss_batch[i+1]=J.item()
                    #update learning rate if using cosine annealing w/warm restarts
                    self.scheduler.step(len(self.train_loss_epoch)+i/self.iters)
                    self.logger.info(f"{i+1}\t{J.item()}\t{self.scheduler.get_last_lr()}")
            self.logger.info(f"End of Epoch {len(self.train_loss_epoch)+1}")
            #Update the train epoch loss; reset running train loss to 0
            self.train_loss_epoch[len(self.train_loss_epoch)+1]=self.running_train_loss/(len(self.train_dl)/accum_steps)
            self.logger.info(f"Macro-Average Train loss over epoch:\t{self.train_loss_epoch[len(self.train_loss_epoch)]}")
            self.running_train_loss=0.
            #Run validation and update relevant metrics
            J_val = self._validation_step_asr(delta=delta)
            self.logger.info(f"Average Validation loss over epoch:\t{J_val}")
            self.logger.info(f"Best Validation Epoch:\t{self.best_val_epoch}")
            self.logger.info(f"Best Validation Loss:\t{self.best_val_loss}")
            self.logger.info(f"Patience Counter:\t{self.patience_counter}")
            #Adjust learning rate if doing reduce on plateau
            #self.scheduler.step(J_val)

    def _validation_step_asr(self,delta):
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
        self.val_loss_epoch[len(self.val_loss_epoch)+1] = J_val
        self.running_val_loss = 0.
        prior_best = self.best_val_loss
        if J_val <=self.best_val_loss:
            self.best_val_loss = J_val
            self.best_val_epoch = len(self.val_loss_epoch)
            self._save_best_ckpt()
        #Update the patience counter
        if J_val>prior_best+delta:
            self.patience_counter+=1
        else:
            self.patience_counter=0
        return J_val
        
    def _train_dial_class(self,optimizer,patience,lr_scheduler):
        #@TODO: If necessary, implement this
        raise Exception("Task not implemented")
    
    def _initialize_metrics(self):#_initialize_metrics(self,patience):
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

        #Number of epochs that meet early stop criteria; stop training when it equals es patience
        self.patience_counter = 0
    
    def _terminate(self,patience):
        #Keep going until all the items in the queue are 0 (x epochs with no change in best validation loss)
        return False if (self.patience_counter==patience) else True
    
    def _save_best_ckpt(self):
        os.makedirs(f'./{self.name}/checkpoints',exist_ok=True)
        if os.path.isfile(f'./{self.name}/checkpoints/best_checkpoint.pt'):
            os.remove(f'./{self.name}/checkpoints/best_checkpoint.pt')

        torch.save({
            'epoch':self.best_val_epoch,
            'loss':self.train_loss_epoch[self.best_val_epoch],
            'model_state_dict':self.network.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scheduler_state_dict':self.scheduler.state_dict()
        },f'./{self.name}/checkpoints/best_checkpoint.pt')