import torch

class Inference():

    def __init__(self, task, model,ckpt_path, test_set, experiment_name):
        if task in ["asr","dial_class","adversarial"]:
            self.task = task
        else:
            raise Exception("Task not implemented")
        
        #Load model
        self.model = model
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

        self.eval_set = test_set

        self.name = experiment_name

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def evaluate(self, beam_size: int):
        if self.task=="asr":
            pass
            #Should I have a separate module of decoder classes that get defined here? Greedy if beam size =1 else beam search.. 
        elif self.task=="adversarial":
            pass
        else:
            raise Exception("Task not implemented")
