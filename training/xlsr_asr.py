import torch
import torchaudio

class XLSR_ASR(torch.nn.Module):
    def __init__(self,O:int, model: str="XLSR_300M"):
        """Provide name of XLS_r model (should be available in torchaudio) and the size of the output layer (vocab size)"""
        super(XLSR_ASR,self).__init__()

        if model =="XLSR53":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        elif model=="XLSR_300M":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        elif model=="XLSR_1B":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
        elif model=="XLSR_2B":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
        else:
            raise Exception('Model not found')

        self.encoder = self.bundle.get_model()
        self.fc = torch.nn.Linear(self.bundle._params["encoder_embed_dim"],O)

    def forward(self,x):
        #Get features
        features, _ = self.encoder.extract_features(x)
        #Return list of features for all layers in xlsr, so take only the final to do classification
        features = features[-1]
        #Reduce to vocabulary dimensions
        logits = self.fc(features)
        return logits