import torch
import torchaudio


class XLSR_ASR(torch.nn.Module):
    def __init__(self, O: int, model: str = "XLSR_300M", freeze_model: bool = True):
        """Provide name of XLS_r model (should be available in torchaudio) and the size of the output layer (vocab size)"""
        super(XLSR_ASR, self).__init__()

        if model == "XLSR53":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        elif model == "XLSR_300M":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        elif model == "XLSR_1B":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
        elif model == "XLSR_2B":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
        else:
            raise Exception("Model not found")

        self.encoder = self.bundle.get_model()
        if freeze_model:
            # Freeze all the param in the XLSR model
            for p in self.encoder.named_parameters():
                p[1].requires_grad = False
        else:
            # Freeze only the CNN Feature extractor, as in original XLSR paper
            for p in self.encoder.named_parameters():
                if "feature_extractor" in p[0]:
                    p[1].requires_grad = False
        self.fc = torch.nn.Linear(self.bundle._params["encoder_embed_dim"], O)

    def forward(self, x, lengths):
        # Get features

        # Return list of features for all layers in xlsr, so take only the final to do classification
        features, lengths = self.encoder(waveforms=x, lengths=lengths)

        # Reduce to vocabulary dimensions
        # batchxtime_framexalphabet_dim
        logits = self.fc(features)

        # Get log probabilites out of logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # reshape to CTC requirements: in_lengthxBxalphabet_dim; return lengths of input in adjusted time frame
        return log_probs.transpose(0, 1), lengths
