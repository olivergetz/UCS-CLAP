import torch
import torch.nn as nn
import torchaudio
from typing import List
from transformers import RobertaModel, RobertaForSequenceClassification, RobertaForTokenClassification, AutoTokenizer, RobertaTokenizer, AutoModel
from models.htsat.htsat import HTSAT_Swin_Transformer
import models.htsat.htsat_config as config

class HANCECLAP(nn.Module):
    def __init__(self,
        settings,
        device: torch.device,
        in_channels = 1,
        print_shapes:bool = False
        ):
        
        super().__init__()

        num_of_input_channels = settings['num_of_input_channels']       # Number of input channels to the first layer of the encoder. Mono = 1. Stereo = 2.
        in_channels = num_of_input_channels                             # Used to update the channels for each layer.
        num_of_output_channels = settings['num_of_output_channels']
        self.print_shapes = print_shapes                                # For debugging shape mismatch.
        
        self.device = device
        self.joint_embed_shape = settings['joint_embed_size'] # n*n

        self.activation_fn = {'leaky': nn.LeakyReLU(), 'relu': nn.ReLU(), 'elu':nn.ELU()}[settings['activation_type']]
        #self.final_activation = {'leaky': nn.LeakyReLU(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[settings['final_activation_type']]

        # Build Architecture
        self.text_encoder = AutoModel.from_pretrained(settings['text_encoder_name']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings['text_encoder_name'])  # No custom vocab, labels don't use audio specific terms.

        # SWIN and SED are from https://github.com/RetroCirce/HTS-Audio-Transformer
        # Used to load HTSAT checkpoint.
        self.audio_encoder = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        ).to(self.device)

        self.audio_encoder_ckpt = torch.load(settings['audio_encoder_path'])
        self.audio_encoder.load_state_dict(self.audio_encoder_ckpt['state_dict'], strict=False)

        # MLP as a projection layer, instead of just a linear layer, as outlined by Chen et al. (2020)
        if (settings['projection'] == 'mlp'):
            self.audio_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                self.activation_fn,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
            ).to(self.device)

            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                self.activation_fn,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
            ).to(self.device)
        
        if (settings['projection'] == 'linear'):
            self.audio_projection = nn.Linear(768, self.joint_embed_shape).to(self.device)
            self.text_projection = nn.Linear(768, self.joint_embed_shape).to(self.device)

    def forward(self, wave, label, target):
        """
        Expected input is a spectrogram of shape (batch, channel, frequency bins, time frames)
        """
        if (self.print_shapes):
            print(f"Model input size: {wave.size()}")

        wave = wave.to(self.device)

        # Encode Audio and Text
        encoded_audio = self.audio_encoder(wave.squeeze(dim=1))['latent_output'] # HTSAT doesn't like the channel dimension. ['framewise_output' | 'clipwise_output' | 'latent_output'] 
        tokens = self.tokenizer(label, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokens = tokens.to(self.device)
        # Average pooling over hidden representations to retrieve sentence embeddings.
        encoded_text = torch.mean(self.text_encoder(tokens['input_ids'], tokens['attention_mask']).last_hidden_state, dim=1)

        if (self.print_shapes):
            print(f"Audio Encoded Shape: {encoded_audio.shape}")
            print(f"Text Encoded Shape: {encoded_text.shape}")

        audio_features = self.audio_projection(encoded_audio)
        text_features = self.text_projection(encoded_text)

        if (self.print_shapes):
            print(f"Audio Features Shape: {audio_features.shape}")
            print(f"Text Features Shape: {text_features.shape}")

        return  (audio_features, text_features)
 

    if __name__ == "__main__":
        pass