from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch
from typing import Literal

class DatasetUCS(Dataset):
    """
    Expects the path to a .json file with model configs.

    Args:
    - sample_rate: resample the data to the given sample rate. Set to None to skip resampling.
    - max_length: waveforms beyond this value (in seconds) will be truncated, waveforms below this value will be looped.
    - class_reduction: None or list of classes to exclude.
    - include: inverts class_reduction to expect a list of classes to include.
    """
    def __init__(self, df:pd.DataFrame, settings, device : torch.device, class_reduction=None, include=False, return_type:Literal['path','waveform','spectrogram']='spectrogram'):
        super().__init__()
        
        self.sample_rate = settings['sample_rate']
        self.max_samples = (int)(settings['sample_length_in_seconds'] * self.sample_rate)
        self.device = device
        self.frame_size = settings['frame_size']
        self.n_fft = settings['n_fft']
        self.hop_length = settings['hop_length']
        self.n_mel_bins = settings['num_of_bins']
        self.return_type = return_type

        sinusoidal_window_fn = lambda L : torch.sqrt (torch.hann_window (L))
        spect_transform = torchaudio.transforms.MelSpectrogram (sample_rate=self.sample_rate, n_fft=self.frame_size, win_length=self.frame_size, hop_length=self.hop_length, n_mels=self.n_mel_bins, power=2,
                                                               window_fn = sinusoidal_window_fn)
        self.spect_transform = spect_transform.to(device)

        # The dataset contains a lot of high resolution data. We resample them to save time.
        if (self.sample_rate != None):
            self.resample_transform_192k = torchaudio.transforms.Resample(orig_freq = 192000, new_freq = self.sample_rate).to(self.device)
            self.resample_transform_96k = torchaudio.transforms.Resample(orig_freq = 96000, new_freq = self.sample_rate).to(self.device)
            self.resample_transform_48k = torchaudio.transforms.Resample(orig_freq = 48000, new_freq = self.sample_rate).to(self.device)
            self.resample_transform_44k = torchaudio.transforms.Resample(orig_freq = 44100, new_freq = self.sample_rate).to(self.device)

        
        if (class_reduction != None):
            mask = df['category'].isin(class_reduction)
            df = df[mask] if (include) else df[~mask]
            df.reset_index(inplace=True)

        # Load path for each file. The actual files will be loaded during training.
        self.audio_paths = df["path"]
        self.audio_drive = settings["dataset_drive"]
        self.channels = df["channels"]
        self.targets = df['category']
        self.unique_classes = df['category'].unique()
        self.n_classes = len(self.unique_classes)

    def get_unique_classes(self):
        """
        Returns a list of all classes used in the dataset.
        """
        return self.unique_classes
    
    def __len__(self):
        return len (self.audio_paths)

    def __getitem__(self, idx):
        """
        Get the waveform during training, convert it to a mel spectrogram.
        """
        # Load label from path+filename, removes commas, and make lower case. TODO: Create function for this to make it more readable and flexible.
        label = self.audio_paths[idx].split('/')[-1].split('.')[0].replace(',', '').lower().strip()
        # Remove Digits
        label = ''.join([i for i in label if not i.isdigit()])
        label = label.strip()

        # Change drive if necessary
        path = self.audio_paths[idx]
        if (path[:3] != self.audio_drive):
            path = path.replace(path[:3], self.audio_drive)

        target = self.targets[idx]

        if(self.return_type == 'path'):
            return path, label, target

        # Load audio file
        waveform, orig_sample_rate = torchaudio.load(path)
        waveform = waveform.to(self.device)

        # If the original sample_rate is not one of these values, resampling will be ignored. Works with multiple channels.
        if (orig_sample_rate == 192000):
            waveform = self.resample_transform_192k(waveform)
        if (orig_sample_rate == 96000):
            waveform = self.resample_transform_96k(waveform)
        if (orig_sample_rate == 48000):
            waveform = self.resample_transform_48k(waveform)
        if (orig_sample_rate == 44100):
            waveform = self.resample_transform_44k(waveform)

        # Convert to mono
        waveform = waveform[0].unsqueeze(0)

        # Loop waveform if it's too short.
        if (waveform.size(dim=1) < self.max_samples):
            while waveform.size(dim=1) < self.max_samples:
                waveform = torch.cat((waveform, waveform), dim=1)

        # Truncate waveform if it's too long.
        if (len(waveform[0]) > self.max_samples):
            waveform = waveform.narrow(1, 0, self.max_samples)

        # To Mel Spectrogram (currently handled in HTSAT using torchlibrosa.stft)
        #spectrogram = self.spect_transform(waveform)

        return waveform, label, target

if __name__ == "__main__":
    import json
    from torch.utils.data import DataLoader
    with open ('model_definitions/clap.json', 'r') as f:
        settings = json.load (f)

    # Load dataset
    train_df = pd.read_csv(settings['dataset_drive'] + settings['train_set_path'])
    dataset_train = DatasetUCS(train_df, settings=settings, device='cuda', class_reduction=['ambience', 'vehicles'])
    train_loader = DataLoader(dataset_train, batch_size=settings['batch_size'], shuffle=True)

    iterator = iter(train_loader)
    batch_01 = next(iterator)
    print(batch_01)
    batch_02 = next(iterator)
    print(batch_02)
