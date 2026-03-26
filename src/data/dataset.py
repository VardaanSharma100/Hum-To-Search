import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class HummingDataset(Dataset):
    def __init__(self, processed_dir):

        self.hum_mel_dir = os.path.join(processed_dir, 'hummings', 'mel')
        self.hum_pitch_dir = os.path.join(processed_dir, 'hummings', 'pitch')
        self.song_mel_dir = os.path.join(processed_dir, 'songs', 'mel')
        self.song_pitch_dir = os.path.join(processed_dir, 'songs', 'pitch')

        self.file_ids = self._get_valid_ids()

    def _get_valid_ids(self):

        hum_mels = set([])