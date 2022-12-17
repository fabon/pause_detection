from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

import tgt

BATCH_SIZE=16
SENTENCE_LENGTH=42
NUM_TAGS=1

pause_dataset_sample_fields = "phonemes", "pauses"
PauseDatasetSample = namedtuple("PauseDatasetSample", ("id", *pause_dataset_sample_fields))
pause_dataset_batch_fields = (*pause_dataset_sample_fields, "sample_lengths")
PauseDatasetBatch = namedtuple("PauseDatasetBatch", ("id", *pause_dataset_batch_fields))

arpa_symbols = [
    "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY",
    "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1",
    "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1", "OW2", "OY", "OY0",
    "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1", "UH2", "UW",
    "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"
]
arpa_symbols_index={k:v for (k,v) in list(zip(arpa_symbols, [v+1 for v in range(len(arpa_symbols))]))}
arpa_silence_symbols = ["sp", "spn", "sil", ""]


class PauseDataset(Dataset):
    def __init__(self, metadata_path, data_root):
        self.metadata_path = metadata_path
        self.data_root = Path(data_root)

        with open(self.metadata_path, "r") as f:
            self.samples = [self.data_root / path.rstrip() for path in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        textgrid_path = self.samples[idx]

        sample = self.load_sample(textgrid_path)

        return sample

    def collate_fn(self, data):
        batch = dict(
            id=[sample.id for sample in data],
            sample_lengths=torch.IntTensor([len(sample.phonemes) for sample in data])
        )

        for key in pause_dataset_sample_fields:
            batch[key] = pad_sequence([getattr(sample, key) for sample in data],
                                      batch_first=True)

        return PauseDatasetBatch(**batch)

    @staticmethod
    def load_sample(textgrid_path):
        # TODO: replace the following stub logic to load phonemes and pauses from textgrid file
        phonemes = torch.zeros(SENTENCE_LENGTH, dtype=torch.int)
        pauses = torch.zeros(SENTENCE_LENGTH, dtype=torch.bool)

        tgt_obj=tgt.io.read_textgrid(textgrid_path)
        tier=tgt_obj.tiers[1]
        # print (tier.start_time)
        # print (tier.end_time)
        # print (len(tier.annotations))
        phonemes_list=[0]
        label_list=[False]
        # print (textgrid_path)
        cpt=1
        for annotation in tier.annotations:
            cpt=cpt+1
            if cpt > SENTENCE_LENGTH:
                break
            # print (annotation)
            # print (annotation.start_time)
            # print (annotation.end_time)
            # print (annotation.text)
            if annotation.text not in arpa_silence_symbols:
                phonemes_list.append(arpa_symbols_index[annotation.text])
                label_list.append(False)
            else:
                label_list[-1]=True
        # print (len(phonemes_list))
        # print (len(label_list))
        # print (np.sum(label_list))

        phonemes_list=phonemes_list + [0] *(SENTENCE_LENGTH-len(phonemes_list))
        label_list=label_list + [0] *(SENTENCE_LENGTH-len(label_list))

        phonemes=torch.IntTensor(phonemes_list)
        pauses=torch.BoolTensor(label_list)

        # print (phonemes.size())
        # print (pauses.size())
        sample = PauseDatasetSample(id=textgrid_path.stem,
                                    phonemes=phonemes,
                                    pauses=pauses)

        return sample

    @staticmethod
    def batch_to_device(batch, device):
        device_tensors = [getattr(batch, key).to(device) for key in pause_dataset_batch_fields]
        return PauseDatasetBatch(batch.id, *device_tensors)

if __name__ == "__main__":
    dataset = PauseDataset("dataset/train.txt", "dataset")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            collate_fn=dataset.collate_fn)

    for sample in dataloader:
        print(sample.phonemes.shape)


    model=CRF(NUM_TAGS)
