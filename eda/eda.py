from collections import namedtuple, OrderedDict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import tgt

BATCH_SIZE=16
SENTENCE_LENGTH=100
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

    def loadFromDisc(self):
        data=[]
        data_n_pauses=[]
        nFiles=3000
        for sample in self.samples:
            sentence, total_n_pauses=self.load(sample)
            data.append(sentence)
            data_n_pauses = data_n_pauses + total_n_pauses
            if nFiles == 0:
                break
            nFiles=nFiles-1
        return (data, data_n_pauses)

    @staticmethod
    def load(textgrid_path):
        tgt_obj=tgt.io.read_textgrid(textgrid_path)
        tier=tgt_obj.tiers[1]
        # print (tier.start_time)
        # print (tier.end_time)
        # print (len(tier.annotations))
        phonemes_list=[0]
        label_list=[False]
        # print (textgrid_path)
        cpt=1
        total_n_pauses=[]
        for annotation in tier.annotations:
            cpt=cpt+1
            if cpt > SENTENCE_LENGTH:
                break
            # print (annotation)
            # print (annotation.start_time)
            # print (annotation.end_time)
            # print (annotation.text)
            if annotation.text not in arpa_silence_symbols:
                phonemes_list.append(annotation.text)
                label_list.append(False)
            else:
                label_list[-1]=True
        # print (len(phonemes_list))
        # print (len(label_list))
        total_n_pauses.append(np.sum(label_list))
        phonemes_list=phonemes_list + [0] *(SENTENCE_LENGTH-len(phonemes_list))
        label_list=label_list + [0] *(SENTENCE_LENGTH-len(label_list))
        sample=(phonemes_list, label_list)
        return (sample, total_n_pauses)

def findPauseIndices(labels):
    return [index for index, pause in enumerate(labels) if pause]

def pauseLocations(data):
    data_pauses=[sent[1] for sent in data]

    all_positions=[]
    all_durations=[]
    nStart=0
    nNoPause=0
    for labels in data_pauses:
        positions = findPauseIndices(labels)
        durations=[pos - positions[i - 1] for i, pos in enumerate(positions)][1:]
        if len(positions):
            if positions[0] == 0:
                nStart=nStart+1
        else:
            nNoPause=nNoPause+1
        all_positions=all_positions+positions
        all_durations=all_durations+durations

    print (nStart/len(data))
    print (nNoPause/len(data))
    plt.hist(all_durations, 100)
    plt.show()
    print (np.median(all_durations))
    
def addToDict(v, cpt):
    if v not in cpt:
        cpt[v] = 0
    cpt[v] = cpt[v]+1

def sortDict(cpt):
    return OrderedDict(sorted(cpt.items(), key=lambda tup:tup[1], reverse=True))

def popularPhonemes(data):
    counts_paused={}
    all_counts={}
    for i in range(len(data)):
        print (i)
        data_phonemes=data[i][0]
        for phoneme in data_phonemes:
            addToDict(phoneme, all_counts)
        
        data_pauses=data[i][1]
        positions=findPauseIndices(data_pauses)
        paused_phonemes= [data_phonemes[i] for i in positions]
        for phoneme in paused_phonemes:
            addToDict(phoneme, counts_paused)
    sorted_paused=sortDict(counts_paused)
    sorted_all=sortDict(all_counts)
    total=np.sum([v for v in sorted_paused.values()])

    ratios={}
    for k,v in sorted_paused.items():
        ratios[k]=v/total
        print ((k,v/total))
    ratios=sortDict(ratios)
    print (np.sum([v for v in list(ratios.values())[:20]]))
    print (len(ratios))
    
    # for k,v in all_counts.items():
    #     print ((k,v))

    merged={}
    for k,v in sorted_paused.items():
        merged[k]=sorted_paused[k]/all_counts[k]
    merged=sortDict(merged)
    # for k,v in merged.items():
    #     print ((k,v,sorted_paused[k]))

    
    # print (np.sum([v for v in list(merged.values())[:10]]))
    
        
    

if __name__ == "__main__":
    dataset = PauseDataset("../dataset/train.txt", "../dataset")
    data, total_n_pauses=dataset.loadFromDisc()
    print (total_n_pauses)
    # pauseLocations(data)
    popularPhonemes(data)

    # plt.hist(total_n_pauses,5)
    # plt.show()
