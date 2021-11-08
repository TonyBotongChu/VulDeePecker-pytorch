from typing import NamedTuple

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from clean_gadget import clean_gadget

from vectorize_bert import BERTVectorizer
# from vectorize_gadget import GadgetVectorizer


class CGDDataset(Dataset):
    def __init__(self, filename: str, vector_length: int):
        self.infos = []
        self.cgds = []
        self.labels = []
        # self.vectorizer = GadgetVectorizer(vector_length)
        self.vectorizer = BERTVectorizer(vector_length)
        with open(filename, "r", encoding="utf8") as file:
            current_case = []
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                if "-" * 33 in line and current_case:
                    self.infos.append(current_case[0])
                    gadget = clean_gadget(current_case[1:-1])
                    self.cgds.append(gadget)
                    self.labels.append(int(current_case[-1]))
                    # self.vectorizer.add_gadget(gadget)
                    current_case = []
                else:
                    current_case.append(stripped)
        # self.vectorizer.train_model()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # current_gadget = self.cgds[index]
        # cgd = clean_gadget(current_gadget[1:-1])
        # label = int(current_gadget[-1])
        # return cgd, label
        vectors = self.vectorizer.vectorize(self.cgds[index])
        label = self.labels[index]
        return torch.from_numpy(vectors).float(), torch.tensor(label).long()
