from typing import NamedTuple

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from clean_gadget import clean_gadget

from vectorize_bert import BERTVectorizer
# from vectorize_gadget import GadgetVectorizer


def parse_info_dic(info: str):
    info_list = info.strip().split()
    file_info = info_list[1]
    vul_name = file_info.split("/")[0]
    file_name = file_info.split("/")[1]

    # if vulerability name is numeric, then it is from SARD
    # otherwise it's from NVD
    if vul_name.isnumeric():
        project_name = "SARD"
        project_version = ""
    else:
        project_info = file_name.split("_"+vul_name)[0]
        project_name = project_info.split("_")[0]
        project_version = project_info.split("_")[1]
    info_dic = {
        "num" : info_list[0],
        "file_info": file_info,
        "cwe": vul_name,
        "project_name": project_name,
        "project_version": project_version,
        "language": info_list[2],
        "line": info_list[3]
    }
    return info_dic


class CGDDataset(Dataset):

    def __init__(self, filename: str, vector_length: int):
        self.infos = []
        self.cgds = []
        self.labels = []
        self.project = []
        # self.vectorizer = GadgetVectorizer(vector_length)
        self.vectorizer = BERTVectorizer(vector_length)
        with open(filename, "r", encoding="utf8") as file:
            current_case = []
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                if "-" * 33 in line and current_case:
                    info_dic = parse_info_dic(current_case[0])
                    self.infos.append(info_dic)
                    self.project.append(info_dic["project_name"])
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
