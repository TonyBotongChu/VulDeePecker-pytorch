"""
ref: https://github.com/johnb110/VDPython/blob/master/vuldeepecker.py
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""
import os
import sys

import torch

from cgd_data import CGDDataset
from config import DefaultTrainConfig
from blstm import BLSTM
from fit import Fitter
# from vectorize_gadget import GadgetVectorizer


"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def main():
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        # print("Usage: python vuldeepecker.py [filename]")
        # exit()
    else:
        filename = "cwe119_cgd.txt"
    cgd_data = CGDDataset(filename, 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DefaultTrainConfig()
    model = BLSTM(
        config.input_size,
        config.hidden_size,
        config.num_layers,
        config.num_classes,
        config.dropout,
        device,
    ).to(device)
    print("[*] training model...")
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adamax(
    #     model.parameters(), lr=config.learning_rate
    # )

    fitter = Fitter(model, device, config)
    total_result = fitter.cross_validation(cgd_data)
    # if result.output is None:
    #     f = open("./cross_val.csv", "w")
    # else:
    #     f = result.output
    # f = open("./cross_val.csv", "w")
    # f.write(
    #     "fold,epoch,train_f1,val_f1,train_acc,val_acc,train_recall,val_recall,train_loss,val_loss\n"
    # )
    # for fold, fold_result in enumerate(total_result):
    #     for epoch, (
    #             train_summary_loss,
    #             train_total_score,
    #             val_summary_loss,
    #             val_total_score,
    #     ) in enumerate(fold_result):
    #         print(
    #             fold + 1,
    #             epoch + 1,
    #             train_total_score.f1,
    #             val_total_score.f1,
    #             train_total_score.precision,
    #             val_total_score.precision,
    #             train_total_score.recall,
    #             val_total_score.recall,
    #             train_summary_loss.avg,
    #             val_summary_loss.avg,
    #             sep=",",
    #             file=f,
    #         )
    # f.close()

if __name__ == "__main__":
    main()