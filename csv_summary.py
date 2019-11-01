import os
from glob import glob
import numpy
import pandas as pd

def read_csv(csv_name):
    return pd.read_csv(csv_name)

if __name__ == "__main__":
    csv_lists_ = sorted(glob("./*/*.csv"))
    for csv_list in csv_lists_:
        print(csv_list)
        if "summary_csv" in csv_list:
            del csv_lists_[csv_lists_.index(csv_list)]
    print(csv_lists_)
    read_lists = []
    for csv_list in csv_lists_:
        readlines = read_csv(csv_list)
        read_lists.append(readlines)
    for i in range(1, len(read_lists)):
        read_lists[0] = pd.merge(read_lists[0], read_lists[i], on=['mask', 'th'])

    print(read_lists[0])
    headers = read_lists[0].columns.tolist()

    #  read_lists[0].to_csv("summary_csv/summary.csv")



