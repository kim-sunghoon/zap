import os
from glob import glob
import numpy
import pandas as pd
import natsort
import argparse

model_names = ['alexnet-cifar100', 'alexnet-imagenet',
               'vgg16-imagenet',
               'resnet18-imagenet']
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names,
                    default = 'alexnet-cifar100',
                    help='model architectures and datasets:\n ' + ' | '.join(model_names))
    return parser.parse_args()

def read_csv(csv_name):
    return pd.read_csv(csv_name)

if __name__ == "__main__":
    args = parse_opts()

    csv_lists_ = glob(os.path.join("filter*", "{}*.csv".format(args.arch)))
    # add filter mode 0's csv files
    csv_lists_.extend(glob(os.path.join("data", "{}*.csv".format(args.arch))))

    #  for csv_list in csv_lists_:
    #      if "summary_csv" in csv_list:
    #          del csv_lists_[csv_lists_.index(csv_list)]

    #### super easy sorting in python, window style sorting
    csv_lists_ = natsort.natsorted(csv_lists_, reverse=False)
    print(csv_lists_)

    read_lists = []
    for csv_list in csv_lists_:
        readlines = read_csv(csv_list)
        read_lists.append(readlines)
    for i in range(1, len(read_lists)):
        read_lists[0] = pd.merge(read_lists[0], read_lists[i], on=['mask', 'th'])

    headers = read_lists[0].columns.tolist()
    print(headers)

    read_lists[0].to_csv("summary_csv/summary_{}.csv".format(args.arch), index=False)



