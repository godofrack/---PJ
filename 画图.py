import matplotlib.pyplot as plt
import csv
import numpy as np


datasets = ['cifar10']
models = ['resnet18']
methods = ['cutmix', 'cutout', 'baseline', 'mixup']
for dataset in datasets:
    for model in models:
        name = dataset + '_' + model
        method = [[] for i in range(5)]
        with open('./结果3.csv', 'r') as f:
            f_csv = csv.reader(f)
            k = 0
            for row in f_csv:
                k += 1
                if k == 1:
                    continue
                for i in range(5):
                    method[i].append(float(row[i]))

        method = np.array(method)
        for i in range(4):
            method[i+1] =method[i+1]
        plt.clf()
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss(%)')
        plt.title(name)
        for i in range(4):
            plt.plot(method[0], method[i+1], label=methods[i])
        plt.legend(methods)
        plt.savefig('./figures/'+name+'.png', format='png')
        plt.show()
