from data.Model import EvaluationDataset
import Orange as DM 

class DataPreparation:
    def __init__(self):
        pass

    def discretizeAttribute(self, dataset, featureName):
        pass

    #Auxiliary function to dynamically create n datasets splitted, each one, in n parts
    #This function allows to apply the 10-Fold cross validation algorithm
    def nFoldCrossPartition(self, dataset,n:int):
        datasets = []
        
        splitedDataset = list(self.__split(dataset, n))
        train = []
        for i in range(n):
            test = splitedDataset[i]
            train = []
            for j in range(n):
                if j == i:
                    continue
                else:
                    for tuple in splitedDataset[j]:
                        train.append(tuple)
            datasets.append(EvaluationDataset(DM.data.Table.from_list(dataset.domain, test),
                                              DM.data.Table.from_list(dataset.domain,train)))
        return datasets

    def __split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))