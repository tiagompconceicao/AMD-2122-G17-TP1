
from data.DataPreparation import DataPreparation
from classifier.Classifier import Classifier

class Evaluator:
    def __init__(self):
        pass

    #Evaluating an classifier with a given dataset using stratified 10-fold cross validator algorithm
    #Returns the accuracy average
    def stratifiedTenFoldCrossValidator(self,dataset,classifier:Classifier):
        dataPreparation = DataPreparation()
        accuracys = []
        for evaluationDataset in dataPreparation.nFoldCrossPartition(dataset,10):
            accuracy = 0
            total = len(evaluationDataset.testDataset)
            classifier.train(evaluationDataset.trainDataset)
            for testTuple in evaluationDataset.testDataset:
                predicted = classifier.predict(testTuple)
                if predicted == testTuple[testTuple.domain.class_var]:
                    accuracy += 1
            accuracys.append(accuracy/total)

        accuracySum = 0
        for accuracy in accuracys:
           accuracySum += accuracy
        

        return (accuracySum/len(accuracys)) * 100