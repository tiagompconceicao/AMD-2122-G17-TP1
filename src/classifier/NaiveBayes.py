from classifier.Classifier import Classifier
import data.DataStatistics as dataStatistics
from data.Model import FeatureError, FeatureProbability

#Class that implements the Naive-Bayes classifier algorithm
class NaiveBayes(Classifier):
    featureProbabilites: list
    classDomain: list
    def __init__(self):
        self.featureProbabilites = []
        self.classDomain = []
        super().__init__()

    #Train the classifier
    #Build the array of conditional probabilities of each value of an attribute knowing the value of the class
    def train(self, dataset):
        attributeNames = []
        for attribute in dataset.domain.variables:
            attributeNames.append(attribute.name)

        for feature in attributeNames:
            ( rowDomain, colDomain, pMatrix ) = dataStatistics.get_featureProbability( dataset, dataset.domain.class_var, feature, True )
            if len(self.classDomain) == 0:
                self.classDomain = rowDomain
            self.featureProbabilites.append(FeatureProbability(feature,pMatrix,colDomain))

    #Method to predict the value of the class for an given pattern
    #It will iterates the array of probabilities and calculates the probabilities of each value of the class knowing the values of the attributes
    def predict(self, dataset):
        classProbabilities = [1,1,1]
        for feature in dataset.domain.attributes:
            for trainedFeature in self.featureProbabilites:
                if feature.name == trainedFeature.name:
                    aux = trainedFeature.domain.index(dataset[feature])
                    for i in range(len(trainedFeature.probabilities[:, aux])):
                         classProbabilities[i] *= trainedFeature.probabilities[:, aux][i]

        bestProbability = classProbabilities[0]
        for i in range(len(classProbabilities)):
            if classProbabilities[i] > bestProbability:
                bestProbability = classProbabilities[i]
        return self.classDomain[classProbabilities.index(bestProbability)]
        
