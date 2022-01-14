from data.Data import Data
import data.DataStatistics as dataStatistics
from fractions import Fraction
from data.Model import AttributeError, FeatureError
from classifier.Classifier import Classifier
import numpy as np

#Class that implements the One rule classifier algorithm
class OneRule(Classifier):
    data: Data
    classificationAttribute: AttributeError
    def __init__(self):
        self.data = Data()
        super().__init__()

    #Method to train the classifier
    #It will computes the choose the attribute with the lower average error to become the classification rule
    def train(self, dataset):
        attributeNames = []
        attributesError = []
        for attribute in dataset.domain.attributes:
            attributeNames.append(attribute.name)

        for featureName in attributeNames:
            ( classDomain, featureDomain, errorMatrix ) = dataStatistics.get_errorMatrix( dataset, featureName, False )
            if( not (classDomain or featureDomain) ): exit
            attributesError.append(self.__calculateAttributeError(featureName,featureDomain,errorMatrix,classDomain))
        
        self.classificationAttribute = self.__chooseAttribute(attributesError)

    #Method to predict the value of the class for a given pattern
    def predict(self, dataset):
        featureToPredict = self.data.get_variableFrom_str(dataset,self.classificationAttribute.name)
        for feature in self.classificationAttribute.featuresErrors:
            if dataset[featureToPredict] == feature.name:
                return feature.classLabel

    #Calculates all features error with the class and choose the value of the class with the lower error for each feature
    def __calculateAttributeError(self,featureName,featureDomain,errorMatrix,classDomain):
        featureErrors = []
        for feature in range(len(featureDomain)):
            errorFeature = errorMatrix[:, feature]
            errorMin = min( errorFeature )
            index_min = np.argmin(errorFeature)
            featureErrors.append(FeatureError(Fraction(errorMin),featureDomain[feature],classDomain[index_min]))

        numerator = 0
        denominator = 0
        for fraction in featureErrors:
            numerator += fraction.error.numerator
            denominator += fraction.error.denominator
        attributeError = numerator/denominator
        return AttributeError(featureName,attributeError,featureErrors)
        
    #For a given set of attributes choose the one with the lower error average
    #In case of two or more attributes as the same error average, the attribute is chosen randomly
    def __chooseAttribute(self,attributesError):
        errorMin = attributesError[0]
        equalErrors = []
        for error in attributesError:
            if (error.value == errorMin.value):
                equalErrors.append(error)
            if error.value < errorMin.value:
                errorMin = error
        if len(equalErrors) == 1:
            equalErrors = []
        if len(equalErrors) != 0:
            errorMin = errorMin = np.random.choice(equalErrors,1)[0]
        return errorMin