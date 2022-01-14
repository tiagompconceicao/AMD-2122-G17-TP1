from fractions import Fraction


class EvaluationDataset:
    testDataset: list
    trainDataset: list

    def __init__(self,testDataset,trainDataset):
        self.testDataset = testDataset
        self.trainDataset = trainDataset

class AttributeError:
    name: str
    value : float
    featuresErrors: list

    def __init__(self,name,value,featuresErros):
        self.name = name
        self.value = value
        self.featuresErrors = featuresErros


class FeatureError:
    error : Fraction
    name: str
    classLabel: str
    def __init__(self,error,name,classLabel):
        self.error = error
        self.name = name
        self.classLabel = classLabel

class FeatureProbability:
    name: str
    probabilities: list
    domain: list
    classDomain: list

    def __init__(self,name, probabilities, domain):
        self.name = name
        self.probabilities = probabilities
        self.domain = domain

class FeatureEntropy:
    name: str
    entropy: float

    def __init__(self,name, entropy):
        self.name = name
        self.entropy = entropy

class TreeNode:
    name:str
    values: list
    childs: list

    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.childs = []

class TreeLeaf:
    value: str
    def __init__(self, value):
        self.value = value
