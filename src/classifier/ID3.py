from classifier.Classifier import Classifier
from data.Data import Data
import data.DataStatistics as dataStatistics
import Orange as DM 
from data.Model import TreeLeaf, TreeNode

#Class that implements the ID3 classifier algorithm
class ID3(Classifier):
    data: Data
    treeRoot: TreeNode
    def __init__(self):
        self.data = Data()
        super().__init__()

    #Method to train the ID3 algorithm by generating a decision tree
    def train(self, dataset):
        self.treeRoot = self.__buildTree(dataset)
        return self.treeRoot

    #Method to predict from a given pattern the value of the class using the generated tree
    def predict(self, dataset):
        currentNode = self.treeRoot
        while True:
            attributeValue = dataset[currentNode.name]
            index = currentNode.values.index(attributeValue)
            nextNode = currentNode.childs[index]
            if isinstance(nextNode,TreeLeaf):
                return nextNode.value
            currentNode = nextNode

    #Method to generate the tree
    def __buildTree(self,dataset):
        the_class = dataset.domain.class_var
        domain = dataset.domain
        
        #The dataset has to be at least one attribute
        if (len(domain.variables) - 1) != 0:
            #Verify if all patterns has only one value of the class
            if self.__hasOnlyOneClass(dataset):
                return TreeLeaf(dataset[0][the_class])
            else:
                #Computes the attribute with the lower average entropy
                bestfeatureEntropy = 10
                bestAttribute: DM.data.DiscreteVariable
                for attribute in dataset.domain.attributes:
                    featureEntropy = dataStatistics.get_entropy(dataset,attribute.name)
                    if featureEntropy < bestfeatureEntropy:
                        bestfeatureEntropy = featureEntropy
                        bestAttribute = attribute
                
                #Generate new tree node
                treeNode = TreeNode(name=bestAttribute.name,values=bestAttribute.values)
                
                #Iterate each value of the chosen attribute
                #Divide the dataset by the values of the chosen attribute
                #Remove the chosen attribute from the dataset
                #Generate new tree generation
                for value in bestAttribute.values:
                    subDataset = []
                    classValues = []
                    for tuple in dataset:
                        if (tuple[bestAttribute.name] == value):
                            subDataset.append(tuple) 
                            classValues.append(tuple[the_class]) 
                    array = self.data.convertTableToArray(subDataset,domain)

                    #Removes the chosen attribute to the algorithm be able to generate a new generation
                    newDataset = self.__removeAttribute(self.data.createTable(array,classValues,domain),bestAttribute.name)

                    #Recursive call to build new node in the new tree generation
                    treeNode.childs.append(self.__buildTree(newDataset))
                return treeNode
        else:
            #The tree returns a leaf with None value
            return TreeLeaf(None)

    #For a given dataset verifies if every pattern as the same value of class
    def __hasOnlyOneClass(self, dataset):
        the_class = dataset.domain.class_var
        firstTuple = dataset[0][the_class]
        for tuple in dataset:
            if tuple[the_class] != firstTuple:
                return False
        return True

    #For a given dataset removes from it the attribute from a given name
    #Returns a new dataset without the given attribute
    def __removeAttribute(self,dataset,attributeName):
        newDataset = []
        attributesDomain = []
        classDomain = [dataset.domain.class_var]
        for attribute in dataset[0].domain.attributes:
                if attribute.name != attributeName:
                    index = dataset[0].domain.index(attribute)
                    attributesDomain.append(dataset[0].domain.__getitem__(index))
        newDomain = self.data.createDomain(attributesDomain,classDomain)
        newDataset = self.data.convertTable(dataset,newDomain)
        return newDataset
