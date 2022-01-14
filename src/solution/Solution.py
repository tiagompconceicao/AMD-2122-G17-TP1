from data.Data import Data
from classifier.Classifier import Classifier
import Orange as DM

class Solution:
    classifier: Classifier
    dataset: DM.data.Table
    data: Data
    def __init__(self, classifier,datasetPath):
        self.data = Data()
        self.classifier = classifier
        self.dataset = self.data.load(datasetPath)
        classifier.train(self.dataset)

    def run(self):
        print("Wellcome...")
        running = True
        while running:
            option = input("Choose an option:\n1 - Predict\n0 - Exit\n> ")
            if option == '1':
                pattern = self.__getUserInputPattern()
                print("The result is: %s\n" % self.classifier.predict(pattern))
                pass
            elif option == '0':
                running = False
            else:
                print("Bad input, please input properly\n")

    #Acquires a value from the user for each attribute of the dataset domain to build a pattern to predict
    #Was implemented the content validation of the user input, to respect the nominal values of the dataset
    #The class label value has to be None, must have a "dummy" value to build the pattern
    def __getUserInputPattern(self):
        inputs = []
        for domain in self.dataset.domain.attributes:
            print("%s %s" % (domain,domain.values))
            validInput = False
            while not(validInput):
                attributeinput = input("Insert %s:\n> " %domain.name)
                if attributeinput in domain.values:
                    validInput = True
                    inputs.append(attributeinput)
                else:
                    print("Please insert data properly")
        inputs.append(None)
        return self.data.convertArraytoTable(self.dataset.domain,inputs)