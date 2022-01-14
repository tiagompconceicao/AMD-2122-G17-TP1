from classifier.ID3 import ID3
from classifier.NaiveBayes import NaiveBayes
from classifier.OneRule import OneRule
from evaluator.Evaluator import Evaluator 
from data.Data import Data

#Demo script used to evaluate the three classifiers using the stratified 10-fold cross validator algorithm

data = Data()
evaluator = Evaluator()

#dataset = data.load("../dataset/lenses")
dataset = data.load("../dataset/lenses_withRepeatedPatterns")

print("Evaluating classifiers...")
result1 = evaluator.stratifiedTenFoldCrossValidator(dataset,OneRule())
result2 = evaluator.stratifiedTenFoldCrossValidator(dataset,NaiveBayes())
result3 = evaluator.stratifiedTenFoldCrossValidator(dataset,ID3())

print("The mean accuracy of OneRule is: %.2f %%" % result1)
print("The mean accuracy of Naive-Bayes is: %.2f %%" % result2)
print("The mean accuracy of ID3 is: %.2f %%" % result3)
