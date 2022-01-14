from classifier.NaiveBayes import NaiveBayes
from solution.Solution import Solution

#Script to run the solution using the Naive-Bayes classifier

datasetPath = "../dataset/lenses_withRepeatedPatterns"
classifier = NaiveBayes()

solution = Solution(classifier,datasetPath)
solution.run()