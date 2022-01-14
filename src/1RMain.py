from classifier.OneRule import OneRule
from solution.Solution import Solution

#Script to run the solution using the One-Rule classifier

datasetPath = "../dataset/lenses_withRepeatedPatterns"
classifier = OneRule()

solution = Solution(classifier,datasetPath)
solution.run()