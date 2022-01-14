from classifier.ID3 import ID3
from solution.Solution import Solution

#Script to run the solution using the ID3 classifier

datasetPath = "../dataset/lenses_withRepeatedPatterns"
classifier = ID3()

solution = Solution(classifier,datasetPath)
solution.run()