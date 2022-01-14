from classifier.OneRule import OneRule
from data.Data import Data
import data.DataStatistics as dataStatistics


#Performs the number of incorrect predictions for each feature of the rule attribute
def simpleValidator(dataset):
    attr = classifier.classificationAttribute
    incorrectPredictions = {}
    for feature in classifier.classificationAttribute.featuresErrors:
        incorrectPredictions[feature.name] = 0

    for tuple in dataset:
        predicted = classifier.predict(tuple)
        if predicted != tuple[tuple.domain.class_var]:
            incorrectPredictions[tuple[attr.name].value] += 1
    return incorrectPredictions

data = Data()
dataset = data.load("../dataset/dataset_A1")
classifier = OneRule()
classifier.train(dataset)
f = open("../oneR_OUTPUT.txt", "w")
attr = classifier.classificationAttribute

#get the number of occurences of each feature in the rule attribute
count = dataStatistics.get_tupleCountByFeature(dataset, attr.name)

incorrectPredictions = simpleValidator(dataset)
index = 0   
for feature in classifier.classificationAttribute.featuresErrors:
    f.write("( %s, %s, %s ) : (%d, %d)\n" % (attr.name,feature.name,feature.classLabel,incorrectPredictions[feature.name],count[index]))
    print("( %s, %s, %s ) : (%d, %d)" % (attr.name,feature.name,feature.classLabel,incorrectPredictions[feature.name],count[index]))
    index += 1

f.close()


