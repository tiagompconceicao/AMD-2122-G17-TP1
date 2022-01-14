import numpy as np
import math
from data.Data import Data

data = Data()

#Functions Based on the functions present on the Module of Practice 5


# contingencyMatrix
# this implementation does account for missing-values
# M(row, column)
def get_contingencyMatrix( dataset, rowVar, colVar, zeroFrequencyTolerance ):
   if( isinstance( rowVar, str ) ): rowVar = data.get_variableFrom_str( dataset, rowVar )
   if( isinstance( colVar, str ) ): colVar = data.get_variableFrom_str( dataset, colVar )
   if( not (rowVar and colVar) ): return ( [], [], None )
   if( not (rowVar.is_discrete and colVar.is_discrete) ):   
      return ( [], [], None )
   
   rowDomain, colDomain = rowVar.values, colVar.values
   len_rowDomain, len_colDomain = len( rowDomain ), len( colDomain )
   contingencyMatrix = np.zeros( (len_rowDomain, len_colDomain) )
   for instance in dataset:
      rowValue, colValue = instance[rowVar], instance[colVar]
      if( np.isnan( rowValue ) or np.isnan( colValue ) ): continue
      
      rowIndex, colIndex = rowDomain.index(rowValue), colDomain.index( colValue )
      contingencyMatrix[ rowIndex, colIndex ] += 1

   if zeroFrequencyTolerance:
      for i in range(len(contingencyMatrix)):
         for j in range(len(contingencyMatrix[i])):
            if contingencyMatrix[i][j] == 0:
               contingencyMatrix[:,j] += 1

   return ( rowDomain, colDomain, contingencyMatrix )

# P( A )
#Calculates the probability of any attribute value due its frequency with class label
def get_featureProbability( dataset, H, E, zeroFrequencyTolerance):
   if( isinstance( H, str ) ): H = data.get_variableFrom_str( dataset, H )
   if( isinstance( E, str ) ): E = data.get_variableFrom_str( dataset, E )
   if( not (H and E) ): return ( [], [], None )
   ( rowDomain, colDomain, cMatrix ) = get_contingencyMatrix( dataset, H, E, zeroFrequencyTolerance )
   len_rowDomain, len_colDomain = len( rowDomain ), len( colDomain )
   E_marginal = np.zeros( len_colDomain )
   for col in range(len_colDomain): E_marginal[col] = sum( cMatrix[:, col] )
   
   for row in range(len_rowDomain):
      for col in range(len_colDomain):
         if E_marginal[col] == 0:
            continue
         cMatrix[row, col] = cMatrix[row, col] / E_marginal[col]

   return ( rowDomain, colDomain, cMatrix )

# error matrix for a given feature and considering the datatset class
def get_errorMatrix( dataset, feature, zeroFrequencyTolerance ):
   if( isinstance( feature, str ) ): feature = data.get_variableFrom_str( dataset, feature )
   the_class = dataset.domain.class_var
   ( rowDomain, colDomain, pMatrix ) = get_featureProbability( dataset, the_class, feature, zeroFrequencyTolerance )
   if( not (rowDomain or colDomain) ): return ( [], [], None )

   errorMatrix = 1 - pMatrix
   return ( rowDomain, colDomain, errorMatrix )

#Calculates the entropy for each value of an attribute
def get_attributeEntropys(dataset, feature):
   entropys = []
   if( isinstance( feature, str ) ): feature = data.get_variableFrom_str( dataset, feature )
   the_class = dataset.domain.class_var
   ( rowDomain, colDomain, pMatrix ) = get_featureProbability( dataset, the_class, feature, False)
   for col in range(len(colDomain)): 
      entropy = 0
      for probability in pMatrix[:, col]:
         if probability == 0:
            continue
         entropy += -(probability * (math.log(probability,2)))
      entropys.append(entropy)
   return entropys

#Calculates the mean entropy of an attribute
def get_entropy(dataset, attribute):
   entropy = 0
   entropys = get_attributeEntropys(dataset,attribute)
   if( isinstance( attribute, str ) ): feature = data.get_variableFrom_str( dataset, attribute )
   the_class = dataset.domain.class_var
   ( rowDomain, colDomain, cMatrix ) = get_contingencyMatrix( dataset, the_class, feature, False )

   for col in range(len(colDomain)):
      entropy += (sum(cMatrix[:, col])/(sum(sum(cMatrix))) * entropys[col])
      
   return entropy

def get_tupleCountByFeature(dataset, attribute):
   count = []
   if( isinstance( attribute, str ) ): feature = data.get_variableFrom_str( dataset, attribute )
   the_class = dataset.domain.class_var
   ( rowDomain, colDomain, cMatrix ) = get_contingencyMatrix( dataset, the_class, feature, False )
   len_colDomain = len(colDomain)
   for col in range(len_colDomain): count.append(sum( cMatrix[:, col] ))
   return count