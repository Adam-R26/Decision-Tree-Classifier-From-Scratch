# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:12:02 2022

@author: adamr
"""
import pandas as pd
import math


class DecisionTreeClassifier:
    def id3(self, trainData:pd.DataFrame, label:str) -> dict:
        trainDataC = trainData.copy()
        tree = {} 
        classList = trainDataC[label].unique() 
        self._buildTree(tree, None, trainData, label, classList)
        return tree
    
    def predictSample(self, tree:dict, instance:pd.DataFrame) -> int:
        if not isinstance(tree, dict): #If we reach a lead node.
            return tree #Base Case of Recursion: Return prediction.
        else:
            rootNode = next(iter(tree)) #Fetch first key/feature
            featureValue = instance[rootNode] #value of the feature
            if featureValue in tree[rootNode]: #checking the feature value in current tree node
                return self.predictSample(tree[rootNode][featureValue], instance) #goto next feature
            else:
                return None
    
    def accuracyScore(self, tree:dict, testData:pd.DataFrame, label:str) -> float:
        correctPredictions = 0
        wrongPredictions = 0
        for index, row in testData.iterrows(): #for each row in the dataset
            result = self.predictSample(tree, testData.iloc[index]) #Make prediction for sample.
            if result == testData[label].iloc[index]: #If prediction == actual.
                correctPredictions += 1 #increase correct oberservation count
            else:
                wrongPredictions += 1 #Otherwise increase incorrect observation count
        accuracy = correctPredictions / (correctPredictions + wrongPredictions)
        return accuracy
    
    def _calculateEntropyRvGivenTarget(self, df:pd.DataFrame, column:str, target:str) -> float:
        '''Calculates the entropy of a random variable(R.V) given the target. Input:training data, name of the R.V, name of target column'''
        uniqueColumnValues = df[column].unique().tolist()
        dataSplits = {}
        entropy = 0
        #Split data set into sets containing each setting an R.V can take.
        for value in uniqueColumnValues:
            dataSplits[value] = df.loc[df[column]==value]
        
        #Calculate entropy of each outcome and combine to get entropy of feature.
        for key in dataSplits.keys():
            entropy += (len(dataSplits[key][target])/len(df[target]))*self._calculateEntropyRv(dataSplits[key], target)
    
        return entropy
        
    
    def _calculateEntropyRv(self, df:pd.DataFrame, column:str) -> float:
        #Get the values a random variable can take on and column values.
        uniqueColumnValues = df[column].unique().tolist()
        dataSplits = []
        probabilityRv = {}
        entropy = 0
        
        #Segregate dataset to find probability of each outcome
        for value in uniqueColumnValues:
            tmpVal  = df.loc[df[column]==value]
            dataSplits.append(tmpVal)
            probabilityRv[value] = (len(tmpVal[column])/len(df[column]))
        
        #Use probabilities to calculate entropy of RV
        for key in list(probabilityRv.keys()):
            tmpProb = probabilityRv[key]
            entropy += tmpProb*math.log(tmpProb, 2)
        
        return -entropy
        
    def _findBestSplit(self, df:pd.DataFrame, target:str) -> str:
        #Get the training data without the target.
        colsExcTarget = list(df.columns)
        colsExcTarget.remove(target)
        targetEntropy = self._calculateEntropyRv(df, target) #Calculate the entropy of target variable.
        igMap = {}
        
        #Compute information gain for each variable in the data.
        for col in colsExcTarget:
            colEntropyGivenTarget = self._calculateEntropyRvGivenTarget(df, col, target)
            ig = targetEntropy - colEntropyGivenTarget
            igMap[col] = ig
        
        #Get highest information gain feature and return it.
        mostInformativeFeature = max(igMap, key=igMap.get)
        return mostInformativeFeature
    
    
    def _buildSubTree(self, featureName:str, trainData:pd.DataFrame, label:str, classList:list) -> [dict, pd.DataFrame]:
        featureValueCountDict = trainData[featureName].value_counts(sort=False) #dictionary of the count of unqiue feature value
        tree = {} #sub tree or node
        
        for featureValue, count in featureValueCountDict.iteritems():
            featureValueData = trainData[trainData[featureName] == featureValue] #Split dataset into
            
            assignedToNode = False #Tracks whether feature setting is pure.
            for c in classList: #for each class
                classCount = featureValueData[featureValueData[label] == c].shape[0] #count of class c
    
                if classCount == count: #Check if target is pure yet.
                    tree[featureValue] = c #Add node to tree.
                    trainData = trainData[trainData[featureName] != featureValue] #Remove rows with given setting of R.V
                    assignedToNode = True
            if not assignedToNode: #If target not pure
                tree[featureValue] = "?" #Mark that we should split again to try to obtain purity.
                
        return tree, trainData
    
    def _buildTree(self, root:dict, prevFeatureValue:str, trainData:pd.DataFrame, label:str, classList:list):
        if trainData.shape[0] != 0: #if dataset becomes empty after updating
            maxInfoFeature = self._findBestSplit(trainData, label) #most informative feature
            tree, trainData = self._buildSubTree(maxInfoFeature, trainData, label, classList) #getting tree node and updated dataset
            nextRoot = None
            
            if prevFeatureValue != None: #add to intermediate node of the tree
                root[prevFeatureValue] = dict()
                root[prevFeatureValue][maxInfoFeature] = tree
                nextRoot = root[prevFeatureValue][maxInfoFeature]
            else: #add to root of the tree
                root[maxInfoFeature] = tree
                nextRoot = root[maxInfoFeature]
            
            for node, branch in list(nextRoot.items()): #iterating the tree node
                if branch == "?": #if it is expandable
                    featureValueData = trainData[trainData[maxInfoFeature] == node] #using the updated dataset
                    self._buildTree(nextRoot, node, featureValueData, label, classList) #recursive call with updated dataset




df = pd.DataFrame({'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
                   'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool','Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
                   'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
                   'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
                   'Played Football': [0,0,1,1,1,0,1,0,1,1,1,1,1,0]})

dfTest = pd.DataFrame({'Outlook': ['Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
                   'Temperature': ['Cool', 'Cool', 'Cool','Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
                   'Humidity': ['Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
                   'Wind': ['Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
                   'Played Football': [1,0,1,0,1,1,1,1,1,0]})


classifier = DecisionTreeClassifier()
tree = classifier.id3(df, 'Played Football')
predictionNeg = classifier.predictSample(tree, df.iloc[0])
predictionPos = classifier.predictSample(tree, df.iloc[11])
acc = classifier.accuracyScore(tree, dfTest, 'Played Football')
