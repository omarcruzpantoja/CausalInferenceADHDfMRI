import numpy as np
from sklearn.svm import SVC , LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from skimage import io
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy import stats
from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

import math
np.set_printoptions(suppress=True)


def loadDataset(path, delim):
    """
        Function to load the dataset
        Input:
            path- Path of the file containing the dataset
            delim- Delimiter of the files
        Output:
            A vector of size [n_subjects, n_features] containing the features for each subject
    """
    
    controlLabels =[]
    controlArray = []
    expLabels = []
    expArray = []
    folder = os.listdir(path)
    for file in folder:

        f = open(path+file, "r") 
        subject = (file.split("_")[0])
        lines = f.readlines()
        array = []
        
        #Create a 1d array containing all the features for the subject
        for line in lines:
            if delim != "": 
                for each in line.split(delim):
                    array.append(float(each))
            else:
                array.append(float(line))

        #Separate between control and experimental group (TO BE ADAPTED CORRESPONDING TO DATASET FILE NAMING)
        if file[0] == "1":
            controlLabels.append(0)
            controlArray.append(array)
        else:
            expLabels.append(1)
            expArray.append(array)

    #Order the vector in shape all experimental and then all control group to facilitate training selections 
    for i in controlArray:
        expArray.append(i)
    labels = np.append( expLabels, controlLabels)

    return  np.array(expArray), labels


def leaveOneOut(datasetRaw, labels, classifier):
    """
        Leave one out method function. If the classifier has a weight of its features, will determine which had the highest
        impact.
        Input:
            dataset- Includes the vector containing the subjects and its features [n_subjects, n_features]
            labels- vector that contains the corresponding label of each subject [n_subjects]
            classifier- name of the classifier desired to be used
        Output:
            accuracy- accuracy obtained by the classifier
            *features- if the classifier has feature weighting, will return the index of the features in strength order
    """
    
    size = datasetRaw.shape[0]
    sumAccuracy = 0 

    #Matrix that will contain feature weights and their order
    

    numadhd = 40
    for idx in range(size):
        
        adhdGroup = datasetRaw[0:numadhd]
        controlGroup = datasetRaw[numadhd:]
        if idx < numadhd:
            concatgroup = np.concatenate([np.arange(0,idx), np.arange(idx+1,numadhd)])
            adhdGroup = datasetRaw[concatgroup]

        if idx >= numadhd:
            concatgroup = np.concatenate([np.arange(numadhd,idx), np.arange(idx+1,size)])
            controlGroup = datasetRaw[concatgroup]

        # print(adhdGroup.shape, controlGroup.shape)
        Idx = tScore(adhdGroup, controlGroup, size)
        dataset = getFeatures(datasetRaw, Idx)


        #Set up the leave-out-one by training all but one subject
        train = np.concatenate([np.arange(0,idx), np.arange(idx+1,dataset.shape[0])])
        test = np.arange(idx, idx+1)
        weights = np.zeros(int((dataset.shape[1])))
        
        if classifier == "linear": 
            clf = SVC(kernel= "linear")
            clf.fit(dataset[train], labels[train])
            #Use feature ranking to accumulate strength for each feature
            for ctr, var in enumerate(np.argsort(np.abs(clf.coef_))[0]):
                weights[var] += ctr

        elif classifier == "RF":
            clf = RandomForestClassifier()

            clf.fit(dataset[train], labels[train])

            for ctr, var in enumerate(np.argsort(clf.feature_importances_)):
            #Use feature ranking to accumulate strength for each feature                
                weights[var] += ctr

        elif classifier == "NB":
            clf = GaussianNB()
            clf.fit(dataset[train], labels[train])
            #Use mean square difference to identify strength
            for ctr, var in enumerate(np.argsort((clf.theta_[1] - clf.theta_[0])**2)):
                weights[var] += ctr

        elif classifier == "discriminant":
            clf = LinearDiscriminantAnalysis()
            clf.fit(dataset[train], labels[train])

        elif classifier == "NN":
            clf = MLPClassifier()
            clf.fit(dataset[train], labels[train])

        elif classifier == "regression":
            clf = LogisticRegression()
            clf.fit(dataset[train],labels[train])

        prediction2 = clf.predict(dataset) 
        correct2 = sum(prediction2[test]==labels[test]) # But compute perf on test only

        
        #Get length of the set
        tot2 = len(test)

        #Ignore this, old implementation 
        #res.append(prediction2[test])
        #resL.append(labels[test])
        #

        #Calculate accuracy of the classifier (for the current model) 
        accuracy2=correct2/tot2

        #Fix prediction for regression as the prediction will return a continuous result rather than discrete
        if classifier == "regression": 
            if prediction2[test] > .5 and labels[test] == 1:
                accuracy2 += 1
            elif prediction2[test] <= .5 and labels[test] == 0 :
                accuracy2 += 1
 

        #Accuracy accumulator
        sumAccuracy += accuracy2

    if classifier!= "discriminant" and classifier != "NN" and classifier != "regression": 
        connectionWeights = np.argsort(weights)[::-1]
        return  sumAccuracy/float(size), connectionWeights

        
    return sumAccuracy/float(size) 





def storeEdges(indexes, numVariables, message, fil, identifier):
    """
        Function to store the edges based on their strength
        Input: 
            indexes- Feature identifier
            numVariables- amount of variable(not features)
            message- message to be written to file before showing edges
            fil - path of file to be modified
            identifier- string used to identify the edges (to make it easier to parse)

    """
    f= open(fil, "a") 
    f.write(message + "\n")

    for variable in indexes:
        f.write("\n" + identifier + " "  + str((variable)%numVariables+1)+  " --> " + str(math.floor((variable)/numVariables)+1))
        # print( str(variable%numVariables) , str(math.floor(variable/numVariables)))
    f.close()


def tScore(cGroup, eGroup, size, pvalue=.05):
    """
    Function to calculate the t-score for each feature in the dataset, uses comparison of means. The t-test will 
    determine which features in the dataset have a significant difference in means

    Input:
        cGroup- Control group data [Shape: n_subjects, n_features]
        eGroup- Experimental group data [Shape: n_subjects, n_features
        size- sample size (both control and experimental group)
        pvalue- value where to reject the null hypothesis
    Output: 
        Array containing the index of the features in the dataset with pvalue less than the input given. 
    """
    
    meancGroup = np.mean(cGroup, axis = 0 )
    meaneGroup = np.mean(eGroup, axis = 0 )

    varcGroup = np.var(cGroup, axis = 0 )
    vareGroup = np.var(eGroup, axis = 0 )

    numcGroup = cGroup.shape[0]
    numeGroup = eGroup.shape[0]

    Vjk = ((numcGroup-1) * varcGroup + (numeGroup-1)*vareGroup) /  float(numeGroup+numcGroup-2)

    tScore = (meancGroup - meaneGroup) / np.sqrt((Vjk) * (float(1)/numcGroup + float(2)/numeGroup))

    return np.array(np.where(stats.t.cdf(tScore,size-2) < pvalue)[0])

def getFeatures(data, idx):
    """
    Function to return an array containing the values of the features found in the tScore Function
    Input:
        data- complete dataset, including both experimental and control group and their features [Shape: n_subjects, n_features]
        idx-  Indeces of the features with significant difference
    Output:
        A vector of size [n_subjects, len(idx)] containing the information of the extracted features. 
    """
    dataFeatures = np.zeros((data.shape[0], idx.shape[0]))
    
    for sub in range(data.shape[0]):
        for feature in range(idx.shape[0]):
            dataFeatures[sub][feature] = data[sub][idx[feature]]

    return dataFeatures



# def classify(dataset, labels, classifier):
#     sumAccuracy = 0
#     size = labels.shape[0]
#     res = []
#     resL = []
#     # for idx in range(size):
#     train = np.concatenate([np.arange(0,35), np.arange(40,75)])
#     test = np.concatenate([np.arange(35,40), np.arange(75,80)])


#     if classifier == "linear": 
#         clf = SVC(kernel= "linear")

#     elif classifier == "RF":
#         clf = RandomForestClassifier()

#     elif classifier == "NB":
#         clf = GaussianNB()

#     clf.fit(dataset[train], labels[train])
#     prediction2 = clf.predict(dataset) 

#     correct2 = sum(prediction2[test]==labels[test]) # But compute perf on test only

#     #print(idx, "Prediction: ", prediction2[test], "Correct Value: ", labels[test])
    
#     #Get length of the set
#     tot2 = len(test)
#     # res.append(prediction2[test])
#     # resL.append(labels[test])
#     #Calculate accuracy of the classifier
#     accuracy2=correct2/tot2

# #       print('The accuracy of the test sample is: {:.3} (correct={}/tot={})'.format(accuracy2, correct2,tot2))

#     cMat = confusion_matrix(labels[test], prediction2[test])
#     sumAccuracy += accuracy2

 
#     return res, resL, sumAccuracy/float(size)

# def seq(init, end, length):

#     store = end - init

#     interval = store/float(length-1)

#     vec = []
#     for i in range(length):
#         vec.append(i*interval+init)
#     return vec