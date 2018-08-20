from misc import *

if __name__ == "__main__":

	resting = open("restResults.txt", "w") 
	bart = open("bartResults.txt", "w")

	print("Load raw data")
	rawRData, rawRLabel = loadDataset("../REST/RawData/", " ")
	rawBData, rawBLabel = loadDataset("../BART/RawData/", " ")


	print("Load resting state data")

	faskRData, faskRLabels  = loadDataset("../REST/Fask/", "\t" )

	twoRData, twoRLabels = loadDataset("../REST/Two-Step/", "\t" )


	print("Load task related data") 

	faskBData, faskBLabels = loadDataset("../BART/Fask/", "\t" )

	twoBData, twoBLabels = loadDataset("../BART/Two-Step/", "\t" )

	print("Datasets loaded")
	



	# print("Start feature selection")
	# fRIdx = tScore(faskRData[0:40], faskRData[40:], faskRData.shape[0])
	# faskRData = getFeatures(faskRData, fRIdx)
    
	# tRIdx = tScore(twoRData[0:40], twoRData[40:], twoRData.shape[0])
	# twoRData = getFeatures(twoRData, tRIdx)


	# fBIdx = tScore(faskBData[0:40], faskBData[40:], faskBData.shape[0])
	# faskBData = getFeatures(faskBData, fBIdx)

	# tBIdx = tScore(twoBData[0:40], twoBData[40:], twoBData.shape[0])
	# twoBData = getFeatures(twoBData, tBIdx)

	resting.write("Final resting state data report\n")
	# resting.write("\nNumber of selected FASK variables: " + str(len(fRIdx)))
	# resting.write("\nNumber of selected 2Step variables: "+ str(len(tRIdx)))

	bart.write("Final bart task data report\n")
	# bart.write("\nNumber of selected FASK variables: " + str(len(fBIdx)))
	# bart.write("\nNumber of selected 2Step variables: "+ str(len(tBIdx)))
	print("Feature selection Completed")


	resting.write("\n\nLogistic regression")
	regRRData = leaveOneOut(rawRData, rawRLabel, "regression")  
	resting.write( "\nLogistic regression with resting state raw data accuracy:" + str(regRRData) )

	print("Start classification resting data") 
	resting.write( "\n\nSupport Vector Machine Linear Classifier") 

	# resting.write( "\nRFask")
	svcRRFask, svcRFConnections = leaveOneOut(faskRData, faskRLabels, "linear") 
	resting.write( "\nResting state featured variables SVC fask accuracy:" + str(svcRRFask))

	# resting.write( "\nRTRwoStep") 
	svcLR2Step, svcRTConnections = leaveOneOut(twoRData, twoRLabels, "linear") 
	resting.write( "\nResting state with featured variables SVC 2step accuracy:" + str(svcLR2Step))


	resting.write( "\n\nRandom Forest Classifier") 

	# resting.write( "\nRFask")
	rfRRFask, rfRRFConnections = leaveOneOut(faskRData, faskRLabels, "RF") 
	resting.write( "\nResting state with featured variables RF fask accuracy:" + str(rfRRFask))

	# resting.write( "\nTwoStep") 
	rfLR2Step, rfRRTConnections = leaveOneOut(twoRData, twoRLabels, "RF") 
	resting.write( "\nResting state with featured variables RF 2step accuracy:" + str(rfLR2Step))

	resting.write( "\n\nNaive Bayes Classifier") 

	# resting.write( "\nRFask")
	nbRRFask, nbRFConnections= leaveOneOut(faskRData, faskRLabels, "NB") 
	resting.write( "\nResting state with featured variables NB fask accuracy:" + str(nbRRFask))

	# resting.write( "\nTwoStep") 
	nbLR2Step, nbRTConnections = leaveOneOut(twoRData, twoRLabels, "NB") 
	resting.write( "\nResting state with featured variables NB 2step accuracy:" + str(nbLR2Step))

	resting.write("\n\nLinear discriminant classifier") 

	# resting.write( "\nRFask")
	dRRFask= leaveOneOut(faskRData, faskRLabels, "discriminant") 
	resting.write( "\nResting state with featured variables linear discriminant fask accuracy:" + str(dRRFask))

	# resting.write( "\nTwoStep") 
	dLR2Step = leaveOneOut(twoRData, twoRLabels, "discriminant") 
	resting.write( "\nResting state with featured variables linear discriminant 2step accuracy:" + str(dLR2Step))

	resting.write("\n\nNeural Network")
	nnRRFask= leaveOneOut(faskRData, faskRLabels, "NN") 
	resting.write( "\nResting state with featured variables neural network fask accuracy:" + str(nnRRFask))

	# resting.write( "\nTwoStep") 
	nnLR2Step = leaveOneOut(twoRData, twoRLabels, "NN") 
	resting.write( "\nResting state with featured variables neural network 2step accuracy:" + str(nnLR2Step))
	print("Start classification bart data") 


	bart.write("\n\nLogistic regression") 

	regRBData = leaveOneOut(rawBData, rawBLabel, "regression")  
	bart.write( "\nLogistic regression with task related (BART) raw data accuracy::" + str(regRBData))

	bart.write( "\n\nSupport Vector Machine Linear Classifier") 

	# bart.write( "\nBFask")
	svcRBFask, svcBFConnections = leaveOneOut(faskBData, faskBLabels, "linear") 
	bart.write( "\nBART task with featured variables SVC fask accuracy:" + str(svcRBFask))

	# bart.write( "\nRTRwoStep") 
	svcLB2Step, svcBTConnections = leaveOneOut(twoBData, twoBLabels, "linear") 
	bart.write( "\nBART task with featured variables SVC 2step accuracy:" + str(svcLB2Step))


	bart.write( "\n\nRandom Forest Classifier") 

	# bart.write( "\nBFask")
	rfRBFask, rfRBFConnections = leaveOneOut(faskBData, faskBLabels, "RF") 
	bart.write( "\nBART task with featured variables RF fask accuracy:" + str(rfRBFask))

	# bart.write( "\nTwoStep") 
	rfLB2Step, rfRBTConnections = leaveOneOut(twoBData, twoBLabels, "RF") 
	bart.write( "\nBART task with featured variables RF 2step accuracy:" + str(rfLB2Step))

	bart.write( "\n\nNaive Bayes Classifier") 

	# bart.write( "\nBFask")
	nbRBFask, nbBFConnections = leaveOneOut(faskBData, faskBLabels, "NB") 
	bart.write( "\nBART task with featured variables NB fask accuracy:" + str(nbRBFask))

	# # bart.write( "\nTwoStep") 
	nbLB2Step, nbBTConnections = leaveOneOut(twoBData, twoBLabels, "NB") 
	bart.write( "\nBART task with featured variables NB 2step accuracy:" + str(nbLB2Step))

	bart.write("\n\nLinear discriminant classifier") 
	dRBFask = leaveOneOut(faskBData, faskBLabels, "discriminant") 
	bart.write( "\nBART task with featured variables linear discriminant fask accuracy:" + str(dRBFask))

	# # bart.write( "\nTwoStep") 
	dLB2Step = leaveOneOut(twoBData, twoBLabels, "discriminant") 
	bart.write( "\nBART task with featured variables linear discriminant 2step accuracy:" + str(dLB2Step))

	bart.write("\n\nNeural Network")
	nnRBFask = leaveOneOut(faskBData, faskBLabels, "NN") 
	bart.write( "\nBART task with featured variables neural network fask accuracy:" + str(nnRBFask))

	# bart.write( "\nTwoStep") 
	nnLB2Step = leaveOneOut(twoBData, twoBLabels, "NN") 
	bart.write( "\nBART task with featured variables neural network 2step accuracy:" + str(nnLB2Step))

	print("Classification completed")

	print("Store edge information") 

	resting.close()
	bart.close()


	#Store edge information in file
	#Edge information has will be stored in order from the highest weight to lowest
	# storeEdges(fRIdx[svcRFConnections], int(math.sqrt(faskRData.shape[1])), "\n\nResting state SVC fask edges", "restResults.txt", "svcRSFask" )
	# storeEdges(tRIdx[svcRTConnections], int(math.sqrt(twoRData.shape[1])), "\n\nResting state SVC 2Step edges", "restResults.txt", "svcRS2Step" )
	

	# storeEdges(fBIdx[svcBFConnections], int(math.sqrt(faskBData.shape[1])), "\n\nBart task SVC fask edges", "bartResults.txt", "svcBTFask" )
	# storeEdges(tBIdx[svcBTConnections], int(math.sqrt(twoBData.shape[1])), "\n\nBart task SVC 2Step edges", "bartResults.txt", "svcBT2Step" )

	# storeEdges(fRIdx[rfRRFConnections], int(math.sqrt(faskRData.shape[1])), "\n\nResting state RF fask edges", "restResults.txt", "rfRSFask" )
	# storeEdges(tRIdx[rfRRTConnections], int(math.sqrt(twoRData.shape[1])), "\n\nResting state RF 2Step edges", "restResults.txt", "rfRS2Step" )

	# storeEdges(fRIdx[nbRFConnections], int(math.sqrt(faskRData.shape[1])), "\n\nResting state NB fask edges", "restResults.txt", "nbRSFask" )


	# storeEdges(fBIdx[rfRBFConnections], int(math.sqrt(faskBData.shape[1])), "\n\nBart task RF fask edges", "bartResults.txt", "rfBTFask" )
	# storeEdges(tBIdx[rfRBTConnections], int(math.sqrt(twoBData.shape[1])), "\n\nBart task RF 2Step edges", "bartResults.txt", "rfBT2Step" )

	# storeEdges(fBIdx[nbBFConnections], int(math.sqrt(faskRData.shape[1])), "\n\nBart task NB fask edges", "bartResults.txt", "nbBSFask" )


	print("Completed")


