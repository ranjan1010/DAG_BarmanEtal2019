# This functions helps us to remove previously created workspace variables....
rm(list=ls());

# This library use to handle deep learning (MLT) related functions
library(h2o);

# use to get input training data in .csv file from user 
inputTrainingDataFile <- readline("please enter the training data file with label in extension (.csv): ");
trainingDataFile <- read.csv(inputTrainingDataFile, header = TRUE);
trainingDataFile

# End index of input file is no of columns
EndIndex1 <- length(trainingDataFile[1,]);

# use to get input independent data in .csv file from user 
inputIndependentDataFile <- readline("please enter the independent data file with label in extension (.csv): ");
independentDataFile <- read.csv(inputIndependentDataFile, header = TRUE);
independentDataFile

# End index of input file is no of columns
EndIndex2 <- length(independentDataFile[1,]);

h2o.init(nthreads = -1);

# Convert training data into h2o data-format
h2oTrainingDataFormat <- as.h2o(trainingDataFile);
h2oTrainingDataFormat

# Convert independent data into h2o data-format
h2oIndependentDataFile <- as.h2o(independentDataFile);
h2oIndependentDataFile

# Building model for training data in deep learning
modelDeepLearningWithCV <- h2o.deeplearning(x = colnames(h2oTrainingDataFormat[2:EndIndex1]),y = "Indicator",training_frame = h2oTrainingDataFormat, nfolds = 10, hidden = c(500,500), reproducible = TRUE, seed = 1234);

print(summary(modelDeepLearningWithCV));

# This is used to predict unknown instance not used in training and testing 
predictIndependentData = h2o.predict(object = modelDeepLearningWithCV, newdata = h2oIndependentDataFile[2:EndIndex2]);

# To genereate AUC score for Independent dataset
AUC <- h2o.auc(h2o.performance(model = modelDeepLearningWithCV, newdata = h2oIndependentDataFile[1:EndIndex2]));

print(summary(predictIndependentData));

outputFile <- readline("please enter output file name for prediction result for independent data set .csv file extension: ");

write.table(as.data.frame(predictIndependentData), file = outputFile, sep = ",", quote =FALSE, row.names = FALSE, col.names = TRUE);

# ---------------------this section will calculate performance measure -----------------------------------

# this section will read the prediction result matrix

predictionResult = read.csv(file = outputFile, header = TRUE);
predictionResult
# Initialized tp, tn, fp, fn 
tp = tn = fp = fn = 0;

# initialization the temporary matrix
# this matrix use to temporary store the sen, spec, Acc  etc
tempMatrix <- matrix(0, nrow = 2 , ncol = 8);


# header initialization
tempMatrix[1,1] = "Sensitivity";
tempMatrix[1,2] = "Specificity";
tempMatrix[1,3] = "Accuracy";
tempMatrix[1,4] = "PPV";
tempMatrix[1,5] = "NPV";
tempMatrix[1,6] = "MCC";
tempMatrix[1,7] = "F1Score";
tempMatrix[1,8] = "AUC";


for(i in 1 : length(predictionResult[,1]))
{
  if((predictionResult[i,1] == "Positive") & (independentDataFile[i,1] == "Positive"))
  {
    tp = tp + 1;
  }
  if((predictionResult[i,1] == "Negative") & (independentDataFile[i,1] == "Negative"))
  {
    tn = tn + 1 ;
  }
  if((predictionResult[i,1] == "Positive") & (independentDataFile[i,1] == "Negative"))
  {
    fp = fp + 1 ;
  }
  if((predictionResult[i,1] == "Negative") & (independentDataFile[i,1] == "Positive"))
  {
    fn = fn + 1 ;
  }
}

# Sensitivity calculation 
if((tp + fn) > 0)
{
  Sensitivity = (tp / (tp + fn)) * 100;
}else
{
  Sensitivity = 0;
}
# Specificity calculation 
if((tn + fp) > 0)
{
  Specificity = (tn / (tn + fp)) * 100 ;
}else
{
  Specificity = 0;
}
# Accuracy calculation
if((tp + fn + tn + fp) > 0)
{
  Accuracy = ((tp + tn) / (tp + fn + tn + fp)) * 100;
}else
{
  Accuracy = 0;
}
# PPV (positive prediction value) calculation
if((tp + fp) > 0)
{
  PPV = (tp / (tp + fp)) * 100; 
}else
{
  PPV = 0;
}
# NPV (negative prediction value) calculation
if((tn + fn) > 0)
{
  NPV = (tn / (tn + fn)) * 100; 
}else
{
  NPV = 0;
}
# MCC (Mathews Correlation coefficient) calculation
if((tp * tn) > (fp * fn))
{
  MCC = (((tp * tn) - (fp * fn))/(sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))));
}else
{
  MCC = 0;
}


tempMatrix[2,1] = Sensitivity; 
tempMatrix[2,2] = Specificity;  
tempMatrix[2,3] = Accuracy;
tempMatrix[2,4] = PPV;
tempMatrix[2,5] = NPV; 
tempMatrix[2,6] = MCC;
tempMatrix[2,7] = 2 *((Sensitivity * PPV)/(Sensitivity + PPV)); 
tempMatrix[2,8] = AUC;

outputFile1 <- readline("please enter the output file name for performance .csv format : ");
write.table(tempMatrix,file = outputFile1, sep=",",quote = FALSE, col.names=FALSE, row.names = FALSE);

h2o.shutdown(prompt = FALSE);