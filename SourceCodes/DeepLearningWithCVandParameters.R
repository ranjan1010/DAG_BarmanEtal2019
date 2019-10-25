# This functions helps us to remove previously created workspace variables....
rm(list=ls());

# This library use to handle deep learning (MLT) related functions
library(h2o);

# use to get input .csv file from user 
inputDataFile <- readline("please enter data file with label in extension (.csv): ");
dataFile <- read.csv(inputDataFile, header = TRUE);
dataFile

# End index of input file is no of columns
EndIndex <- length(dataFile[1,]);


h2o.init(nthreads = -1, ip="172.16.16.68");  

# Convert data into h2o data-format
h2oDataFormat <- as.h2o(dataFile);
h2oDataFormat

# use to set hidden layers for deep learning 

hiddenParameter <- list(hidden = c(50,50), hidden = c(100,100), hidden = c(150,150), hidden = c(200,200),
                        hidden = c(250,250), hidden = c(300,300),hidden = c(350,350), hidden = c(400,400),
                        hidden = c(450,450), hidden = c(500,500), hidden = c(50,50,50), hidden = c(100,100,100),
                        hidden = c(150,150,150), hidden = c(200,200,200), hidden = c(250,250,250), 
                        hidden = c(300,300,300),hidden = c(350,350,350), hidden = c(400,400,400),
                        hidden = c(450,450,450), hidden = c(500,500,500));

#use to store parameters value and their corresponding performances
tempMatrix <- matrix(nrow = length(hiddenParameter) + 1, ncol = 20);

# assign the header value for performance measure
tempMatrix[1,1] = "Hidden_Layers"
tempMatrix[1,2] = "accuracy_mean";
tempMatrix[1,3] = "auc_mean";
tempMatrix[1,4] = "err_mean";
tempMatrix[1,5] = "err_count_mean";
tempMatrix[1,6] = "f0point5_mean";
tempMatrix[1,7] = "f1_mean";
tempMatrix[1,8] = "f2_mean";
tempMatrix[1,9] = "lift_top_group_mean";
tempMatrix[1,10] = "logloss_mean";
tempMatrix[1,11] = "max_per_class_error_mean";
tempMatrix[1,12] = "mcc_mean";
tempMatrix[1,13] = "mean_per_class_accuracy_mean";
tempMatrix[1,14] = "mean_per_class_error";
tempMatrix[1,15] = "mse_mean";
tempMatrix[1,16] = "precision_mean";
tempMatrix[1,17] = "r2_mean";
tempMatrix[1,18] = "recall_mean";
tempMatrix[1,19] = "rmse_mean";
tempMatrix[1,20] = "specificity_mean";

# use to indicate the current row number for temp matrix
tempCount = 2;

# Iterate for all hidden parameters
for(i in 1 : length(hiddenParameter))
{
  # run the deep learning 
  modelDeepLearningWithCVandParameter <- h2o.deeplearning(x = colnames(h2oDataFormat[2:EndIndex]),y = "Indicator",training_frame = h2oDataFormat, nfolds = 10, hidden = hiddenParameter[i]$hidden, reproducible = TRUE, seed = 1234);
    
  # use to store hidden layer from model parameters
  tempMatrix[tempCount,1] = gsub(',','_',toString(modelDeepLearningWithCVandParameter@allparameters$hidden));
  
  #tempMatrix[tempCount,2] = toString(modelDeepLearningWithCVandParameter@allparameters$epochs);
  
  # iterate for all (19) performance measures likes accuracy, auc, recall, specificity etc ..
  for(k in 1 : length(modelDeepLearningWithCVandParameter@model$cross_validation_metrics_summary$mean))
  {
    tempColumnIndex = k + 1; 
    tempMatrix[tempCount, tempColumnIndex] = toString(modelDeepLearningWithCVandParameter@model$cross_validation_metrics_summary$mean[k]);
  }
  tempCount = tempCount + 1;
}
tempMatrix
outputFile <- paste0("PerformanceOn",inputDataFile);
write.table(tempMatrix, file = outputFile, sep = ",", quote =FALSE, row.names = FALSE, col.names = FALSE);

h2o.shutdown(prompt = FALSE)