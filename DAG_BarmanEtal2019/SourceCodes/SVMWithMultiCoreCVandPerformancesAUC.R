# This functions helps us to remove previously created workspace variables....
rm(list=ls());

# This library use to handle ML related functions
library(e1071);
library(ggplot2);
library(caret);

# this library use to calculate AUC (Area under ROC)
library(ROCR);

# load dependencies libraries
library(lattice)
library(iterators)
library(parallel)

# This library use to handle multi-core processing for (for loop)
library(foreach);
library(doMC);
#use 10-cores processor
registerDoMC(10); 

# use to get input .csv file from user 
inputDataFile <- readline("please enter data file with label in extension (.csv): ");
dataFile <- read.csv(inputDataFile, header = TRUE);
dataFile

# use to get number of cross validation user want 
inputFoldCV <- readline("please enter the no. of fold cross-validation user want: ");
foldCV <- as.numeric(inputFoldCV);
foldCV

# No. of positive data in whole data set 
noOfPositive<-length(which(dataFile[,1] == "Positive"));
print(paste0("No of positive data in whole dataset is::",noOfPositive))

# No. of negative data in whole data set 
noOfNegative<-length(which(dataFile[,1] == "Negative"));
print(paste0("No of negative data in whole dataset is::",noOfNegative))


# This matrix use to store the performance Result 

# header initialization
headerforResultMatrix <- c("IterationForCV", "SVM_Cost_Parameter", "SVM_Gamma_Parameter",
                           "TP","FN","TN","FP",
                          "Sensitivity","Specificity","Accuracy",
                          "PPV","NPV","MCC","F1Score","AUC");

# Iterate for all the folds 
resultTempMatrix <- foreach(k=1 : foldCV,.combine = rbind) %dopar% {
  sizeOfWholeDataSet <- length(dataFile[,1]);
  
  # Size of test data set for each fold
  sizeOfTestFold <- floor(sizeOfWholeDataSet/foldCV);
  
  # Start index of positive test set based on the no of fold 
  startIndexOfPositiveTest =ceiling(((foldCV -k) * (sizeOfTestFold/2))) + 1;
  
  # End index of positive test based on the no of fold 
  if(sizeOfTestFold > 0)
  {
    endIndexOfPositiveTest = ceiling(((foldCV - (k -1)) * (sizeOfTestFold/2)));
  }else
  {
    print("please check the size of test fold !!!!!!!!!!!!!!!");
  }
  
  # Start index of negative test set based on the no of fold 
  startIndexOfNegativeTest = noOfPositive + startIndexOfPositiveTest;
  
  # End index of negative test based on the no of fold 
  endIndexOfNegativeTest = noOfPositive + endIndexOfPositiveTest;
  
  # Index of positive test set
  indexOfPositiveTest <- startIndexOfPositiveTest:endIndexOfPositiveTest;
  
  # Index of negative test set
  indexOfNegativeTest <- startIndexOfNegativeTest:endIndexOfNegativeTest;
  
  # Index of test set 
  indexOfTest <- c(indexOfPositiveTest,indexOfNegativeTest);
  
  # use to fetch test data based on the index for test 
  testData <- dataFile[indexOfTest,];
  
  # rest index are used for training 
  trainingData <- dataFile[-indexOfTest,];
  
  # This used to reproduce the result
  set.seed(1234)
  #Tune SVM parameter with Cost and gama
  SVMParameterTune <- tune(svm, Indicator~.,data = trainingData, ranges = list( cost = 2^(1:10), gamma = 2^(-5:10)));
  
  # Select best model based on the parameters
  BestModelTrain<- SVMParameterTune$best.model
  summary(BestModelTrain)
  
  # prediction test data 
  predTest <- predict(BestModelTrain, testData)
  tabTest<- table(Predicted = predTest, Actual = testData$Indicator)
  tabTest
  
  # This section is included for calculate Area Under ROC curve 
  
  # SVM model again generated based on best tune model parameter for probability 
  modelForAUC <- svm(Indicator~.,data = trainingData, cost = BestModelTrain$cost,  
                        gamma = BestModelTrain$gamma, cross = 10, probability = TRUE);
  # this section will predict the test data also able to give the probability score and decision value (prediction score) 
  predScoreAndProbabilityForTestData <- predict(modelForAUC, testData, probability = TRUE, decision.values = TRUE);
  
  # used to extract the decision value (prediction score) 
  extractPredScore <- attr(predScoreAndProbabilityForTestData,"decision.values");
  
  # used to predict based on ROCR prediction function
  predScore <- prediction(extractPredScore,testData$Indicator)
  
  # used to finally calculate AUC (Area under AUC curve)
  AUC <- performance(predScore,"auc")@y.values[[1]];
  
  # Initialized tp,tn,fp,fn 
  tp = tn = fp = fn = 0;
  
  # Assign the confusion matrix value for the corresponding measures
  tp = tabTest[2,2];
  fn = tabTest[1,2];
  tn = tabTest[1,1];
  fp = tabTest[2,1];
  
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
  
  # to check MCC denominator 
  denMCC = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  
  # MCC (Mathews Correlation coefficient) calculation
  if(denMCC > 0)
  {
    MCC = (((tp * tn) - (fp * fn))/(sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))));
  }else
  {
    MCC = 0;
  }
  
  # to check F1-Score denominator 
  denF1Score = Sensitivity + PPV;
  
  # F1-Score calculation
  if(denF1Score > 0)
  {
    f1Score = 2 * ((Sensitivity * PPV)/(Sensitivity + PPV));
  }else
  {
    f1Score = 0;
  }
  
  # Assign value to the temp vector
  tempVector<- c(k, BestModelTrain$cost, BestModelTrain$gamma, tp, fn, tn, fp, Sensitivity, Specificity, 
                 Accuracy, PPV, NPV, MCC, f1Score, AUC);
  
  #print(paste0("Processing completed for fold ::",k))
  # return performance result for each iteration 
  return(tempVector)
}
resultTempMatrix

#This is use to add header to the final matrix 
finalTempMatrix <- rbind(headerforResultMatrix,resultTempMatrix)
finalTempMatrix

outputFile <- paste0("PerformanceOnSVMIncAUC",inputDataFile);
write.table(finalTempMatrix,file = outputFile, sep=",",quote = FALSE, col.names=FALSE, row.names = FALSE)
