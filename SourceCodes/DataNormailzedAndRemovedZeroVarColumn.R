# This functions helps us to remove previously created workspace variables....
rm(list=ls());

# reproduce the result
set.seed(123)

## library imported 

library(mlbench)
library(caret)
library(corrplot)

# data from the file 

# use to get input .csv file from user 
inputDataFile <- readline("please enter data file to be filter with label in extension (.csv): ");
data = read.csv(inputDataFile, header = TRUE)

# initial column count 

print ("initial columns")
print (ncol(data) - 1)

# zero std column were removed 
nd = Filter(sd, data)

#Column count after removing zero value columns 

print ("After removing zero std value columns")
print  (ncol(nd) - 1)


# Data normalization (zero mean, unit variance)

x =  ncol(nd)
preObj <- preProcess(nd[,2:x ], method=c("center", "scale"))
normalized_Data <- predict(preObj, nd[,2:x])

new_data = normalized_Data;


print ("after removing zero variance columns")
print (ncol(new_data))

# Again insert first column 
FinalMatrix = cbind(data[,1],new_data);

# Assign names of first columns 
names(FinalMatrix)[1] = names(data[1]);

# final data written to the .csv file. 
outputFile <- paste0("NormalizedAndRemovedZeroVar", inputDataFile);
write.csv(FinalMatrix, file = outputFile, row.names = FALSE)

