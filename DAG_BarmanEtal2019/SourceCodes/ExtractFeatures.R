# use to handle string related function
library("stringr")
# use to get input .csv file from user 
inputDataFile <- readline("please enter data file to be extract for feature slection with label in extension (.csv): ");
dataFile = read.csv(inputDataFile, header = TRUE);

inputSelectedFeatureFile <- readline("please enter the selected features file name in extension (.csv): ");
selectedFeaturesFile = read.csv(inputSelectedFeatureFile, header = TRUE)
# initial column count 

print ("initial columns in original data file::")
print (ncol(dataFile) - 1)

print ("No of selected features::")
print (nrow(selectedFeaturesFile))

#noOfSelectedFeatures =nrow(selectedFeaturesFile);


# This section will find the index of selected feature name 
#selectedFeaturesColumnName = which(names(dataFile) %in% selectedFeaturesFile[,1]); // This section works properly in past
selectedFeaturesColumnName = which(names(dataFile) %in% str_trim(selectedFeaturesFile[,1]))
selectedFeaturesColumnName

# This section included first column since it was indicator (Class Label) column
finalCoumnIndexes = c(1, selectedFeaturesColumnName)

new_data = dataFile[,c(finalCoumnIndexes)];



print ("after removing features that are not selected by EFS. Then the no of columns::")
print (ncol(new_data))



# final data written to the .csv file. 
outputFile <- paste0("selectedFeatures", inputDataFile);
write.csv(new_data, file = outputFile, row.names = FALSE)

