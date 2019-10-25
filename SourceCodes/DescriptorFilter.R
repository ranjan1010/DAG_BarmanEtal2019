# R script to pre-process  Descriptor Data 

## library imported 

library(mlbench)
library(caret)
library(corrplot)

# data from the file 

# use to get input .csv file from user 
inputDataFile <- readline("please enter data file to be filter with label in extension (.csv): ");
dat = read.csv(inputDataFile, header = TRUE)

# initial column count 

print ("initial columns")
print (ncol(dat))

# zero std column were removed 
nd = Filter(sd, dat)


#Column count after removing zero value columns 

print ("After removing zero std value columns")
print  (ncol(nd))


# Data normalization (zero mean, unit variance)

x =  ncol(nd)
preObj <- preProcess(nd[,2:(x-1) ], method=c("center", "scale"))
normalized_Data <- predict(preObj, nd[,2:(x-1)])

# correlation matrix 

y = ncol(normalized_Data)
m = cor(normalized_Data[,2:(y-1)])

#removes Highly correlated columns  

hc = findCorrelation(m, cutoff=0.8)
new_hc = sort(hc)
new_dat = normalized_Data[,-c(new_hc)]

#column count, After removing the correlated columns . 

print ("after removing correlated columns")
print (ncol(new_dat))


# final data written to the .csv file. 
outputFile <- readline("please enter output file after filter in extension (.csv): ");
write.csv(new_dat, file = outputFile, row.names = FALSE)

