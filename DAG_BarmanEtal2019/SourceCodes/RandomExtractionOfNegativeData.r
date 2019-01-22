# this functions helps us remove previously created workspace variables....
rm(list=ls());

# this library used to handle string related functions.......
library("stringr");

# use to get protein list not present in DisGeNET .csv file from user 
inputListOfProteinsNotPresentInDisGeNET<- readline("please enter the protein list not present in DisGeNET file in .csv format: ");
listOfProteinsNotsPresentInDisGeNET <- read.csv(inputListOfProteinsNotPresentInDisGeNET, header = TRUE);

# use to get number of random number user want
inputNoOfRandom <-readline("please enter the number of random number you want: ");
noOfRandom <- as.numeric(inputNoOfRandom);

# use to get minimum random number user want 
inputMinimumNumberForRandom <-readline("please enter minimum random number you want: ");
minimumNumberForRandom <- as.numeric(inputMinimumNumberForRandom);

# use to get maximum random number user want
inputMaximumNumberForRandom <-readline("please enter maximum random number you want: ");
maximumNumberForRandom <- as.numeric(inputMaximumNumberForRandom);

# This below line generate random number without repetition
listofRandomNumbers <- sample(minimumNumberForRandom:maximumNumberForRandom,noOfRandom,replace = FALSE);

# This matrix used to temporary store the UniprotID based on random selection.
tempMatrix<- matrix(0, nrow = noOfRandom + 1 );

tempMatrix[1] <- toString("UniprotID");

for (i in 1 : noOfRandom)
{
  tempMatrix[i+1] = toString(listOfProteinsNotsPresentInDisGeNET[listofRandomNumbers[i],]);
}
tempMatrix
ouputFileName <- readline("Please enter the output file name in .csv format: ");
write.table(as.data.frame(tempMatrix),file = ouputFileName, sep=",", quote = FALSE, row.names = FALSE,col.names=FALSE);
