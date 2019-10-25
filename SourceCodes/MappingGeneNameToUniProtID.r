# this functions helps us remove previously created workspace variables....
rm(list=ls());

# this library used to handle string related functions.......
library("stringr");

# use to get input gene name with GeneID in .csv file from user 
inputGeneNameFile <- readline("please enter the input gene name with GeneID file in .csv format: ");
geneNameFile <- read.csv(inputGeneNameFile, header = TRUE);


# use to get input UniProtID with GeneID in .csv file from user 
inputUniProtIDFile <- readline("please enter the input UniProtID with GeneID file in .csv format: ");
uniProtIDFile <- read.csv(inputUniProtIDFile, header = TRUE);

# this matrix use to store all gene name related UniProtID 
tempMatrix <- matrix(0, nrow = length(geneNameFile[,1]) + 1, ncol = 2);

tempMatrix[1,1] = toString("GeneName");
tempMatrix[1,2] = toString("UniProtID");

# this is used to find the column index of gene name in gene name with GeneID file
columnIndexForGeneName = match("geneName", names(geneNameFile));
columnIndexForGeneName

# this is used to find the column index of gene Id in gene name with GeneID file
columnIndexForGeneIDinGeneNameFile =  match("geneId", names(geneNameFile));
columnIndexForGeneIDinGeneNameFile

# this is used to find the column index of uniprot id in UniProtID with GeneID file
columnIndexForUniProtID = match("UniProtKB", names(uniProtIDFile));
columnIndexForUniProtID

# this is used to find the column index of gene Id in UniProtID with GeneID file
columnIndexForGeneIDinUniProtIDFile = match("GENEID", names(uniProtIDFile));
columnIndexForGeneIDinUniProtIDFile


# use to temporary store the count 
tempCount = 1;

for(i in 1 : length(geneNameFile[,1]))
{
  isMatchedGeneName = match(toString(geneNameFile[i,columnIndexForGeneName]), geneNameFile[,columnIndexForGeneName]);
  isMatchedGeneName
  
  isMatchedGeneId = match(geneNameFile[isMatchedGeneName,columnIndexForGeneIDinGeneNameFile], uniProtIDFile[,columnIndexForGeneIDinUniProtIDFile]);  
  isMatchedGeneId

  if(as.numeric(isMatchedGeneName) > 0) 
  {
    # increases the count to store values
    tempCount = tempCount + 1;
    
    tempMatrix[tempCount,1] = toString(geneNameFile[isMatchedGeneName,columnIndexForGeneName]);
    tempMatrix[tempCount,2] = toString(uniProtIDFile[isMatchedGeneId,columnIndexForUniProtID]);
  
  }
}
tempMatrix
outputFile <- readline("please enter output file for gene name to uniprotId name: ");
write.table(as.data.frame(tempMatrix),file = outputFile,sep=",",col.names=FALSE, row.names = FALSE, na = "NA");