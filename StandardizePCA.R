#Import data
train <- read.csv("D:/ISEN 613/Project/Dataset 3 HAR/drive-download-20180424T041945Z-001/HartrainUncor.csv", 
                  header = TRUE, sep = ",")
test <- read.csv("D:/ISEN 613/Project/Dataset 3 HAR/drive-download-20180424T041945Z-001/HartestUncor.csv", 
                 header = TRUE, sep = ",")
HARdata <- read.csv("D:/ISEN 613/Project/Dataset 3 HAR/drive-download-20180424T041945Z-001/HarUncor.csv", 
                    header = TRUE, sep = ",")

dim(HARdata)
numPredictors=ncol(HARdata)#number of columns
numPredictors #number of predictors

library(caret)
library(lattice)
#Standardize
##First column in the data is Response, second is subject id which cannot be a predictor
##Train
zScaleTrain = preProcess(train[, 3:numPredictors])
scaledXTrain = predict(zScaleTrain, train[, 3:numPredictors])
##Test
zScaleTest = preProcess(test[, 3:numPredictors])
scaledXTest = predict(zScaleTest, test[, 3:numPredictors])
##Total
zScaleHARdata = preProcess(HARdata[, 3:numPredictors])
scaledXHARdata = predict(zScaleHARdata, HARdata[, 3:numPredictors])

dim(scaledXTrain)
n=ncol(scaledXTrain)#number of columns
n #number of pure predictors is n

#PCA
pr.out=prcomp(scaledXTrain, scale=TRUE)
names(pr.out)
pr.out$center
pr.out$scale
pr.out$rotation

biplot(pr.out,scale=0)
pr.out$sdev
pr.var=pr.out$sdev^2
pr.var

pve=pr.var/sum(pr.var)
sum=0
pve
#cumulative sum of variance ratios (PVE)
for (i in 1:n)
{
  if(i==1)
  {sum[1]=pve[1]}
  else
  {sum[i]=sum[i-1]+pve[i]}
}
sum[1:n]

#computing PVE indices for different % of variances
varlim=0#fraction of total variance to be explained
j=0
for(k in 2:9)
{
  varlim[k]=k*0.1
  for (i in 2:n-1)
  {
    if(sum[i]>=varlim[k] && sum[i-1]<varlim[k])
    {j[k]=i}
  }
}
b=cbind(varlim*100,j)#% of variance explained vs number of PCs
b
plot(pve, xlab="PC",ylab="pve",ylim=c(0,0.2),xlim=c(0,50),type='b')#PVE vs PCs
plot(sum, ylim=c(0,1),xlim=c(0,n-1),type='b')#% of variance explained vs number of PCs