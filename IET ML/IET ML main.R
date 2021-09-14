library(caret)
library(arm)
library(mboost)
library(klaR)
library (ipred)
library(nnet)
library(writexl)
library(readr)


Identification <- read_csv("Gender_Identification_IET_talk.csv")
#View(Identification)
# attach the iris dataset to the environment
#data(Identification)

# rename the dataset
dataset <- Identification[-c(1)]


#
# # define the filename
# filename <- "iris.csv"
#
# # load the CSV file from the local directory
# dataset <- read.csv(filename, header=FALSE)
#
# # set the column names in the dataset
# colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")
predict <- Gender~.

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Gender, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

#dimension of dataset
dim(dataset) #120 instances (rows) 5 attributes (column)

# list types for each attribute
sapply(dataset, class) #factor is nominal data. levels. ranking.

# take a peek at the first 5 rows of the data
head(dataset)
tail(dataset)

# list the levels for the class
#levels(dataset$Gender)

# summarize the class distribution
percentage <- prop.table(table(dataset$Gender)) * 100
cbind(freq=table(dataset$Gender), percentage=percentage)

summary(dataset)

#split numeric and factor. input and target
x <- dataset[,1:4]
y <- dataset[,5]


# boxplot for each attribute on one image
par(mfrow=c(1,9))
for(i in 1:9) {
  boxplot(x[,i], main=names(Identification)[i])
}


par(mfrow = c(1,1))
plot(y)

# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- c("Accuracy")



# Random Forest
set.seed(100)
tunegridRF = expand.grid(mtry = seq(0, 200, by=1))
fit.rf <- train(predict, data=dataset,
                method="rf",
                metric = metric,
                tuneGrid = tunegridRF,
                trControl = control)


#############################################################################################################
#############################################################################################################


#summarize accuracy of models
results <- resamples(list(
                          rf=fit.rf
                          ))

summary(results)

# summarize Best Model
print(fit.rf)
ggplot(fit.rf) + theme_bw()+ggtitle("Random Forest Model: Accuracy vs Number of Randomly Selected Predictors")+theme(plot.title = element_text(hjust = 0.5))
ggsave("Random Forest Model Accuracy vs Number of Randomly Selected Predictors.png", width = 5, height = 5)
write.csv(fit.rf[["results"]],"C:\\Users\\User\\Google Drive\\Master Syahmi\\5G\\PHD\\Subcarrier Prediction\\Machine learning subcarrier prediction2\\subcarrier prediction\\IET_Rf_Results.csv", row.names = TRUE)


# estimate skill of RF on the validation dataset
predict.rf <- predict(fit.rf, validation)
confusionMatrix(predict.rf, validation$Gender)



predict.rf <- predict(fit.rf, dataset)


