# Practical Machine Learning Project

Overview
--------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.



Data Processing
---------------

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
if (!file.exists("./data")) {
  dir.create("./data")
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./data/pml-training.csv", method = "curl")
  download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', destfile = "./data/pml-testing.csv", method = "curl")
}

pmlTraining <- read.csv('./data/pml-training.csv', na.strings = c("NA", "", " ", "#DIV/0!"))
pmlTesting <- read.csv('./data/pml-testing.csv', na.strings = c("NA", "", " ", "#DIV/0!"))
```

Exploratory analysis & Feature Reduction
----------------------------------------

Training dataset contains 19622 observations of 160 variables.


```r
dim(pmlTraining)
```

```
## [1] 19622   160
```

Remove all empty and NA columns. Now we have 60 variables.


```r
pmlTraining <- Filter(function (x) {
  !all(is.na(x) || x == '')
}, pmlTraining)
dim(pmlTraining)
```

```
## [1] 19622    60
```

Remove unnecessary columns (case num, username, timestamps, time windows). Finally we have 53 variables.

```r
pmlTraining <- pmlTraining[,-(1:7)]
names(pmlTraining)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```


Model training
--------------

Split data into training and cross validation sets:


```r
inTrain <- createDataPartition(pmlTraining$classe, p = .6, list = FALSE)
trainSet <- pmlTraining[inTrain,]
testSet <- pmlTraining[-inTrain,]
```

Use Random Forest for model training:


```r
trCtrl <- trainControl(method = "cv", number = 5)
model <- train(classe ~ ., method = "rf", data = trainSet, preProcess = "pca", ntree = 100, trControl = trCtrl)
model
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction, scaled, centered 
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 9420, 9422, 9421, 9420, 9421 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9609380  0.9505785  0.004712204  0.005984505
##   27    0.9499836  0.9367269  0.004303985  0.005441049
##   52    0.9496441  0.9362951  0.004606875  0.005812717
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Cross Validation
----------------

Let's validate model on the cross validation set. We achieved 96.6% accuracy.


```r
pr <- predict(model, testSet);
confusionMatrix(pr, testSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2215   39    7    2    1
##          B    4 1443   36    1    5
##          C    6   35 1304   67   28
##          D    7    1   15 1216    7
##          E    0    0    6    0 1401
## 
## Overall Statistics
##                                           
##                Accuracy : 0.966           
##                  95% CI : (0.9617, 0.9699)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9569          
##  Mcnemar's Test P-Value : 2.616e-15       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9924   0.9506   0.9532   0.9456   0.9716
## Specificity            0.9913   0.9927   0.9790   0.9954   0.9991
## Pos Pred Value         0.9784   0.9691   0.9056   0.9759   0.9957
## Neg Pred Value         0.9970   0.9882   0.9900   0.9894   0.9936
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2823   0.1839   0.1662   0.1550   0.1786
## Detection Prevalence   0.2886   0.1898   0.1835   0.1588   0.1793
## Balanced Accuracy      0.9918   0.9717   0.9661   0.9705   0.9853
```

Results
-------

Apply model to the test set and submit results to Coursera. Total Score is 19 / 20 (95% accurancy).


```r
answers <- predict(model, pmlTesting)

if (!file.exists("./answers")) {
  dir.create("./answers")
}

pml_write_files = function(x) {
  n = length(x)
  for (i in 1:n) {
    filename = paste0("./answers/problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}
pml_write_files(answers)
answers
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
