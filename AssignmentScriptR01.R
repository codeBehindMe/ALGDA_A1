## Assignment 1 ###

## Part A. ​Model Complexity and Model Selection

#   Question 1 [KNN Regressor, 20 Marks]
#       1.1 Implement the KNN regressor function:
#           knn(train.data, train.label, test.data, K = 3)
#       1.2 Plot the training and the testing errors versus 1/K for K=1,..,20 in one plot, using the Task1A_train.csv and Task1A_test.csv ​ datasets provided for this assignment. Save the plot as fig.1.a.1.pdf​ and attach it to your report.
#       1.3 Report the optimum value for K in terms of the testing error. Discuss the values of K corresponding to underfitting and overfitting based on your plot in fig.1.a.1.pdf​


#   Question 2 [K­fold Cross Validation, 20 Marks]
#       2.1 Implement a K­Fold Cross Validation (CV) function for your KNN regressor: cv(train.data, train.label, numFold = 10) which takes the training data and their labels(continuous values), the number of folds, and returns errors for different folds of the training data.
#       2.2 Using the training data, run your K­Fold CV where the numFold is set to 10. Change the value of K = 1, .., 20 and for each K compute the average 10 error numbers you have got. Plot the average error numbers versus 1 / K for K = 1, .., 20. Further, add two dashed lines around the average error indicating the average + / ­ standard deviation of errors. Save the plot as fig.1.a.2.pdf a​ nd attach it to your report.
#       2.3 Report the values of K that result to minimum average error and minimum standard deviation of errors based on your cross validation plot in fig.1.a.2.pdf.​


## Part B. ​Prediction Uncertainty with Bootstrapping

#   Question 3 [Bootstrapping, 20 Marks]
#       3.1 Modify the code in Activity 2 to handle bootstrapping for KNN regression.
#       3.2 Load Task1B_train.csv and Task1B_test.csv ​sets. Apply your bootstrapping for KNN regression with times = 100(the number of subsets), size = 25(the size of each subset), and change K = 1, .., 20(the neighbourhood size) . Now create a boxplot where the x ­axis is K, and the y ­axis is the average error(and the uncertainty around it) corresponding to each K. Save the plot as fig.1.b.1.pdf a​ nd attach it to your report.
#       3.3 Based on fig.1.b.1.pdf,​how does the test error and its uncertainty behave as K increases ?
#       3.4 Load Train1B_train.csv and Train1B_test.csv ​sets. Apply your bootstrapping for KNN regression with K = 10(the neighbourhood size), size = 25(the size of each subset), and change times = 10, 20, 30, .., 200(the number of subsets) . Now create a boxplot where the x ­axis is ‘times’, and the y ­axis is the average error(and the uncertainty around it) corresponding to each value of ‘times’. Save the plot as fig.1.b.2.pdf ​ and attach it to your report
#       3.5  Based on fig.1.b.2.pdf,​how does the test error and its uncertainty behave as the number of subsets in bootstrapping increases ?

## Part C. ​Probabilistic Machine Learning

#   Question 4 [Bayes Rule, 20 Marks]
#       4.1 ​Recall the simple example from Appendix A of Module1. Suppose we have one red and one blue box. In the red box we have 2 apples and 6 oranges, whilst in the blue box we have 3 apples and 1 orange. Now suppose we randomly selected one of the boxes and picked a fruit. If the picked fruit is an apple, what is the probability that it was picked from the blue box ?

## Question 5 [Maximum Likelihood, 20 Marks]
#   5.1 ​As opposed to a coin which has two faces, a dice has 6 faces. Suppose we are given a dataset which contains the outcomes of 10 independent tosses of a dice:D := { 1, 4, 5, 3, 1, 2, 6, 5, 6, 6 }. We are asked to build a model for this dice, i.e. a model which tells what is the probability of each face of the dice if we toss it. Using the maximum likelihood principle, please determine the best value for our model parameters.




# 1.1
# read data
train.data <- read.csv("Task1A_train.csv")
test.data <- read.csv("Task1A_test.csv")

udf_knn <- function(trainingSetFull, trainingLabels, testSetFull, K = 3) {

    warning("this function uses euclidian distance only!")

    # cast locals
    dt_tr <- trainingSetFull
    lb_tr <- trainingLabels
    dt_ts <- testSetFull

    #store Lengths
    lnTrDt <- nrow(as.data.frame(train.data)) # training data
    lnTrLb <- nrow(as.data.frame(train.label)) # training label
    lnTsDt <- nrow(as.data.frame(test.data)) # test set data
    
    # quick check to see nrows match
    if (lnTrDt != lnTrLb) {
        stop("lengths of training and test sets do not match!")
    }
    if (class(train.label) != "numeric") {
        if (ncol(train.label) > 1) {
            stop("training labels are multi dimensional!")
        }
            else {
                stop("please provide a numeric labels for regression.")
            }
    }

    # calc distance matrix
    
    dist_ <- as.matrix(dist(rbind(test.data, train.data), method = 'euclidian'))[1:lnTsDt, (lnTsDt + 1):(lnTsDt + lnTrDt)] # subset the test set data points and their distance to the training set data points.

    ## for each test sample...
    pred_ <- as.numeric() # instantiate prediction container
    for (i in 1:lnTsDt) {
        ## get the distances 
        nn <- as.data.frame(sort(dist_[i,], index.return = TRUE))[1:K, 2]

        ## get the mean of the k nearest neigbours. 
        pred_[i] <- (mean(train.label[nn]))
    }
    return (pred_)
}


# 1.2
udf_knnErrors <- function(train.data,train.label,test.data,Kmax=20) {

    err_ <- data.frame('K' = 1:Kmax, 'train' = rep(0, Kmax), 'test' = rep(0, Kmax))

    for (k in 1:Kmax) {
        err_[k,'train'] <- sum(udf_knn(train.data,train.label,train.data,k) != train.label)/nrow(train.data) * 100
    }

    return(err_)
}