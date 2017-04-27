## dependancies
require(ggplot2)
require(reshape2)
# Utilities

udf_utils_castFlexibleDataFrame <- function(object) {

    # Utility function that coerces vectors, dataframes, matrices and other enumerable types to data frame.

    cNames_ <- colnames(object) # get object column names.
    dfObj_ <- as.data.frame(object) # cast object as data frame.

    if (is.null(cNames_)) {
        # if no column names assign generic.
        for (i in 1:length(dfObj_)) {
            colnames(dfObj_)[i] <- paste0("c", i)
        }
    }
    return(dfObj_)
}

udf_utils_rootMeanSquareError <- function(predVals, actVals) {

    # this function returns teh root mean square error of predicted values and actual values. 

    predVals <- udf_utils_castFlexibleDataFrame(predVals) 
    actVals <- udf_utils_castFlexibleDataFrame(actVals)

    if (nrow(predVals) != nrow(actVals)) {
        stop("differring predictions and actual values.")
    }

    eW_ <- (sum((predVals - actVals) ^ 2)) / 2

    eRMS_ <- sqrt(2 * (eW_) / nrow(predVals))

    return(eRMS_)

}

udf_utils_checkLabelsObj <- function(labels) {

    # utility function for checking label array object for comformance.

    # for the regresser, we are looking for numeric types.
    # handle data frames.
    if (class(labels) == "data.frame") {
        if (ncol(labels) == 1) {
            # if dataframe object is passed, and has the length of one column, try cast as numeric.
            # any non numeric characters will be forced NA. Error out.
            suppressWarnings(labels <- as.numeric(labels))

            # atthe risk of falling through many if conditions
            if (sum(is.na(labels)) > 0) {
                stop("Non numeric values found in label set. Non numerics cannot be implemented in regressor.")
            }
        } else {
            stop("labels must be a one dimensional array!")
        }
    } else {
        # if not data frame, 
        suppressWarnings(labels <- as.numeric(labels))

        if (sum(is.na(labels)) > 0) {
            stop("Non numeric values found in label set. Non numerics cannot be implemented in regressor.")
        }
    }

    return(labels)
}


# 1.1
# read data
tr_dt <- read.csv("Task1A_train.csv")
ts_dt <- read.csv("Task1A_test.csv")

udf_knn <- function(trainingSet, trainingLabels, testSet, K = 3) {

    #store Lengths
    lnTrDt <- nrow(as.data.frame(trainingSet)) # training data
    lnTrLb <- nrow(as.data.frame(trainingLabels)) # training label
    lnTsDt <- nrow(as.data.frame(testSet)) # test set data

    # convert the passed in objects to data frames.
    trainingSet <- udf_utils_castFlexibleDataFrame(trainingSet)
    testSet <- udf_utils_castFlexibleDataFrame(testSet)

    # check labels object conforms or force conform.
    trainingLabels <- udf_utils_checkLabelsObj(trainingLabels)


    # quick check to see nrows match
    if (lnTrDt != lnTrLb) {
        stop("lengths of training and test sets do not match!")
    }
   
    # calc distance matrix
    
    # subset the test set data points and their distance to the training set data points.
    dist_ <- as.matrix(dist(rbind(testSet, trainingSet)))[1:lnTsDt, (lnTsDt + 1):(lnTsDt + lnTrDt)] 

    ## for each test sample...
    pred_ <- as.numeric() # instantiate prediction container
    for (i in 1:lnTsDt) {
        ## get the distances 
        nn <- as.data.frame(sort(dist_[i,], index.return = TRUE))[1:K, 2]

        ## get the mean of the k nearest neigbours. 
        pred_[i] <- (mean(trainingLabels[nn]))
    }
    return(pred_)
}

udf_knn(tr_dt, tr_dt$y, ts_dt, 3)



# 1.2
udf_knnErrors <- function(trainingSet, trainingLabels, testSet, testLabels, Kmax = 20) {
    # function takes training and test set data along with maximum K value and returns the RMSE of the test and training sets.

    # convert the passed in objects to data frames.
    trainingSet <- udf_utils_castFlexibleDataFrame(trainingSet)
    testSet <- udf_utils_castFlexibleDataFrame(testSet)

    # check labels object conforms or force conform.
    trainingLabels <- udf_utils_checkLabelsObj(trainingLabels)
    testLabels <- udf_utils_checkLabelsObj(testLabels)

    # Instantiate container for errors.
    err_ <- data.frame('K' = 1:Kmax, 'train' = rep(0, Kmax), 'test' = rep(0, Kmax))

    for (k in 1:Kmax) {

        # for each k, append training set RMSE value. 
        err_[k, 'train'] <- udf_utils_rootMeanSquareError(udf_knn(trainingSet, trainingLabels, trainingSet, k),trainingLabels)

        # for each k, append test set RMSE value. 
        err_[k, 'test'] <- udf_utils_rootMeanSquareError(udf_knn(trainingSet, trainingLabels, testSet, k),testLabels)

    }
    return(err_)
}

# get the errors K= 1:20
errs_ <- udf_knnErrors(tr_dt$x1, tr_dt$y, ts_dt$x1, ts_dt$y)
# plot
ggplot(data = errs_, aes(1 / K)) + geom_line(aes(y = train, colour = "train")) + geom_line(aes(y = test, colour = "test")) + ggtitle("Error RMS") + xlab("1/K") + ylab("RMSE") +labs(colour = "Dataset")

# save file.
ggsave("fig.1.a.1.pdf", device = "pdf")

# get the optimum value of K in terms of testing error.
errs_[errs_$test == min(errs_$test), 'K']


# 2.1
udf_CrossValidation <- function(trainingData, trainingLabels, numFolds = 10, K = 3,Seed=1234) {

    # this is a cross validation function which returns the errors of different folds of the training data.

    # comform dataframe
    trainingData <- udf_utils_castFlexibleDataFrame(trainingData)
    trainingLabels <- udf_utils_checkLabelsObj(trainingLabels)
    
    #get lengths
    lnTrDt <- nrow(as.data.frame(trainingData)) # training data
    lnTrLb <- nrow(as.data.frame(trainingLabels)) # training label

    
    # quick check to see nrows match
    if (lnTrDt != lnTrLb) {
        stop("lengths of training and test sets do not match!")
    }

    # reindex data randomly
    # set seed for reproducibility
    set.seed(Seed)
    index_ <- sample(1:lnTrDt)
    # shuffle training and test sets
    trainingData <- udf_utils_castFlexibleDataFrame(trainingData[index_,])
    trainingLabels <- trainingLabels[index_]

    # generate folds.
    folds_ <- cut(seq(1:lnTrDt), breaks = numFolds, labels = FALSE)

    # cast dataframe to store RMS.
    errs_ <- data.frame("KFold" = 1:numFolds, "RMSE" = rep(0, numFolds))


    # run kNN
    for (i in 1:numFolds) {

        # get index of folds belonging to current fold
        intern_index <- which(folds_ == i, arr.ind = TRUE) 

        #internal test data
        intern_testData <- trainingData[intern_index,]
        #internal test labels
        intern_testLabels <- trainingLabels[intern_index]

        #internal train data
        intern_trainData <- trainingData[ - intern_index,]
        #internal train labels
        intern_trainLabels <- trainingLabels[ - intern_index]

        #get kNN predictions
        pred_ <- udf_knn(intern_trainData, intern_trainLabels, intern_testData, K = K)

        # errs_[i, 'K-Fold'] <- i
        errs_[i,'RMSE'] <- udf_utils_rootMeanSquareError(pred_,intern_testLabels)

    }

    return (errs_)


}

udf_CrossValidation(tr_dt$x1,tr_dt$y)


udf_multipleKCrossVal <- function(trainingData, trainingLabels, numFolds = 10, Seed = 1234, KMax = 20) {
    
    # wrapper function for doing CV on multiple K values.

    # instantiate data frame to contain the average of 10 error numbers.
    err_ <- data.frame('K' = 1:KMax, 'AvgError' = rep(0, KMax), 'stDev' = rep(0,KMax) ,'minStDev' = rep(0, KMax), 'maxStDev' = rep(0, KMax))
        
    # iterate through k and get the average errors
    for (k in 1:KMax) {

        # get the average error.
        errMean_ <- mean(udf_CrossValidation(trainingData, trainingLabels, numFolds, Seed, K = k)$RMSE)
        
        # n.b we are using $ notation here since this is a wrapper function and only usable with this specific cross validation function.

        # get the sd
        stDev_ <- sd(udf_CrossValidation(trainingData, trainingLabels, numFolds, Seed, K = k)$RMSE)

        # check if the mean or Sd < 0 and error our

        if (errMean_ < 0 | stDev_ < 0) {
            stop("Mean or standard deviation is < 0, something is wrong!")
        }

        # fill data frame
        err_[k, 'AvgError'] <- errMean_
        err_[k, 'stDev'] <- stDev_
        err_[k, 'minStDev'] <- errMean_ - stDev_
        err_[k, 'maxStDev'] <- errMean_ + stDev_
                        
    }
    
    return(err_)
}

# run an instance.
errs_ <- udf_multipleKCrossVal(tr_dt$x1, tr_dt$y)

# plot the graph
ggplot(data = errs_, aes(1 / K)) + geom_line(aes(y = AvgError, colour = "Average Error")) + geom_line(aes(y = minStDev, colour = "Error Bounds"), linetype = 2) + geom_line(aes(y = maxStDev, colour = "Error Bounds"), linetype = 2) + ggtitle("Average RMSE cross-validation vs K") + xlab("1/K") + ylab("Average RMSE") + labs(colour = "")

# save the plot
ggsave("fig.1.a.2.pdf", device = "pdf")

# Report K with min avg RMSE and min sd RMSE
paste("K of minimum average error:",errs_[errs_$AvgError == min(errs_$AvgError), 'K'])
paste("K of minimum standard deviation of error:", errs_[errs_$stDev == min(errs_$stDev), 'K'])



### Bootsrapping

# load datasets
tr_dt <- read.csv("Task1B_train.csv")
ts_dt <- read.csv("Task1B_test.csv")

# subset training and test variables. (explicitly pass in labels)
trainSet <- subset(tr_dt, select = -c(y))
testSet <- subset(ts_dt, select = -c(y))

# modifying the boots trap code to handle regression
# we need to make the function return RMSE and note misclassification

# design bootstrapper.
udf_bootstrapIndexer <- function(x = 100, sampleSize = x, times = 100, Seed = 1234) {
    
    # this function applies bootstrap sampling and returns a MATRIX OF INDICES with the sample pattern.

    # develop container for matrices
    index_ <- matrix(nrow = times, ncol = sampleSize)
    set.seed(Seed)
    for (t in 1:times) {
        index_[t,] <- sample(x = x, size = sampleSize, replace = TRUE)

    }
    return(index_)
}

x <- udf_bootstrapIndexer(nrow(tr_dt), 25, 100)


udf_knnBootstrapErr <- function(trData, trLabels, tsData, tsLabels, sampleSizeN = 25, timesL = 100, K = 1:20, Seed = 1234) {

    # generate bootstrap indexes
    indexes_ <- udf_bootstrapIndexer(nrow(trData), sampleSizeN, timesL, Seed)

    # container
    err_ <- data.frame('K' = 0, 'L' = 0, 'test' = rep(0, timesL * length(K)))

    # instantiate iterator tracker
    i <- 1
    for (k in K) {

        for (l in 1:timesL) {

            #get the indices from the bootstrap sampling
            index_ <- indexes_[l,]

            # save the value of k & l
            err_[i, 'K'] = k
            err_[i, 'L'] = l

            # calculate RMSE and store
            err_[i, 'test'] <- udf_utils_rootMeanSquareError(udf_knn(trData[index_,], trLabels[index_], tsData[index_,], k), tsLabels[index_])

            # increment iterator count.
            i <- i + 1
        }
    }
    return(err_)
}

# get the bootstrap errors
errs_ <- udf_knnBootstrapErr(trainSet, tr_dt$y, testSet, ts_dt$y, 25, 100)

# melt the data frame
errsMelt_ <- melt(errs_, id = c('K', 'L'))

# recast names
names(errsMelt_) <- c("K", "L", "type", "RMSE")

# make the plot
ggplot(data = errsMelt_[errsMelt_$type == 'test',], aes(factor(K), RMSE, fill = type)) + geom_boxplot(outlier.shape = NA) + scale_color_discrete(guide = guide_legend(title = NULL)) + ggtitle('RMSE vs. K (Box Plot)') + theme_minimal()

# save the plot
ggsave("fig.1.b.1.pdf", device = "pdf")



# changing with times.
udf_knnBootstrapErrTimes <- function(trData, trLabels, tsData, tsLabels, sampleSizeN = 25, timesL = 1:100, K = 10, Seed = 1234) {

    # generate bootstrap indexes
    indexes_ <- udf_bootstrapIndexer(nrow(trData), sampleSizeN, max(timesL), Seed)

    # container
    err_ <- data.frame('K' = 0, 'L' = 0, 'test' = rep(0, length(timesL) * length(K)))

    # instantiate iterator tracker
    i <- 1
    for (k in K) {
        
        for (l in timesL) {

            #get the indices from the bootstrap sampling
            index_ <- indexes_[l,]

            # save the value of k & l
            err_[i, 'K'] = k
            err_[i, 'L'] = l

            # calculate RMSE and store
            err_[i, 'test'] <- udf_utils_rootMeanSquareError(udf_knn(trData[index_,], trLabels[index_], tsData[index_,], k), tsLabels[index_])

            local_ <- local_ + 1


            # increment iterator count.
            i <- i + 1
        }
    }
    return(err_)
}





##
# our times sequence
timesL <- seq(10, 200, 10)

# vector 
timesVec <- c()

# get iteration length
for (i in timesL) {
    for (ii in 1:i) {
        timesVec <- append(timesVec,i)
    }
}

# create a data frame with iteration length
errs_ <- data.frame("times" = timesVec, "rmse" = rep(0, length(timesVec)))

# create an iterator for keeping track of rows.
r <- 1


for (i in timesL) {
    # create bootstrap indexes with each size times l
    indexes_ <- udf_bootstrapIndexer(nrow(trainSet), 25, i)
    for (ii in 1:i) {
        # make index
        index_ <- indexes_[ii,]
        # make predictions
        preds_ <- udf_knn(trainSet[index_,], tr_dt$y[index_], testSet[index_,], K = 10)
        # get the rmse
        rmse_ <- udf_utils_rootMeanSquareError(preds_, ts_dt$y[index_])

        # store it in teh dataframe
        errs_[r, 'rmse'] <- rmse_

        # increment iterator
        r <- r + 1
        print(r)

    }

}


errsMelt_ <- melt(errs_, id = c('times'))

names(errsMelt_) <- c("times", "type", "RMSE")

# make the plot
ggplot(data = errsMelt_[errsMelt_$type == 'rmse',], aes(factor(times), RMSE, fill = type)) + geom_boxplot(outlier.shape = NA) + scale_color_discrete(guide = guide_legend(title = NULL)) + ggtitle('RMSE vs. times (Box Plot)') + theme_minimal()

# save the plot
ggsave("fig.1.b.2.pdf", device = "pdf")