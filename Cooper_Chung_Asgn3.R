# BTMA 531 Assignment 3
# Cooper Chung (Redacted Student ID)
# cooper.chung1@ucalgary.ca

# Question 1a
# For this example, we DO NOT separate the data into training and test datasets. For the purpose of this question where we're not necessarily trying to make a prediction
# (Unsupervised learning - not output variable), but rather trying to cluster observations, we do not need to separate the data into training and test datasets.
# However, I will mention that in general it is a good practice to split the data into training and test datasets, because it reduces the overfitting to
# the training data, and helps reduce bias when developing a model. It makes more sense to evaluate the performance of a model using data it has never seen before,
# rather than data it has seen during the training phase. This also helps us assess how well the model fares when given new, unseen data. The dataset is also
# somewhat small at only 200 observations, splitting it into training and test will help us get the most use out of our limited dataset.

data_ad <- read.csv("ad.csv", T) # Read the ad data into R and get rid of the first and sales column since they are not needed.
data_ad <- data_ad[,-1]
data_ad <- data_ad[,-4]


# Question 1b
data_ad_scaled <- scale(data_ad)                        # Scale the data

set.seed(1)                                             # Set seed to 1 for replicability

km3 <- kmeans(data_ad_scaled, centers = 3, nstart = 10) # Cluster once using K-means with 3 clusters

set.seed(1)                                             # Set seed to 1 again for replicability (not sure if needed - just to be safe)

km4 <- kmeans(data_ad_scaled, centers = 4, nstart = 10) # Cluster once using K-means with 4 clusters


# Question 1c
plot(data_ad, col = (km3$cluster), main = "K-means Clustering, K=3", pch = 19, cex = 1)  # Create the plot
# From this plot, we can observe a clear line between the three scaled dimensions. On the bottom right are sales observations with above average TV spending,
# but lower than average radio spending. On the bottom left to the top left, we can see observations with a low TV and low Radio spending, up to a low TV and high Radio
# spending. Finally for the third cluster on the top right, we can see observations with either an average spending or higher than average spending in both the TV and Radio
# categories. From this plot, we can also see that there seems to be a cluster with spending on Radio (both low and high) when the TV spending is lower than average.


# Question 1d
error_ratios <- km4$withinss / km4$tot.withinss  # Calculate within-cluster and  total error ratios for each cluster for K=4.

barplot(error_ratios, main = "Within-cluster errors vs Total Errors for K=4", xlab = "Cluster", ylab = "Error Ratio", col = "blue", names.arg = 1:4)  # Plot errors

# From this barplot, we can see that Cluster 2 is the most homogeneous, due to it having the lowest error ratio. 


# Question 1e
hier_ad_scaled_comp <- hclust(dist(data_ad_scaled), method = "complete")  # Cluster the data using hierarchical clustering and complete linkage

cutree(hier_ad_scaled_comp, 4)  # Cluster the data into 4 clusters

par(mfrow = c(1, 1))  # Plot the dendrogram
plot(hier_ad_scaled_comp, main = "Complete Linkage", xlab = "", ylab = "")


# Question 1f
cutree(hier_ad_scaled_comp, h = 3)  # Cluster the data using the same dendrogram from the previous part, using a dissimilarity level of h=3
# From the output, we can see a maximum number of 10, which is the number of clusters in this clustering.


# Question 2a
library(neuralnet)  # Load neuralnet to perform neural net functions
library(MASS)       # Load MASS to use Boston dataset

set.seed(1)         # Set seed for replicability

training_index <- sample(nrow(Boston), 400, replace = F)  # Create indices for training dataset

maxes <- apply(Boston[training_index,], 2, max) # Gather maxes and mins for each column in the training dataset
mins <- apply(Boston[training_index,], 2, min)

data_scaled <- data.frame(scale(Boston, center = mins, scale = maxes - mins)) # Scale data so it is suitable for use in neural net

boston_train_nn <- data_scaled[training_index,] # Create training and test data
boston_test_nn <- data_scaled[-training_index,]

set.seed(1) # Set seed again for replicability (again, not sure if needed, doing it just in case)

boston_NN <- neuralnet(medv ~ ., boston_train_nn, linear.output = T, lifesign = "minimal")  # Perform neural net

plot(boston_NN) # Plot neural network


# Question 2b
test_nn_pred <- compute(boston_NN, boston_test_nn[, c(1:13)]) # Using the neural network that we created above, predict medv for the test dataset

test_nn_pred <- (test_nn_pred$net.result * (maxes[14] - mins[14])) + mins[14] # De-normalize data

mean((Boston[-training_index,]$medv - test_nn_pred)^2)  # Calculate MSE - get a value of 22.5076


# Question 3a
library(nnet) # Load library to use the class.ind() function

shopper_data <- read.csv("online_shoppers_intention2.csv", T) # Read data into R

set.seed(1) # Set seed for replicability

training_index <- sample(nrow(shopper_data), 2000, replace = F) # Create list of indexes for training data

maxes <- apply(shopper_data[training_index, 1:10], 2, max) # Gather maxes and mins for each numerical column in the training dataset
mins <- apply(shopper_data[training_index, 1:10], 2, min)

# Scale shopper data with categorical data being scaled independently
shopper_data_scaled <- data.frame(scale(shopper_data[, 1:10], center = mins, scale = maxes - mins), VisitorType = class.ind(as.factor(shopper_data[,11])),
                                  Weekend = class.ind(as.factor(shopper_data[,12])), Revenue = class.ind(as.factor(shopper_data[,13])))

shopper_train_nn <- shopper_data_scaled[training_index,] # Create training and test data
shopper_test_nn <- shopper_data_scaled[-training_index,]

set.seed(1) # Set seed for replicability when doing NN

shopper_NN <- neuralnet(Revenue.FALSE + Revenue.TRUE ~ ., shopper_train_nn, hidden = c(10, 10, 10), linear.output = F) # Create Neural Net Model

plot(shopper_NN)  # Plot Neural Net


# Question 3b
pred_shopper_NN <- compute(shopper_NN, shopper_test_nn[, 1:15])     # Predict test dataset using neural net model

pred_shopper_NN <- apply(pred_shopper_NN$net.result, 1, which.max)  # Assign the appropriate classes

pred_shopper_NN <- pred_shopper_NN - 1                # Subtract 1 from results since dataset is using 1's and 0's, not 1's and 2's to delineate classes

mean(pred_shopper_NN == shopper_test_nn$Revenue.TRUE) # Calculate MSE, gives us 0.8608906 or 86.09%

table(pred_shopper_NN, shopper_test_nn$Revenue.TRUE)  # Draw confusion Matrix


# Question 3c
library(mltools)
library(xgboost)

shopper_data <- read.csv("online_shoppers_intention2.csv", T, 
                         colClasses = c("numeric", "numeric", "numeric",
                                        "numeric", "numeric", "numeric",
                                        "numeric", "numeric", "numeric",
                                        "numeric", "factor",  "factor", 
                                        "factor"))  # Read the data into R again - note that this is the only way I was able to make XGBoost work

hotShopperData_train <- as.matrix(one_hot(as.data.table(shopper_data[training_index,])))  # One-hot the data
hotShopperData_test <- as.matrix(one_hot(as.data.table(shopper_data[-training_index,])))

shopper_xgb2 <- xgboost(data = hotShopperData_train[, 1:15], label = hotShopperData_train[, 17], nrounds = 1000, eta = 0.01, max_depth = 2, verbose = F)  # Perform XGBoost

library(gbm)

shopper_gbm <- gbm(Revenue ~ ., distribution = "gaussian", data = shopper_data[training_index,], interaction.depth = 2, n.trees = 1000, shrinkage = 0.01,
                   n.cores = NULL, verbose = F)


# Question 3d
shopper_xg_pred <- predict(shopper_xgb2, hotShopperData_test[, 1:15]) # Using the model we created above, predict the test dataset

shopper_xg_pred <- ifelse(shopper_xg_pred >= 0.5, 1, 0) # Using 0.5 as our threshold, assign classes either 0 or 1

mean(shopper_xg_pred == hotShopperData_test[, 17])  # Calculate MSE - gives us 0.8940949, or 89.40%

table(shopper_xg_pred, hotShopperData_test[, 17])   # Create confusion matrix


# Question 4a
library(e1071)  # Load package to perform SVM

accent_data <- read.csv("accent-mfcc-data-1.csv", T)    # Read data into R

accent_data$language <- as.factor(accent_data$language) # Turn the language column into a factor so SVM can use the data

set.seed(1) # Set seed for replicability

training_index_accent <- sample(nrow(accent_data), 280, replace = F)  # Sample training indices

accent_training <- accent_data[training_index_accent,]  # Create training and test datasets
accent_test <- accent_data[-training_index_accent,]

accent_svm <- svm(language ~ ., data = accent_training, kernel = "radial", cost = 4, scaled = T)  # Perform SVM


# Question 4b
accent_pred <- predict(accent_svm, accent_test) # Using the model we generated above, predict the language of the test dataset

mean(accent_pred == accent_test$language) # Calculate MSE, gives us 0.8163265, or 81.63%

table(accent_pred, accent_test$language)  # Create confusion matrix
