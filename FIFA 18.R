#FIFA 18 Complete Player Dataset
#The dataset contains all the statistics and playing attributes of all the players
#in the Full version of FIFA 18.

#library for this Project
library(dplyr)
library(tidymodels)
library(randomForest)
library(caret)
library(class)
library(rpart)
library(pROC)
set.seed(3086122)


#Data Importation: Player Dataset
setwd("C:/Users/stanley/Desktop/Stiles/sem2/Predictive/Rating")
personal<- read.csv("PlayerPersonalData.csv")
attribute <- read.csv("PlayerAttributeData.csv")


# Perform the left join with the "ID" column as the join condition
joined_data <- left_join(personal, attribute, by = "ID", suffix = c(".persona", ".attribute"), 
                         relationship = "many-to-many")
glimpse(joined_data) 
view(joined_data)

#Creating a new Column to Categorise the Overall Ratings
joined_data1 <- joined_data %>%
  mutate(Rate_Class = case_when(
    Overall >= 81 ~ "Excellent",
    Overall >= 61 ~ "Average",
    TRUE ~ "Below Average"
  ))


glimpse(joined_data1)
view(joined_data1)

#Create a subset of variables of interest
rate <- joined_data1%>%
  select(Rate_Class, Acceleration, Aggression, Agility, Ball.control, Composure,
         Curve, Dribbling,  Finishing, Free.kick.accuracy, Heading.accuracy, 
         Interceptions,Sliding.tackle,Marking, Stamina, Standing.tackle, Balance, 
         Penalties, Positioning, Reactions,Short.passing, Shot.power, Sprint.speed,
         Strength, Vision, Long.passing, Long.shots,Jumping)


glimpse(rate)

#Create Factors for the Rate Class Column
rate_1 <- within(rate, {
  Rate_Class <- factor(Rate_Class,
                  levels = c("Excellent", "Average", "Below Average"),
                  labels = c("1", "2", "3"))
})

glimpse(rate_1)

#Changing Some varables in numeric values
rate_2 <- rate_1 %>%
  mutate(
    Acceleration = as.numeric(Acceleration),
    Aggression = as.numeric(Aggression),
    Agility = as.numeric(Agility),
    Ball.control = as.numeric(Ball.control),
    Composure = as.numeric(Composure),
    Curve = as.numeric(Curve),
    Dribbling = as.numeric(Dribbling),
    Finishing = as.numeric(Finishing),
    Free.kick.accuracy = as.numeric(Free.kick.accuracy),
    Heading.accuracy = as.numeric(Heading.accuracy),
    Interceptions = as.numeric(Interceptions),
    Sliding.tackle = as.numeric(Sliding.tackle),
    Marking = as.numeric(Marking),
    Stamina = as.numeric(Stamina),
    Standing.tackle = as.numeric(Standing.tackle),
    Balance = as.numeric(Balance),
    Penalties = as.numeric(Penalties),
    Positioning = as.numeric(Positioning),
    Reactions = as.numeric(Reactions),
    Short.passing = as.numeric(Short.passing),
    Shot.power = as.numeric(Shot.power),
    Sprint.speed = as.numeric(Sprint.speed),
    Strength = as.numeric(Strength),
    Vision = as.numeric(Vision),
    Long.passing = as.numeric(Long.passing),
    Long.shots = as.numeric(Long.shots),
    Jumping = as.numeric(Jumping)
  )
rate_3 <- na.omit(rate_2) #Removing NA characters
rate_3
glimpse(rate_3)


sum(is.na(rate_3)) #Checking For Missing Value within the dataset

#Data Partioning
rate_split <- initial_split(rate_3, prop = 0.7,
                                 strata = Rate_Class )

#Populating the training dataset
rate_training <- rate_split%>%
  training()   

#Populating the testing dataset
rate_testing <- rate_split%>%
  testing()

#FEATURE ENGINEERING

set.seed(3086122)

#Preprocessing Techniques for Numeric Predictor Variables
#Finding Correlated Predictor Variables
rate_training%>%
  select_if(is.numeric)%>%
  cor()

# Removing Multicollinearity and normalizing the data
rate_recipe <- recipe(Rate_Class ~ ., data = rate_training) %>%
  step_corr( Acceleration, Aggression, Agility, Ball.control, Composure,
            Curve, Dribbling,  Finishing, Free.kick.accuracy, Heading.accuracy, 
            Interceptions,Sliding.tackle,Marking, Stamina, Standing.tackle, Balance, 
            Penalties, Positioning, Reactions,Short.passing, Shot.power, Sprint.speed,
            Strength, Vision, Long.passing, Long.shots,Jumping,
            threshold = 0.9)%>%
  step_normalize(all_numeric())

#Train Recipe with Prep Function
rate_recipe_prep <- rate_recipe%>%
  prep(training= rate_training)

#Preprocess Training Data
rate_training_prep <- rate_recipe_prep%>%
  bake(new_data = NULL)

rate_training_prep

#Pre-process the Test Data
rate_testing_prep <- rate_recipe_prep%>%
  bake(new_data = rate_testing)

rate_testing_prep

#########################################################################################


#Fit Random Forest Model

randmodel <- randomForest(formula =Rate_Class~.,
                          data = rate_training_prep,
                          ntree = 100)
randmodel
#Prediction with Rand Forest

rate_class_preds <- predict(randmodel, 
                            newdata = rate_testing_prep,
                            type = "class")
rate_class_preds

#Probability Predictions
rate_prob_preds <- predict(randmodel,
                           new_data = rate_testing_prep,
                           type = 'prob')
rate_class_preds

rate_cm <- confusionMatrix(data = rate_class_preds,
                           reference = rate_testing_prep$Rate_Class)
rate_cm #Confusion Matrix
#Visualise the Importance
randomForest::varImpPlot(randmodel,
                         sort= TRUE,
                         main="Variable Importance Plot")



####### AUC ROC CURVE FOR RANDOM FOREST

# Get unique class labels
class_labels <- unique(rate_testing_prep$Rate_Class)

# Function to convert multiclass to binary for a specific class
convert_to_binary <- function(class_label, rate_testing_prep) {
  binary_labels <- ifelse(rate_testing_prep == class_label, "Positive", "Negative")
  return(factor(binary_labels, levels = c("Positive", "Negative")))
}


# Calculate ROC curves and AUC values for each class
roc_curves <- lapply(class_labels, function(class_label) {
  binary_actual <- convert_to_binary(class_label, rate_testing_prep$Rate_Class)
  binary_predicted <- convert_to_binary(class_label, rate_class_preds)
  return(roc(binary_actual, as.numeric(binary_predicted)))
})

# Plot ROC curves for each class
plot(roc_curves[[1]], col = "blue", main = "ROC Curves for Random Forest", ExcellentAverageBelowAverage = 2)
for (i in 2:length(roc_curves)) {
  plot(roc_curves[[i]], add = TRUE, col = rainbow(length(class_labels))[i], ExcellentAverageBelowAverage = 2)
}

# Calculate macro-average AUC value
auc_values <- sapply(roc_curves, function(x) x$auc)
macro_auc_value <- mean(auc_values)

# Print macro-average AUC value
cat("Macro-average AUC-ROC value:", macro_auc_value, "\n")

# Add diagonal reference line for random classification
abline(a = 0, b = 1, lty = 2, col = "red")

# Add legend
legend("bottomright", legend = paste("Class", class_labels, "AUC =", 
                                     round(auc_values, 2)), col = rainbow(length(class_labels)),lwd=2)



###############################################################################################################

#K Nearest Neighbors (KNN)

set.seed(3086122)


#Using the datasets that have already been Feature engineered.
rate_training_prep
rate_testing_prep


glimpse(rate_training_prep) #Checking for missing Values

#Creating train labels
train_labels <- rate_training_prep$Rate_Class
#Fit KNN Model
rate_Pred <-knn(train = rate_training_prep[-23], test =rate_testing_prep[-23], cl= train_labels )

rate_Pred

rate_actual <- rate_testing_prep$Rate_Class

#Create a Confusion Matrix
rate_cm1 <-table(rate_Pred,rate_actual )
rate_cm1

#True Positive: 63+3745+823 =4631
#False Positive: : (55+0) + (27+287) + (0+241)=610
#False Negative: : (27+0) + (55+241) + (0+287) =610
#True Negative: 0

#Accuracy
print(mean(rate_Pred == rate_actual))



## VISUALIZATION For K Nearest Neighbors
# Convert 'rate_cm' to a numeric matrix if needed
rate_cm <- as.matrix(rate_cm)

# Create the heatmap
heatmap(rate_cm,
        col = cm.colors(256),
        scale = "none",
        margins = c(5, 10),
        main = "Confusion Matrix",
        xlab = "rate_actual",
        ylab = "rate_Pred")


# Fit KNN model and predict on the test data

# Get unique class labels
class_labels <- unique(rate_testing_prep$Rate_Class)

# Convert actual labels to binary for each class
binary_actuals <- lapply(class_labels, function(class_label, actual_labels) {
  binary_actual <- ifelse(actual_labels == class_label, 1, 0)
  return(binary_actual)
}, actual_labels = rate_actual)

# Calculate ROC curves and AUC values for each class
roc_curves <- lapply(1:length(class_labels), function(i) {
  roc_obj <- roc(binary_actuals[[i]], as.numeric(rate_Pred == class_labels[i]))
  return(roc_obj)
})

# Calculate macro-average AUC value
auc_values <- sapply(roc_curves, function(x) x$auc)
macro_auc_value <- mean(auc_values, na.rm = TRUE)

# Print macro-average AUC value
cat("Macro-average AUC-ROC value:", macro_auc_value, "\n")

# Plot ROC curves for each class
plot(roc_curves[[1]], col = "blue", main = "ROC Curves for KNN Model", lwd = 2)
for (i in 2:length(roc_curves)) {
  plot(roc_curves[[i]], add = TRUE, col = rainbow(length(class_labels))[i], lwd = 2)
}

# Add diagonal reference line for random classification
abline(a = 0, b = 1, lty = 2, col = "gray")

# Add legend
legend("bottomright", legend = paste("Class", class_labels, "AUC =", round(auc_values, 2)), col = rainbow(length(class_labels)),lwd=2)





###########################################################################################################################

#DECISION TREE
set.seed(3086122)

#Using the datasets that have already been Feature engineered.
rate_training_prep
rate_testing_prep

# Assuming 'result' column is converted to a factor with levels 
decision_tree_model <- rpart(Rate_Class~ ., data = rate_testing_prep, method = "class")


predictions <- predict(decision_tree_model, newdata = rate_testing_prep, type = "class")

# Step 3: Create a confusion matrix
actual <- rate_testing_prep$Rate_Class
confusion_matrix <- table(predictions, actual)
print(confusion_matrix)

# Calculate accuracy from the confusion matrix
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(accuracy)

# Load the necessary library for rpart.plot
library(rpart.plot)

# Assuming 'decision_tree_model' is the result of rpart() function
# Plot the decision tree
rpart.plot(decision_tree_model, extra = 101)

####### AUC ROC CURVE 
# Convert 'actual' and 'predictions' to factors
actual <- factor(actual)
predictions <- factor(predictions)

# Create a list to store ROC objects for each class
roc_curves <- list()

# Calculate AUC-ROC curve for each class (one-vs-all)
for (class_label in levels(actual)) {
  binary_actual <- factor(ifelse(actual == class_label, "Positive", "Negative"), levels = c("Positive", "Negative"))
  binary_predicted <- factor(ifelse(predictions == class_label, "Positive", "Negative"), levels = c("Positive", "Negative"))
  
  roc_obj <- roc(binary_actual, as.numeric(binary_predicted))
  roc_curves[[class_label]] <- roc_obj
}

# Calculate macro-average AUC value
auc_values <- sapply(roc_curves, function(roc_obj) roc_obj$auc)
valid_auc_values <- auc_values[!is.na(auc_values) & sapply(auc_values, is.numeric)]
macro_auc_value <- mean(valid_auc_values, na.rm = TRUE)

# Print macro-average AUC value
cat("Macro-average AUC-ROC value:", macro_auc_value, "\n")

# Plot ROC curves for each class
par(mfrow = c(1, length(roc_curves)))
for (i in seq_along(roc_curves)) {
  plot(roc_curves[[i]], col = rainbow(1), main = paste("ROC Curve for Class", levels(actual)[i]), lwd = 2)
}

# Add legend
legend_labels <- sapply(roc_curves, function(roc_obj) paste("Class", roc_obj$levels[[2]], "AUC =", round(roc_obj$auc, 2)))
legend("bottomright", legend = legend_labels, col = rainbow(length(roc_curves)),lwd =2)

