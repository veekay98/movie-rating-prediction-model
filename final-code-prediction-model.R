library(readr) # For reading CSV files.
library(text2vec) # For text processing and vector space models.
library(randomForest) # For implementing the random forest algorithm.
library(dplyr) # Provides a consistent set of verbs for data manipulation.
library(caret) # Useful for data splitting and pre-processing.

# Read 1000 rows from the dataset
df <- read_csv("/Users/vigneshkrishnan/Downloads/final-movies.csv", n_max = 2500) 

# Remove Rows with Empty Values in Specified Columns
df <- df[!is.na(df$overview) & df$overview != "" & 
           !is.na(df$Cast) & df$Cast != "" & 
           !is.na(df$Directors) & df$Directors != "" & 
           !is.na(df$year) & df$year != "" & 
           !is.na(df$encoded_genres) & df$encoded_genres != "" & 
           !is.na(df$popularity) & df$popularity != "", ] 

# Function to Tokenize Text
tokenize <- function(texts) {
  # Tokenize text using text2vec's itoken function.
  itoken(texts, tokenizer = word_tokenizer, ids = NULL, progressbar = FALSE)
}

# Custom tokenizer function for cast and director names
tokenize_names <- function(texts) {
  # Remove square brackets and split on comma
  cleaned_texts <- gsub("\\[|\\]", "", texts) 
  # Tokenize the cleaned text.
  itoken(strsplit(cleaned_texts, split = ",\\s*"), 
         tokenizer = identity, 
         ids = NULL, 
         progressbar = FALSE)
}

# Function to Create TF-IDF Matrix from Tokens
create_tfidf_matrix <- function(tokens) {
  vocab <- create_vocabulary(tokens) # Create a vocabulary from the tokens.
  vectorizer <- vocab_vectorizer(vocab) # Create a vectorizer based on the vocabulary.
  dtm <- create_dtm(tokens, vectorizer) # Create a document-term matrix.
  tfidf <- TfIdf$new() # Initialize a new TfIdf object.
  # Transform the document-term matrix using TF-IDF.
  fit_transform(dtm, tfidf)
}

# Apply Text Processing and TF-IDF Transformation
tokens_overview <- tokenize(df$overview) # Tokenize the 'overview' column.
tfidf_overview <- create_tfidf_matrix(tokens_overview) # Create a TF-IDF matrix for the 'overview' column.

# Use custom tokenizer for 'Cast' and 'Directors'
tokens_cast <- tokenize_names(df$Cast) # Tokenize the 'Cast' column.
tfidf_cast <- create_tfidf_matrix(tokens_cast) # Create a TF-IDF matrix for the 'Cast' column.

tokens_directors <- tokenize_names(df$Directors) # Tokenize the 'Directors' column.
tfidf_directors <- create_tfidf_matrix(tokens_directors) # Create a TF-IDF matrix for the 'Directors' column.

# Rename Columns in TF-IDF Dataframes to Avoid Duplication
colnames(tfidf_overview) <- paste("ovw", colnames(tfidf_overview), sep = "_") # Rename columns in 'overview' TF-IDF matrix.
colnames(tfidf_cast) <- paste("cast", colnames(tfidf_cast), sep = "_") # Rename columns in 'Cast' TF-IDF matrix.
colnames(tfidf_directors) <- paste("dir", colnames(tfidf_directors), sep = "_") # Rename columns in 'Directors' TF-IDF matrix.

# Combining Data in Chunks
final_df <- cbind(df[, c("popularity", "year", "encoded_genres", "vote_average")], 
                  as.data.frame(as.matrix(tfidf_overview))) # Combine the main dataframe with 'overview' TF-IDF matrix.
final_df <- cbind(final_df, as.data.frame(as.matrix(tfidf_cast))) # Add 'Cast' TF-IDF matrix.
final_df <- cbind(final_df, as.data.frame(as.matrix(tfidf_directors))) # Add 'Directors' TF-IDF matrix.

# Splitting Data into Training and Testing Sets
partition <- createDataPartition(final_df$vote_average, p=0.8, list=FALSE) # Split the data into training and testing sets.
training_set <- final_df[partition, ] # Create the training set.
testing_set <- final_df[-partition, ] # Create the testing set.

# Separate Features and Target
train_features <- select(training_set, -vote_average) # Extract features for the training set.
train_target <- training_set$vote_average # Extract target variable for the training set.
test_features_aligned <- select(testing_set, -vote_average) # Extract features for the testing set.
test_target <- testing_set$vote_average # Extract target variable for the testing set.

# Random Forest Model using randomForest package
model <- randomForest(x = train_features, y = train_target, ntree = 500) # Train a random forest model with 500 trees.

# Prediction and Evaluation
predictions <- predict(model, test_features_aligned) # Make predictions on the testing set.
mse <- mean((predictions - test_target)^2) # Calculate Mean Squared Error.
r_squared <- postResample(pred = predictions, obs = test_target)[["Rsquared"]] # Calculate R-squared value.

# Print the evaluation metrics
print(paste("Mean Squared Error:", mse))
print(paste("R-squared:", r_squared))
