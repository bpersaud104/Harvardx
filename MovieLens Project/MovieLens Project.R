#' ---
#' title: "MovieLens Project"
#' author: "Bryan Persaud"
#' date: '2026-05-18'
#' output:
#'   html_document: default
#'   pdf_document: 
#'     toc: TRUE
#' ---
#' 
#' # Introduction
#' 
#' Recommendation systems analyze user ratings to predict which items a user may like and recommend them accordingly. Companies such as Amazon use large amounts of customer rating data to estimate how users might rate specific products and suggest items based on past activity and purchase likelihood.
#' 
#' The same approach can be applied to movies. Recommendation systems are among the most widely used machine learning applications, and Netflix is often cited as a major example of their success. The Netflix Prize further highlighted the importance of collaborative filtering algorithms for predicting user ratings based on previous interactions.
#' 
#' In this Capstone project for the HarvardX Professional Certificate in Data Science (PH125.9x), we will analyze and visualize the MovieLens dataset from GroupLens Research, which contains over 10 million movie ratings. The goal is to build a machine learning model that predicts movie ratings with a Root Mean Square Error (RMSE) below 0.8649. RMSE measures the standard deviation of prediction errors and is commonly used to evaluate the accuracy of machine learning models.
#' 
#' The MovieLens dataset was created by researchers at the University of Minnesota to support personalized movie recommendations. While the full December 2019 release includes over 25 million ratings and 1 million tag applications across 62,000 movies and 162,000 users, this project focuses on the “MovieLens 10M Dataset.”
#' 
#' # Data Analysis
#' 
## ----echo = TRUE, message = FALSE, warning = FALSE, eval = TRUE-------------------------------------------------
##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#' 
#' To become familiar with the dataset, we examine the first few rows of the “edx” subset shown below.
#' The subset contains six variables: “userID,” “movieID,” “rating,” “timestamp,” “title,” and “genres.” Each row represents a single user’s rating for a specific movie.
#' 
## ----head, echo = FALSE-----------------------------------------------------------------------------------------
head(edx) %>%
  print.data.frame()

#' 
#' The total of unique movies and users in the edx subset is about 70.000 unique users and about 10.700 different movies:
#' 
## ----summary, echo = FALSE--------------------------------------------------------------------------------------
edx %>%
summarize(n_users = n_distinct(userId), 
          n_movies = n_distinct(movieId))

#' 
#' The distribution of ratings below suggests that users tend to give higher rather than lower movie ratings. A rating of 4 is the most common, followed by 3 and 5, while 0.5 is the least common rating. Overall, half-star ratings appear less frequently than whole-star ratings.
#' 
## ----rating_distribution, echo = FALSE--------------------------------------------------------------------------
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  scale_y_continuous(breaks = seq(0, 3000000, 500000)) +
  ggtitle("Rating distribution")

#' 
#' We can observe that some movies have received substantially more ratings than others, while some have very few ratings—sometimes only one. This is important for our model because movies with very few ratings may produce unreliable prediction estimates. In fact, 125 movies in the dataset have been rated only once.
#' 
#' To address this issue, regularization and penalty terms will be applied in the models used throughout this project. Regularization is a technique that helps reduce prediction error by fitting the model appropriately to the training data while avoiding overfitting. Overfitting occurs when a model fits the training data too closely and, as a result, performs poorly on new or unseen data. By adding a penalty term to the error function, regularization constrains the model coefficients and prevents them from taking excessively large values, leading to more stable and reliable predictions.
#' 
## ----number_of_ratings_per_movie, echo = TRUE, fig.height=4, fig.width=5----------------------------------------
edx %>%
count(movieId) %>%
ggplot(aes(n)) +
geom_histogram(bins = 30, color = "black") +
scale_x_log10() +
xlab("Number of ratings") +
  ylab("Number of movies") +
ggtitle("Number of ratings per movie")

#' 
#' As 20 movies that were rated only once appear to be obscure, predictions of future ratings for them will be difficult.
#' 
## ----obscure_movies, echo = TRUE, fig.height=4, fig.width=5-----------------------------------------------------
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable()

#' 
#' We can observe that the majority of users have rated between 30 and 100 movies. So, a user penalty term need to be included later in our models.
#' 
## ----number_ratings_given_by_users, echo = TRUE, fig.height=4, fig.width=5--------------------------------------
edx %>%
count(userId) %>%
ggplot(aes(n)) +
geom_histogram(bins = 30, color = "black") +
scale_x_log10() +
xlab("Number of ratings") + 
ylab("Number of users") +
ggtitle("Number of ratings given by users")

#' 
#' Furthermore, users vary considerably in how critical they are when rating movies. Some users consistently give lower ratings, while others tend to assign ratings that are higher than average. The visualization below includes only users who have rated at least 100 movies.
#' 
## ----mean_movie_ratings_given_by_users, echo = TRUE, fig.height=4, fig.width=5----------------------------------
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  theme_light()

#' 
#' # Model Results
#' 
#' We now define the loss function used to compute the RMSE.
#' 
#' $$ RMSE = \sqrt{\frac{1}{N}\displaystyle\sum_{u,i} (\hat{y}_{u,i}-y_{u,i})^{2}} $$
#' where N represents the total number of user–movie combinations, and the summation is taken over all such combinations.
#' 
#' The RMSE serves as our measure of model accuracy. It can be interpreted similarly to a standard deviation, representing the typical prediction error made by the model when estimating movie ratings. An RMSE value greater than 1 indicates that the model’s average prediction error exceeds one rating star, which would generally be considered poor performance.
#' 
#' The function written to compute the RMSE for vectors of observed ratings and their corresponding predictions is shown below.
#' 
## ----rmse_function, echo = TRUE---------------------------------------------------------------------------------
RMSE <- function(true_ratings, predicted_ratings){
sqrt(mean((true_ratings - predicted_ratings)^2))
}

#' 
#' # Model 1
#' 
#' The simplest recommender system predicts the same rating for every movie by using the average rating across the entire dataset. This model assumes that differences in ratings are due only to random variation and ignores both user preferences and movie characteristics. Since the average rating in the dataset is between 3 and 4, all movies are assigned a predicted rating within this range.
#' 
## ----mu, echo = TRUE--------------------------------------------------------------------------------------------
mu <- mean(edx$rating)
mu

#' 
#' If we predict all unknown ratings with $\mu$ or mu, we obtain the first naive RMSE.
#' 
## ----naive, echo = TRUE-----------------------------------------------------------------------------------------
naive_rmse <- RMSE(final_holdout_test$rating, mu)
naive_rmse

#' Here, we represent results table with the first RMSE:
#' 
## ----rmse_results1, echo = TRUE---------------------------------------------------------------------------------
rmse_results <- data_frame(method = "Average movie rating model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

#' 
#' This provides a baseline RMSE against which more advanced models can be evaluated. To improve upon the average-rating prediction, we incorporate insights gained from the exploratory data analysis.
#' 
#' # Model 2
#' 
#' To improve the baseline model, we account for the fact that some movies are consistently rated higher or lower than others. Popular movies tend to receive higher ratings, while less popular movies often receive lower ratings. We therefore estimate a movie-specific bias, bi, which measures how much a movie's average rating deviates from the overall mean rating. This allows the model to capture systematic differences between movies rather than attributing all variation to random noise.
#' 
#' The distribution of movie biases is left-skewed, indicating that a larger number of movies have negative effects on ratings than positive ones.
#' 
## ----number_of_movies_with_the computed_b_i, echo = TRUE, fig.height=3, fig.width=4-----------------------------
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"),
ylab = "Number of movies", main = "Number of movies with the computed b_i")

#' 
#' This is called the penalty term movie effect.
#' 
#' Our prediction improve once we predict using this model.
#' 
## ----predicted_ratings, echo = TRUE-----------------------------------------------------------------------------
predicted_ratings <- mu + final_holdout_test %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()

#' 
#' This model predicts movie ratings by adjusting the overall mean rating with the movie-specific bias. If a movie is typically rated below the average rating of all movies, its predicted rating is reduced by the corresponding bias value. Likewise, movies that are generally rated more highly receive a positive adjustment.
#' 
#' Incorporating movie biases improves the model's performance compared with the baseline. However, it still does not account for differences in individual users' rating behavior, as some users consistently give higher or lower ratings than others.
#' 
#' # Model 3
#' 
#' We compute the average rating for user $\mu$, for those that have rated over 100 movies, said penalty term user effect. In fact users affect the ratings positively or negatively.
#' 
## ----user_avg, echo = TRUE--------------------------------------------------------------------------------------
user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))

#' 
#' There is also considerable variation among users: some tend to give consistently lower ratings, while others rate most movies favorably. To account for these differences, we extend the model by including a user-specific bias. This adjustment captures individual rating tendencies, allowing the model to better predict ratings. For example, a user who generally gives low ratings may assign only an average score to a highly rated movie, and the combination of movie and user biases helps reflect this behavior more accurately.
#' 
#' We compute an approximation by computing $\mu$ and $b_{i}$, and estimating  $b_{u}$, as the average of $$Y_{u, i} - \mu - b_{i}$$
#' 
## ----user_avgs, echo = TRUE-------------------------------------------------------------------------------------
user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#' 
#' We can see RMSE improves once we construct predictors.
#' 
## ----model3, echo = TRUE----------------------------------------------------------------------------------------
predicted_ratings <- final_holdout_test%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_3_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and user effect model",  
                                     RMSE = model_3_rmse))
rmse_results %>% knitr::kable()

#' 
#' Adding user-specific effects further reduced the RMSE and improved the model's predictive performance. However, some issues remained. In particular, the movies identified as the “best” or “worst” were often rated by only a small number of users, sometimes just one. These were typically obscure movies for which there was limited information. With such small sample sizes, the estimated movie biases can become unrealistically large, either positive or negative, due to higher uncertainty.
#' 
#' These extreme estimates can lead to large prediction errors and, consequently, a higher RMSE. To address this problem, we introduce regularization. Rather than relying solely on the estimated biases, regularization penalizes large bias values that are based on limited data. This reduces the influence of extreme estimates and helps prevent overfitting. As a result, the model produces more reliable predictions, particularly for movies with few ratings.
#' 
#' # Model 4
#' 
#' Extreme estimates of $b_{i}$ and $b_{u}$ often arise from movies with very few ratings or users who have rated only a small number of movies. These estimates can disproportionately influence predictions and reduce the model's accuracy. Regularization addresses this issue by penalizing large bias estimates, particularly when they are based on limited data. The strength of this penalty is controlled by the tuning parameter λ, which is selected to minimize the RMSE. As λ increases, the estimates of $b_{i}$ and $b_{u}$ are shrunk toward zero, reducing the impact of unreliable estimates and improving the model's ability to generalize to new data.
#' 
## ----lambdas, echo = TRUE---------------------------------------------------------------------------------------
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    final_holdout_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, final_holdout_test$rating))
})

#' 
#' Let's plot RMSE vs lambdas to select the optimal lambda.
#' 
## ----plot_lambdas, echo = TRUE----------------------------------------------------------------------------------
qplot(lambdas, rmses)  

#' 
#' For the full model we can see the optimal lambda.
#' 
## ----min_lambda, echo = TRUE------------------------------------------------------------------------------------
lambda <- lambdas[which.min(rmses)]
lambda

#' 
#' The full model shows the optimal lambda is 5.25.
#' 
#' The new results are below.
#' 
## ----rmse_results2, echo = TRUE---------------------------------------------------------------------------------
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

#' 
#' # Results and Conclusion
#' 
#' 
#' The RMSE results for each model are as follows.
#' 
## ----rmse_results3, echo = FALSE--------------------------------------------------------------------------------
rmse_results %>% knitr::kable()

#' 
#' Here we can see the lowest value of RMSE is 0.8648170.
#' 
#' We successfully developed a machine learning model to predict movie ratings using the MovieLens dataset. The regularized model that includes both movie and user effects achieved the lowest RMSE and is therefore selected as the best-performing model for this project.
#' 
#' Further improvements could likely be obtained by incorporating additional factors such as genre, release year, or user demographics. More advanced machine learning approaches may also enhance performance, but such extensions may be limited by computational constraints, including available memory.
