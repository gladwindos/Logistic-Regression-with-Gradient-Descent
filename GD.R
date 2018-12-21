#Sigmoid function and it's derivative

sig <- function(v) {
  1 /(1 + exp(-v))
}

sig.deriv <- function(v) {
  exp(v) / (exp(v) + 1)^2
}


# Function to implement logistic regression gradient descent
lgr <- function(y, X, steps=100, learning = 0.01, seed = 1) {
  set.seed(seed)
  
  # Number of observations
  n <- length(y)
  
  # Initial wieghts
  b0 <- runif(1, -0.7, 0.7)
  b <- runif(ncol(X), -0.7, 0.7)
  
  while (steps > 0) {
    steps <- steps - 1
    
    P <- vector()
    d <- vector()
    for (i in 1:n) {
      w <- b0 + sum(b*X[i,])
      #Forward pass
      P[i] <- sig(w)
      #Backward pass
      d[i] <- 2*(y[i]-P[i])*sig.deriv(w)
    }

    #Update weights
    b0 <- b0 + (learning/n)*sum(d*1)
    
    for (j in 1:(length(b))) {
      b[j] <- b[j] + (learning/n)*sum(d*X[,j])
    }
  }
  return(c(b0,b))
}

# Function to calculate the predictions 
lgr.predict <- function(b, X) {
  
  predictions <- vector()
  
  for (i in 1:nrow(X)) {
    predictions[i] <- sig(b[1] + sum(b[2:length(b)]*X[i,]))
  }
  return(predictions)
}

# Function to calculate the MSE
mse <- function(y, p) {
  return(mean((y - p)^2))
}

# Loading Auto dataset
library(ISLR)

# Creating new variable 'high'
high <- rep(0, 392)
high[Auto$mpg >= 23] <- 1
Auto$high <- high

#Creating dummy variables for origin
origin3 <- rep(0, 392)
origin2 <- rep(0, 392)
origin3[Auto$origin == 3] <- 1
origin2[Auto$origin == 2] <- 1
Auto$origin3 <- origin3
Auto$origin2 <- origin2

# Normalising quantitative attributes
set.seed(1111)
X <- scale(Auto[,c(4,5,7)])

# Splitting dataset into training and test sets
X <- cbind(X, origin2,origin3)
y <- high

train <- sample(1:nrow(X), 196)
X.train <- X[train,]
y.train <- y[train]
X.test <- X[-train,]
y.test <- y[-train]

# Training algorithm on different learning rates and number of steps

rates <- c(0.0001, 0.001, 0.01, 0.1)
steps <- c(1, 10, 100, 1000) 
results <- vector()
names <- vector()

for (rate in rates) {
  for (step in steps) {
    auto.lgr <- lgr(y.train, X.train, steps = step, learning = rate)
    print(auto.lgr)
    train.pred <- lgr.predict(auto.lgr, X.train)
    train.mse <- mse(y.train, train.pred)
    
    test.pred <- lgr.predict(auto.lgr, X.test)
    test.mse <- mse(y.test, test.pred)
    
    results <- rbind(results, c(train.mse, test.mse))
    names <- rbind(names, paste('r: ', rate, 's: ', step))
  }
}

colnames(results) <- c('Training MSE', 'Test MSE')
rownames(results) <- names
print(results)


