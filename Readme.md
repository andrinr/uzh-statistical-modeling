# Statistical Modeling

1. Hypthesis to investigate Pehonmena to study
2. Design Experiment
3. Collect Data
4. EDA
5. Propose Statistical Model
6. Fit Model
7. Model Validation 
8. (Go back to 5. )
9. Summarize Model Validation
10. Scientific Conclusion

### Mean Square Error

Lets consider the statistical model: Y = g(x) + e

Then the mean squared error is: E((Y - g(x))^2) = bias(g(x))^2 + Var(g(x)) + variance

### Bias  Variance Tradeoff

Higher variance and very low bias corrsponds to overfitting, lower variance but high bias correpsonds to underfitting. Mean square error encodes both and a minimum of the MSE is desirable.

___



## Resampling

### Empirical density of the estimator

1. Take many samples from the population
2. Compute the estimator
3. Analyze / Plot the empericial density of the estimator

```{r}
N = 1000
vars = numberic(N)
for (i in c(0:N)) {
    samples = rnorm(n = 5, mean = 0, sd = 1)
    vars[i] = var(samples)
}
# Histogram with 50 breaks
hist(vars, breaks = 50)
```

### Cross validation

- **Holdout:** Simple test and training split
- **K-fold:** divide dataset into k subsets, then use one of the subsets as test and the others as training sets
- **Leave one out:** Similar as K-fold but in this case k = N.

### Bootstrapping

Take a sample from the sample.

### Parametric Bootstrapping

1. Estimate parameters of population distribution
2. Use the estimates to simulate the samples

Example on how to generate an universally usable test and train split:

```r
sampleSize = 0.7 * n
# Seq_len creates a sequence which starts at 1 and ends at the given value
train_ind = sample(seq_len(nrow(mtcars)), size = sampleSize)
train <- mtcars[train_ind, ]
test <- mtcars[-train_ind, ]
```

___



## PCA

Create new independant variables where the new independant variables each is a combination of old independant variables. 

We want to maximize the variance on the selected axis. 

```{r}
# scaling will scale the variables to have a unit variance before performing the PCA
pca = prcomp(data, scale = TRUE)
# Gives a summary of the pca
# sdev are the standard deviations of the prinipal components
# rotation, contains the matrix with the eigenvectors
# center The varibale means
# scale the variables standard deviations => the scaling applied to each variable
# Coordinates of the individual observarions
str(pca)
# Plot of the amount of variance explained by each PC
screeplot(pca)
plot(pca)
# Explaines how the two PC are being constructed based on the original variables
biplot(pca)

```

### How many features should we keep?

*Method 1* take as many features s.t. we can explain 75% of the variance. 

*Method 2* Identify the ellbow in the screeplot

### When is PCA suitable

- Multiple continous variables
- Linear relationship between all values
- Large samples are required
- Adequate correlations to be able to reduce variables to a smaller number of dimensions
- No siginifacant outliers

### EOF 

Empirical Orthogonal Functions is when we apply a PCA onto for example a grid. Can be done identical, we simply need to transform the data from a 2D grid data into 1D array. 

### QA

- PCA can be used for projecting and visualizing data in lower dimensions: T
- Max number of PC’s = number of features: T
- All PC’s ar orthogonal to each other: T
- What happens when eigenvalues are roughly equal? PCA will perform badly

___

## Clustering

### Hierarchical Clustering

Dissimilarity can be defined with different strategies. 

- **Single Linkage:** (Nearest neighbour linkage) will measure the distance between the two nearest samples from two groups.
- **Complete Linkage:** (Furhest Neighbout linkage) will measure the distance between the two furthest samples from the two groups. 
- **Ward Method:** In each step find the pair which leads to minimal increase in total within cluster variance.

```{r}
# hclust requries a dissimilary structure, in this case we will go with dist which computes the eulerian distance between all the rows of the input matrix
hc1 = hclust(dist(matrix), method = "single")
hc2 = hclust(dist(matrix), method = "complete")
hc3 = hclust(dist(matrix), method = "ward.D")

# We can plot the clustering like this:
plot(hc1)

# We can return a datastructure where each item is assigned to a cluster, provided number of clusters
cut = cuttree(hc1, 6)

# We can do the same visually like this:
par(mfrow = c(1, 1))
plot(hc1, xlab = "", sub = "")
rect.hclust(hc1, k = 5, border = rainbow(6))
```



### K-Means Clustering

1. Start with k (random) cluster centers
2. Assign obervations to the nearest center
3. Recompute the centeres
4. If centers remain the same, stop, otherwise repeat step 2.

### Model Based Clustering

We assum that the data is a mixture of several clusters, this means that there are soft border between them. 

### Assesing Quality

Is unsupervised therefore measuring the quality is difficult. Cluster can easily be found in random data, and sometimes its hard to distinguish these from meaninungful clusters.

- Shilouette plots indicate the nearest distance of each sample to the another cluster. Distances towards 1 are favourable, 0 or even negative may indicate wrong assignments. 

### QA

- Two runs of K-Means will have the same results: F
- It is possible that the K-Means assignment does not change between two iterations: T
- Clustering is unsupervised: T
- K-Means automatically chooses an optimal number of clusters: F
- Hier. Clustering depends on the linkeage method: T

___



## Classification

### Linear Discrimination

We assume that both categories have a gaussian distribution. 

In the linear case we assume for the distributions to have the same covariance matrices, only the means can vary. We have several distributions and we want to associate each sample to a distribution. The distributions are known, but the samples are intertwined, thus we have to use LDA to decide which sample can be associated with which distribution. 

The discrimination line is at the point (in 2D) with the same densities.

```{r}
lda = lda(response ~ predictor1 + predictor2, data = data)
```



### Quadratic Discrimination

In the quadratic case the distributions can have varying variances. In fact for all samples both variants can be applied but QDA performs than LDA with different variances.

```{r}
qda = qda(response ~ predictor1 + predictor2, data = data)
```

### Fishers Discriminant Rule

It is identital to LDA when covariance matrices are equal. 

### Classification Tree

Actually pretty much the same as a K-D tree, but in this case we have more than 3 Dimensions.

- Easy to understand, explain and visualize. 
- Non robust method, 

```{r}
ct = ctree(response ~ ., data = data)
# Plot it
plot(ct)
```



### Bagging

Use bootstrapping to create many training sets and for each of them create a new tree. Take the average for the final classification.

### Boosting

Similar to bagging but samples are associated with a weight which corresponds to the amount of missclassifications, with this the training will focus on hard cases.

### Random Forest

Similar to bagging but for each tree we choose a random subsample of the featutres, where k = sqrt(n). 

```{r}
rf = randomForest(response ~ ., data = data)
plot(rf)

# Optionally repeat rd with only most important features extracted with
varImpPlot(rf)
rf2 = randomForest(response ~ a + b + c)
```

### QA

- LDA will fail when the discriminatory information lies in the variance and not the mean: T
- LDA tries to model difference in classes, PCA does not take any differences into account: T
- In RF individual trees are buit on a subset of features and observations: T
- Given one very strong predictor, would you rather use bagging or random forest? RF

___



## Linear Model (!)

```{r}
# Classical linear model
lm(predictor ~ responseA + responseB, data = data)
# Linear model with explicitly defined intercept term
lm(predictor ~ 1 + responseA + responseB, data = data)
# Linear model with formulate applied to response variable
# Note that the I() is necessary to bypass the meaning of operators in the context of the LM formula
lm(predictor ~ responseA + I(responseB * 2))
```



### Assumptions

- **Homoscedastic:** Var(error_i) = variance. Look at scale location plot, watch out for a horizontal line and equally spread points. 
- **Independence:** Corr(error_i, error_j) = 0 for i not equal j
- **Gaussian:** The errors are gaussian with a mean of 0. We can look at the QQ plot, if we see a more or less linear distribution, we can assume that the error are gaussian distributed.
- **Linearity:** The regression model is linear in parameters. Look at residuals vs. fitted. Equally spread residuals without distinct patterns are a good indicator for a linear relationship.

Residuals versus leverage helps us identify cases which have high leverage.

### Leave on out for linear models

What is the effect of the i-th obersvation on

- the estimate
- the prediction
- the estimated standard errors

````R
lm = lm(predictor ~ response1 + response2, data=data)
influence.measures(lm)
````

### ANOVA (!)

Analysis of variance

```R
# One way ANOVA
# Test weather any of the group means are different from the overall mean of the data
anova(lmA)
```

Variance that is not explained by the model is called residual variance. 

- **Sum Sq:** Sum of squares, the total variation between the group means and the overall means
- **Mean Sq:** Mean sum of squares, Sum Sq divided by degrees of freedom
- **F-value:** The higher this value, the more likely the variation is caused by real effects and not by chance
- **P-value:** Likelyhood of null hypothesis

```{r}
# Two way ANOVA
anova(lmA, lmB)
```



### Information Criterion (!)

Balances goodess of a fit with its complexity. 

### Standard Error

The standard error of a variable is the same as the variance of its distribution.

### Hypthesis testing for Estimators

For example: H0: B_hat = 0

Known: 

B_hat = 15

SE(B_hat) = 2

n = 10

```R
(15 / 2) > qt(0.975, df = n - 3)
```

### QA

- If we add a feature to a model, the r-squared value itself cannot make any statements about possible improvements of the new model over the old one. 
- We have a LM with z having a p-value of 0.0001. Thus we can conclude that changing z by 1 will have a great impact on the value of y: F
- LR is sensitive to outliers: T

## Mixed Linear Model

### Independency

Is violated if: 

- We heave repeated measures from the same individual
- Hierarchical models
- Longtidual setting, the experiment follows one subject ober an extended period of time

Independency can falsely increase the accuracy of a model, even though this might not be true.

### Random Effect vs Fixed Effect Model

If we want to know more about the performance of the machines. Let us now conider the production time of three machines with random operators, the effect of the machines are fixed while the error term and the influence of the random operators are random. 

On the other hand if we were to be interested in the performance of the operators, the model would be vice versa.

For random effects we are more interested in the distribution of the variable itself, and less on the comparisons of values.

```{r}
library(lme4)
# Random intercept (pred2)
# Different starting points determined by pred2, but not different slopes determined by pred2
lmer(response ~ pred1 + (1 | pred2) , data=data)
# For longtidual data, where x is for example time and pred2 is the id of subjects for which we have multiple datapoints over time
lmer(response ~ pred1 + (x | pred2) , data=data)
```

We can check weather a model is singular with ``isSingular(model)``. In a singular model the parameters are on the boundary of the feasible parameter space.

### QA

- We have measurements for a single patient over time, should we use a mixed model? F

___



## Non Parametric Regression

Datasets where linear models do not fit. Thus we extend the model to Y = g(x) + e, where e follows a normal distribution. In a parametric regression we are looking for the parameters of the function but the shape of the function is known. In a non parametric regression the general shape of the function is unknown.

### Kernel Smoothing (Local estimation approach)

The logic behind smoothing is that we estimate g to be a smoothed version of the sample points y.

Kernel Functions must be: 

- Symmetric around zero and positive
- normalized, i.e. the integral will yield = 1
- […]

Examples for Kernels are:

- Uniform
- Triangle
- Gauss
- [..]

Bandwith choice, the higher the gamma, the the smaller the kernel. 

Kernel smoothers suffer from boundary bias, since there we have observation from one side only. 

````R
# The kernel, the higher the bandwith, the smoother the result
m2 <- with(data, ksmooth(x, y, kernel = "normal / normal / box", bandwidth = 15))
````



### Local Polynomials (Local estimation approach)

Often better at boundaries. Kernel approach cannot correctly fit a line. 

```{r}
lowess(x, y, f = 0.3, iter = 3, delta = 0)
loess(x~y, span = 0.3, degree = 1, family = "symmetric", iterations = 4, surface = "direct")
```

### Splines

Similar to local polynomials but tries to fit polynomial for each point seperatly. Smoothing is reached by reducing the number of knots. 

```{R}
m5 <- with(flies.noNA, smooth.spline(day, mort.rate, spar = 0.5))
```

### Locally adaptive methods

Suited for function which require different smoothing behaviours for different segments of the data. 

### QA

___



## Generalized Linear Models

### Logistic Regression

Model the probability of a certain event to happen. I.e. Pass/Fail, Win/Loose. We use the logistic function to model the probabilities. 

```{r}
glm = glm()
```

Instead of talking by certain factor of increase in y when we change position in x, instead we talk about a change in log odds. 

Fromt the summary of the glm in r we get the temp estime. Its value x will tell us about the log odd increase when we increase x. Which corresponds to a factor inrease of e^value. 



### Logistic Inference

We try to find information about the underlying distribution based on the data. We have E(Y_i) / n_i = p_i.

### Poisson Regression

With the poisson regression we can model count data. 

```{r}
glm = glm(y ~ x, family="poisson", data = data)
```

I the number of y is expected to follow a poisson distribution with Poisson(gamma = exp(intercept + x * x_point)). If Poisson(a) = E(a).

For one unit change in x, the log of expected counts is expected to change by value. => Factor of e^value

### Generalized Linear Model

- Distribution of Y (Binomial, Poisson) ((Normal distribution for linear regression))
- Linear function of predictors x_transposed * Beta
- A function linking E(Y_i) = y_i and the predictors. (In linear regression this is the identity, in Binomial this is logit, in Poisson its log)

Deviance, the higher, the worse the fit. Residual Deviance is the difference between the saturated model (ideal fit) and the fitted model. Another comparison is the null model which simply fits the null hypothesis. Null Deviance is the difference between the null model and the saturated model. 

```R
# Predict the outcome
glm = predict(glm, new=data)
# And its probabilities
glm = predict(glm, new=data, type=response)
```

### Exponential Family of Distributions

Poisson and Binomial Distributions fall into the category of exponential distributions. 

___



## Survival Analysis

Model time till event. For example time till death, lifespan of a product etc. 

Exponenital or Weibull distribution is used. Censoring is when the outcome for a patient is not availabe. 

- We have the CDF of T. F(t). The cummulative distribution function.
- We have the PDF of T. f(t). The probability density function.
- The survival time T, a random variable with a distribution
- Survivor Function: S(t) = P(T >= t), non increasing, approaches zero, 
- The hazard rate: h(t) = P(t <= T < t + delta | T >= t) where delta approaches zero. The item has survived for a time t and we desire the probability that it will not survive another delta time.
- Cumulative hazard function: H(t) = Integral from zero to t over h(u) du
- h(t) = f(t) / S(t)
- S(t) = exp(-H(t))

### Weibull distribution

![Weibull](../img/weibull.png)



### Non parametric methods for survival analysis

Kaplan Meier Kurve, essentially a step function which is a 1:1 fit of the data. 

```R
survfit(surv ~ t, data=data)
```



### Log Rank Test

Used to compare different survival functions. 

- n_1,k is the number of people not having had an event at time k. 
- d_2,k is the number of events at time k.

H0: The two groups have an identical hazard function.  h_1(t) = h_2(t). Under this circumstances we have:

- Expected value for d_i,j: E_i,j = n_i,j * d_j / n_j

```R
survdiff(surv ~ t, data = data)
```



### Binomial vs. Hypergeometric

- Both describe Number of times an event occurs in a fixed number of trials
- Probability is the same for every trial in Binomial, for Hypergeometric distributions the probability changes with every trial as there is no replacement.

### Parametric Modeling of Survivial Data

We want to quantify the effects of covariates and make statemens about it. 

We try to split the hazard function into a baseline component and a covariate effects component. 

```R
cox = coxph(surv ~ rx, data = data)
summary(cox)
```

In the summary we can see the coef and the exp coef value. When the rx increases by 1, the hazard is hazard * exp coef value.

```R
# Formal statistical test
cox.zph()
# Graphical assesment
plot(cox.zph())
```

___



## Time Series

For time series there might be a dependence between variables, as a measurement x_i is dependant on x_i-1. Thus we want to plot the residual of x_i against x_i-1 for all observations. In R this looks something like this:

```R
plot( resids[-c(n,n-1)], resids[-c(1:2)])
```

### Autoregressive Model

The current observation depends on the previous one. Y_t = a * Y_t-1 + e

We can also extend this model by considering more previous datapoints, where we call the number of datapoints the order of the model. 

```R
ar(resid(lm), order= 2)
```

### Moving Average Model

Present observation depends on a weighted average of white noise components: Y_t = e_t + a_0 * e_t-1 … a_q * e_t-q

### ARMA Model

Combined MA and AR model into ARMA  model.

### Auto Correlative Function (ACF)



## Spatial Statistics

The covariance function c(v) is a function on a vector in the domain D. For example if we want to know the covariance between Z(s_1) and Z(s_2) we need to evalutate c(s_1 - s_2). 

- If this function is related to the distance between the two points i.e.: || s_1 - s_2 || then it is called isotropic, otherwise its called anisotropic.

- The difference H = s_1 - s_2 is called the spatial lag.

  

![iso_aniso](../img/isotropic_anisotropic.png)

### Intrinsic Stationary

A process is called intrinsic stationary if E(Z(s_1)) = mean and Var(Z(s_1), Z(s_2)) = 1/2 * gamma(s_1 - s_2). 

Gamma is a function of ||H|| only. 

### Covariogram

```R
# Convert the data into long format
sim.long <- cbind(expand.grid(x1 = 1:100, x2 = 1:100), z = c(sim1))
# Fit a linear model
m1 <- lm(z ~ x1 * x2, data = sim1.long)
# create new dataframe
vario_df <- cbind.data.frame(z_res = resid(m1), x1 = sim1.long$x1, x2 = sim1.long$x2)
# Set coodinates
coordinates(vario_df) <- ~x1 + x2
# Create a covariogram
plot(variogram(z_res ~ 1, vario_df, covariogram = T))
```



### Kriging Predictor

- A Nugget effect model takes the average over all other points as estime, if the data point lies on top of a prediction we use the prediction instead.

___



## Extreme Value Theory

We model the distribution of P(max(X_i) > m). 

- In contrary to classical statistcs we do not model averages. 
- In classical statistics central values are normal distrbited (CLT)
- In extreme values the distribution of the maximas converges to the generalized extreme value distribution. 

```R
require(extRemes) 
gevfit = fevd(data)
summary(gevfit)
```

Where the shape will define the following distributions:

- bigger than 0 : Frechet
- Smaller than 0: Weibull
- Equal 0: Gambel

We can also inspect the fit visually:

```R
plot(gevfit, type="density")
plot(gevfit, type="qq")
plot(gevfit, type="prob")
```

### Modeling Peaks over thresholds

Instead of modeling the maxima, which severely limits the datapoints, we can also model the peaks over threshold. In this case we have P(X - u > y | X > u)

But in this case the limit theorem does not hold anymore. In this case the theoretical distribution is the generalized Pareto distribution. 

```R
gevfit = fevd(data, threeshold = 10)
```

### Threshold Selection for Modeling Peaks over thresholds

- Mean residual life plot: 

___

## Neural Networks





