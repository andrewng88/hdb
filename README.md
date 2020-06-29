
![](https://www.99.co/blog/singapore/wp-content/uploads/2018/09/geylang-pasir-ris.jpg)
<sup>image courtesy of 99.co</sup>

# Prediction of resale HDB ( House ) price in Singapore

Aside from solving the core problem of prediction(regression) we will be doing EDA to 
<li> uncover price trends 
<li> premium locations 
<li> budget friendly locations
<li> factors that (do/do not ) dictate the price

## Dataset
Obtained from [data.gov.sg](https://data.gov.sg/dataset/resale-flat-prices)<br>
Data timeline from 1990 to 2019.

## Timeline

Project started mid May 2020 and due to complete by mid April.

### Progress

* :+1:Brainstorm on the problem and technology to be used 
* <li> Feature Engineering
* :+1: General data cleaning before Feature Engineering ( as below )
* :+1: Scraping longitude, latitude for MRTs untill 2025 ( includes those that are yet to be opened but with price factored in )
* :+1: Scraping longitude, latitude for popular schools  
* :+1: Scraping distance between location to CBD ( Raffles Place MRT )
* :+1: Scraping Postcodes for each HDB - for deployment purposes
* :+1: Exploratary Data Analysis 
* :+1: Visualization 
* :+1: Feature Selection 
* :+1: Modelling 
* :+1: Deployment 
* :+1: Conclusion and Recommendations 

## Deployment

Initial Proof Of Concept deployed to AWS[done](https://github.com/andrewng88/streamlit_aws)
Second Proof Of Concept deployed locally [done](https://github.com/andrewng88/streamlit_model) 

## Technology used

<li> geoPY - generic(backup) scraping API and data processing
<li> ONEMap- specific scraping API for SG context
<li> pandas - data processing
<li> numpy - data processing
<li> plotly express - interactive viz
<li> pydeck - 3D 
<li> docker - containerization
<li> streamlit - viz/ML frontend framework for ML
<li> AWS - cloud hosting

## Visualization
At the notebook level we will use plotly express
and we will convert useful charts to streamlit for deployment.

## Modelling
Baseline would just be the mean of our target variable ( naive )
We will only be using explainable algorithms Linear Regression ( Lasso, Ridge , Elasticnet) 

Linear Regression is the vanila model whereby there is no regularization introduced whereas
Lasso, Ridge and Elasticnet has regularization which will prevent overfitting.

Usually **Lasso** will be used for feature selection<br>
![](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1543418448/eq11_ij4mms.png)
The LHS is the minimization cost function and the RHS is the regularization term. The features are
optimized based on Beta and there is high tendency that features will be zero-ed. 
The lambda is the hyperparameter that can be optimzied by sklearn<br>. Lambda of zero will convert Lasso
to plain Linear Regression. <br>Drawing parallel to sklearn , alpha is the same as lambda.

![](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1543418449/eq7_ylxudw.png)
With respect to **Ridge**, the beta is squared hence the tendency to reduce the co-efficients to infinitesimal 
but not zero. It helps with reducing multi-collinearity too.


For **Elasticnet**, it's the combi between Lasso and Ridge. You can see the two terms there.<br>
![](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1543418448/eq12_vh6ilt.png)
When alpha is 0, it will be full Ridge.<br>
When alpha is 1, it will be full Lasso.
Lambda is how powerful the regularization is.

For SK_learn, we will use l1_ratio instead of alpha.
Sklearn's alpha is actually the lambda.

## Metrics
We wil be using Adjusted R2 

* [Adjusted R2-Squared](https://www.listendata.com/2014/08/adjusted-r-squared.html) will be used to compare between the models as it will take into account to the total variables used.
The lesser the better it will be.

![](https://3.bp.blogspot.com/-MFqQLwbGwd4/WMO7tz39beI/AAAAAAAAF9I/Bjpfohgv_5E4lu7HzHJBQtXsBM--byqPwCLcB/s1600/rsquared.png) <br>
This is the usual R2 formula whereby we divide 1 - ( SSE predicted squared ) / ( SSE mean squared ) . Lies between 0% to 100%.
The higher the better because the numerator will have a lower error which will contribute to the higher R2 value.

![](https://4.bp.blogspot.com/-qEGt3DaQIF0/V2meLITZj3I/AAAAAAAAEp4/WKCs0FrI1JsovDMwaw1r1iUboULfRI7MwCLcB/s1600/stb1.png) <br>
A slightly modified Adjusted R2 will make use of the number of predictors and sample size. A higher p(number of predictors) will reduce/penalize the score.
Hence, feature selection will be useful in this area.

* RMSE is selected as has the same unit as the dependent variable. If we have an RMSE of $10,000 it means our prediction is off by exact figure



## Author

**Andrew Ng ** - [linkedin](https://www.linkedin.com/in/sc-ng-andrew/)<br>
**Lau Lee Ling ** - [linkedin](https://www.linkedin.com/in/lauleeling/)

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
