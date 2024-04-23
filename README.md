![](cover123.png)

Welcome to our project for the NTU course *SC1015 Introduction to Data Science and Artificial Intelligence*!

In this project, we explore how to become more attractive on dating app(lovoo).

The main page of our project is [here](https://wangkinga.github.io/).

And our presentation video is [here](https://youtu.be/sWD81_SmO8E).

## Content

All code is located under the src directory.

Please read through the code in the flowing sequence:

- [`Base data1`](lovoo_v3_users_api-results.csv)
- [`Base data2`](lovoo_v3_users_instances.csv)
- [`Whole project`](1015P_gp7.ipynb)

## Motivation

Digital Evolution

- In the age of digital courtship, dating apps have become a central platform for romantic connections
- These platforms' success significantly depends on the perceived attractiveness of users' profiles, impacting both match potential and user engagement.
Profile Optimization Benefits

-For individual users, comprehending what makes a profile attractive enhances their dating prospects.
-For app developers, this knowledge helps improve user experience and satisfaction on their platforms.

## Problem Formulation

How to become more attractive in dating app?

- Which variable most effectively indicates the attractiveness of a user on dating apps?
- What variables demonstrate a strong correlation with the key indicator of attractiveness?

## Data preparation

### Data Exploration
- 3972 responses
- 43 features
- All female
- Taken integer and boolean values as primary exploration

### Understanding the Data

#### Integer Data

##### Central Tendency of frequency
##### Spread of frequency
- Mean Medium Q25 Q50 Skew

#### Boolean Data

##### Feature engineering
##### Spread of frequency
- Mean Medium Q25 Q50 Skew

### Quantile-based discretization

- Helps in capturing the inherent variability within the data
- Reduce noise and focusing on broader trends rather than individual data points

### Feature engineering

- Convert individual boolean indicators into a more informative ordinal scale
- Simplifies the input for modeling and may reveal patterns more effectively


## Machine Learning

### LinearRegression

- Explore what integer values imposes an effect on counts_kisses
- Explore which integer variable have a stronger correlation with counts_kisses

### Decision Tree

- Explore correlation between boolean value and counts_kisses
- Explore which boolean value have a stronger correlation with counts_kisses

### Chi-test

- Explore correlation between boolean values

## Conclusion

What are the key features to classify the gender of a speaker through their voice?

> According to classification tree analysis, `IQR` and `meanfun` have been identified as the two main predictors for differentiating male and female voices. A higher `IQR` and lower `meanfun` are more indicative of a male speaker.

Which models can predict the gender of a speaker with higher accuracy?

> Among the various models, the SVM model with an RBF kernel achieved the highest accuracy, with a score of 0.9834.

## What We Learnt

- Importance of data preparation
  - The initial lack of normalization has resulted in poor performance of the SVM model. Despite spending significant time adjusting the SVM parameters, the model still showed poor accuracy. However, after performing normalization, we observed a significant improvement in the accuracy of our SVM model.
- Exploring Various Machine Learning Models for Accurate Predictions
  - Supervised learning: Classification Tree, Random Forest, Logistic Regression, K Nearest Neighbour, Support Vector Machines, Multi-Layer Perceptron
  - Unsupervised learning: Principal Component Analysis
  - Use of Cross-Validation to evaluate the accuracy of each model
- Ensemble Vote model

## Group Members

| Name | Email | 
| --- | --- | 
| Wang Yanjie | [WANG2037@e.ntu.edu.sg](mailto:WANG2037@e.ntu.edu.sg) |
| Dai Shiyu | [dais0013@e.ntu.edu.sg](mailto:dais0013@e.ntu.edu.sg) | 

## Reference

Various resources were used to help us gain a better understanding of the project and the various machine learning methods.

1. [DataSet from Kaggle](https://www.kaggle.com/datasets/primaryobjects/voicegender)
2. [R documentation (`specan`)](https://www.rdocumentation.org/packages/warbleR/versions/1.1.2/topics/specan)
   1. Helped us understand the meaning of the features.
   2. Helped us understand how to extract various features from audio signals.
3. [An Introduction to Statistical Learning](https://www.statlearning.com/)
   1. Helped us gain a basic understanding of various supervised learning methods.
   2. Helped us understand Cross Validation.
4. [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
   1. Helped us dive deeper into the theory behind support vector machines.
5. [Learning Materials from Nanyang Technological University](https://ntulearn.ntu.edu.sg/)
   1. Helped us gain a basic understanding of machine learning.
   2. Lab classes guided us to start using Jupyter Notebook.
6. [UC Berkeley Data 100: Principles and Techniques of Data Science](https://ds100.org/)
   1. Enabled us to make further progress in Python programming.
   2. Helped us gain a basic understanding of some machine learning algorithms.
7. [ChatGPT](https://chat.openai.com/)
   1. Can patiently explain to me when I don't understand a specific topic.
   2. Help me debug my code when it's not working properly.
8. [`scikit-learn` documentation](https://scikit-learn.org/stable/)
   1. Helped us understand the usage of various machine learning models.
9. [`pandas` documentation](https://pandas.pydata.org/pandas-docs/stable/)
   1. Helped us understand the usage of various `pandas` functions.
