![](cover123.png)

Welcome to our project for the NTU course *SC1015 Introduction to Data Science and Artificial Intelligence*!

In this project, we explore how to become more attractive on dating app(lovoo).

The main page of our project is [here](https://wangkinga.github.io/).

And our presentation video is [here](https://www.youtube.com/watch?v=ILE9QUTryWs).

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

Insights 

> - Distance seems to play a significant role in user engagement, as suggested by the chi-square test results between distance_category and kisses_category.
> - The levels of expressed flirtatious interest are strongly associated with the likelihood of receiving more 'kisses', an indicator of attractiveness on the platform.
> - The logistic regression analysis highlighted the importance of specific categories within flirt interest and distance_category, quantifying their unique impacts on the likelihood of higher kisses_category.
> - The use of decision trees demonstrated the importance of counts_kisses as a feature, and how different variables interact with it to affect a user's perceived attractiveness.

Sub-problem
- Which variable most effectively indicates the attractiveness of a user on dating apps?
> Counts_kisses

- What variables demonstrate a strong correlation with the key indicator of attractiveness?
> Profile_visits ， Distance category ， Flirt_interest

## Improvements

- Model exploration
  - Apply other machine learning models that might be better suited for the data characteristics. Neural networks, support vector machines, or ensemble methods may reveal different insights.
  - Use regularization techniques in logistic regression (e.g., Ridge or Lasso) to prevent overfitting and to handle multicollinearity.
- Cross-Validation(CV)
  - Employ cross-validation techniques to assess model stability and reliability, rather than relying on a single train-test split.
  - Use stratified sampling in the cross-validation to maintain the proportion of classes across folds.

 ## Group Members

| Name | Email | Contribution |
| --- | --- | --- | 
| Wang Yanjie | [WANG2037@e.ntu.edu.sg](mailto:WANG2037@e.ntu.edu.sg) | Machine Learning, Conclusion, Slides , Script |
| Dai Shiyu | [dais0013@e.ntu.edu.sg](mailto:dais0013@e.ntu.edu.sg) | Motivation , Problem formulation , Data Preparation , Slides , Script |

## Reference

Various resources were used to help us gain a better understanding of the project and the various machine learning methods.

1. [DataSet from Kaggle](https://www.kaggle.com/datasets/utkarshx27/lovoo-dating-app-dataset)
2. [DataSet from Kaggle](https://www.kaggle.com/datasets/thedevastator/lovoo-v3-dating-app-user-profiles-and-statistics)
3. [Learning Materials from Nanyang Technological University](https://ntulearn.ntu.edu.sg/)
   - Helped us gain a basic understanding of machine learning.
   - Lab classes guided us to start using Jupyter Notebook.
4. [ChatGPT](https://chat.openai.com/)
   - Help us understand the code.
   - Help us debug code when it's not working properly.

