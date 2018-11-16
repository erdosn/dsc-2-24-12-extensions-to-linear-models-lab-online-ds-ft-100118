
# Extensions to Linear Models - Lab

## Introduction

In this lab, you'll practice many concepts learned in this section, from adding interactions and polynomials to your model to AIC and BIC!

## Summary

You will be able to:
- Build a linear regression model with polynomial features/interactions
- Perform regularization
- Use AIC and BIC to select the best value for the regularization parameter


## Let's get started!

Import all the necessary packages.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from itertools import combinations

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
```

## Look at a Baseline Boston Housing Data Model

Import the Boston housing data set, use all the predictors in their scaled version (using `preprocessing.scale`. Look at a baseline model using *scaled variables* as predictors. Use 5-fold cross-validation this time and use the $R^2$ score to evaluate the model.


```python
load_boston().keys()
```




    dict_keys(['data', 'target', 'feature_names', 'DESCR'])




```python
# Your code here
X = load_boston().data
y = load_boston().target
features = load_boston().feature_names
```


```python
X_scaled = preprocessing.scale(X)
```


```python
linreg = LinearRegression()
crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)

cross_val_scaled = cross_val_score(linreg, X_scaled, y, scoring='r2', n_jobs=-1, cv=crossvalidation)
cross_val_reg = cross_val_score(linreg, X, y, scoring='r2', n_jobs=-1, cv=crossvalidation)

baseline_scaled = np.mean(cross_val_scaled)
baseline_reg = np.mean(cross_val_reg)

print(baseline_reg, baseline_scaled)
```

    0.7176324491383014 0.7176324491383005


## Include interactions

Look at all the possible combinations of variables for interactions by adding interactions one by one to the baseline model. Next, evaluate that model using 5-fold classification and store the $R^2$ to compare it with the baseline model.

You've created code for this before in the interactions lab, yet this time, you have scaled the variables so the outcomes may look different. 

Print the 7 most important interactions.


```python
df_reg = pd.DataFrame(X, columns=features)
df_scaled = pd.DataFrame(X_scaled, columns=features)
y = pd.DataFrame(y, columns=['target'])
df_all = pd.concat([df_scaled, y])
```


```python
feature_combinations = list(combinations(features, 2))
feature_combinations[:3]
```




    [('CRIM', 'ZN'), ('CRIM', 'INDUS'), ('CRIM', 'CHAS')]




```python
# Your code here
"""
Look at all the possible combinations of variables for interactions by adding 
interactions one by one to the baseline model. Next, evaluate that model using 5-fold 
classification and store the  R2R2  to compare it with the baseline model.
"""
interactions = []
data = df_scaled.copy()
for comb in feature_combinations:
    data["interaction"] = data[comb[0]] * data[comb[1]]
    score = np.mean(cross_val_score(linreg, data, y, scoring="r2", cv=crossvalidation))
    if score > baseline_scaled: 
        interactions.append((comb[0], comb[1], round(score,3)))
            
print("Top 7 interactions: %s" %sorted(interactions, key=lambda inter: inter[2], reverse=True)[:7])
```

    Top 7 interactions: [('RM', 'LSTAT', 0.783), ('RM', 'TAX', 0.775), ('RM', 'RAD', 0.77), ('RM', 'PTRATIO', 0.764), ('INDUS', 'RM', 0.757), ('NOX', 'RM', 0.746), ('RM', 'AGE', 0.742)]



```python
print("Top 7 interactions: %s" %sorted_interactions[:7])
```

    Top 7 interactions: [('RM', 'LSTAT', 0.783), ('RM', 'TAX', 0.775), ('RM', 'RAD', 0.77), ('RM', 'PTRATIO', 0.764), ('INDUS', 'RM', 0.757), ('NOX', 'RM', 0.746), ('RM', 'AGE', 0.742)]



```python
print("Top 7 interactions: {}".format(sorted_interactions[:7]))
```

    Top 7 interactions: [('RM', 'LSTAT', 0.783), ('RM', 'TAX', 0.775), ('RM', 'RAD', 0.77), ('RM', 'PTRATIO', 0.764), ('INDUS', 'RM', 0.757), ('NOX', 'RM', 0.746), ('RM', 'AGE', 0.742)]


Write code to include the 7 most important interactions in your data set by adding 7 columns. Name the columns "var1_var2" with var1 and var2 the two variables in the interaction.


```python
# Your code here
sorted_interactions = sorted(interactions, key=lambda inter: inter[2], reverse=True)
df_scaled_new = df_scaled.copy()
count = 0
limit = 7
for interaction in sorted_interactions:
    col_name = '_'.join([interaction[0], interaction[1]])
    df_scaled_new[col_name] = df_scaled_new[interaction[0]] * df_x_scaled_new[interaction[1]]
    count += 1
    if count == 7:
        break
```


```python
df_scaled_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>RM_LSTAT</th>
      <th>RM_TAX</th>
      <th>RM_RAD</th>
      <th>RM_PTRATIO</th>
      <th>INDUS_RM</th>
      <th>NOX_RM</th>
      <th>RM_AGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.417713</td>
      <td>0.284830</td>
      <td>-1.287909</td>
      <td>-0.272599</td>
      <td>-0.144217</td>
      <td>0.413672</td>
      <td>-0.120013</td>
      <td>0.140214</td>
      <td>-0.982843</td>
      <td>-0.666608</td>
      <td>-1.459000</td>
      <td>0.441052</td>
      <td>-1.075562</td>
      <td>-0.444930</td>
      <td>-0.275757</td>
      <td>-0.406574</td>
      <td>-0.603547</td>
      <td>-0.532772</td>
      <td>-0.059659</td>
      <td>-0.049646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.415269</td>
      <td>-0.487722</td>
      <td>-0.593381</td>
      <td>-0.272599</td>
      <td>-0.740262</td>
      <td>0.194274</td>
      <td>0.367166</td>
      <td>0.557160</td>
      <td>-0.867883</td>
      <td>-0.987329</td>
      <td>-0.303094</td>
      <td>0.441052</td>
      <td>-0.492439</td>
      <td>-0.095668</td>
      <td>-0.191813</td>
      <td>-0.168607</td>
      <td>-0.058883</td>
      <td>-0.115279</td>
      <td>-0.143814</td>
      <td>0.071331</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.415272</td>
      <td>-0.487722</td>
      <td>-0.593381</td>
      <td>-0.272599</td>
      <td>-0.740262</td>
      <td>1.282714</td>
      <td>-0.265812</td>
      <td>0.557160</td>
      <td>-0.867883</td>
      <td>-0.987329</td>
      <td>-0.303094</td>
      <td>0.396427</td>
      <td>-1.208727</td>
      <td>-1.550451</td>
      <td>-1.266461</td>
      <td>-1.113245</td>
      <td>-0.388783</td>
      <td>-0.761138</td>
      <td>-0.949544</td>
      <td>-0.340960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.414680</td>
      <td>-0.487722</td>
      <td>-1.306878</td>
      <td>-0.272599</td>
      <td>-0.835284</td>
      <td>1.016303</td>
      <td>-0.809889</td>
      <td>1.077737</td>
      <td>-0.752922</td>
      <td>-1.106115</td>
      <td>0.113032</td>
      <td>0.416163</td>
      <td>-1.361517</td>
      <td>-1.383713</td>
      <td>-1.124148</td>
      <td>-0.765197</td>
      <td>0.114875</td>
      <td>-1.328183</td>
      <td>-0.848901</td>
      <td>-0.823092</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.410409</td>
      <td>-0.487722</td>
      <td>-1.306878</td>
      <td>-0.272599</td>
      <td>-0.835284</td>
      <td>1.228577</td>
      <td>-0.511180</td>
      <td>1.077737</td>
      <td>-0.752922</td>
      <td>-1.106115</td>
      <td>0.113032</td>
      <td>0.441052</td>
      <td>-1.026501</td>
      <td>-1.261136</td>
      <td>-1.358947</td>
      <td>-0.925023</td>
      <td>0.138869</td>
      <td>-1.605599</td>
      <td>-1.026210</td>
      <td>-0.628023</td>
    </tr>
  </tbody>
</table>
</div>



## Include Polynomials

Try polynomials of 2, 3 and 4 for each variable, in a similar way you did for interactions (by looking at your baseline model and seeing how $R^2$ increases). Do understand that when going for a polynomial of 4, the particular column is raised to the power of 2 and 3 as well in other terms. We only want to include "pure" polynomials, so make sure no interactions are included. We want the result to return a list that contain tuples of the form:

`(var_name, degree, R2)`, so eg. `('DIS', 3, 0.732)`


```python
# Your code here
polynomials = []
for col in df_reg.columns:
    for degree in [2,3,4]:
        data = df_scaled.copy()
        poly = PolynomialFeatures(degree, include_bias=False)
        X = poly.fit_transform(df_reg[[col]])
        data = pd.concat([data.drop(col, axis=1),pd.DataFrame(X)], axis = 1)
        score = np.mean(cross_val_score(linreg, data, y, scoring="r2", cv=crossvalidation))
        if score > baseline_scaled: 
            polynomials.append((col, degree, round(score,3)))
print("Top 10 polynomials: %s" %sorted(polynomials, key=lambda poly: poly[2], reverse=True)[:10])
```

    Top 10 polynomials: [('RM', 4, 0.8), ('RM', 2, 0.782), ('LSTAT', 4, 0.782), ('RM', 3, 0.781), ('LSTAT', 3, 0.774), ('LSTAT', 2, 0.772), ('DIS', 3, 0.737), ('DIS', 2, 0.732), ('DIS', 4, 0.731), ('TAX', 4, 0.724)]


For each variable, print out the maximum R2 possible when including Polynomials.


```python
# Your code here
polynom = pd.DataFrame(polynomials)
polynom.groupby([0], sort=False)[2].max()
```




    0
    ZN         0.723
    INDUS      0.723
    CHAS       0.718
    NOX        0.721
    RM         0.800
    AGE        0.722
    DIS        0.737
    RAD        0.719
    TAX        0.724
    PTRATIO    0.721
    B          0.720
    LSTAT      0.782
    Name: 2, dtype: float64



Which two variables seem to benefit most from adding Polynomial terms?

Add Polynomials for the two features that seem to benefit the most, as in have the best R squared compared to the baseline model. For each of the two feature, raise to the Polynomial that generates the best result. Make sure to start from the data set `df_inter` so the final data set has both interactions and polynomials in the model.


```python
# Your code here
for col in ["RM", "LSTAT"]:
    poly = PolynomialFeatures(4, include_bias=False)
    X = poly.fit_transform(df_reg[[col]])
    colnames= [col, col+"_"+"2", col+"_"+"3", col+"_"+"4"]
    df_scaled_new = pd.concat([df_scaled_new.drop(col, axis=1),pd.DataFrame(X, columns=colnames)], axis = 1)
```

check out your final data set and make sure that your interaction terms as well as your polynomial terms are included.


```python
# Your code here
df_scaled_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>...</th>
      <th>NOX_RM</th>
      <th>RM_AGE</th>
      <th>RM</th>
      <th>RM_2</th>
      <th>RM_3</th>
      <th>RM_4</th>
      <th>LSTAT</th>
      <th>LSTAT_2</th>
      <th>LSTAT_3</th>
      <th>LSTAT_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.417713</td>
      <td>0.284830</td>
      <td>-1.287909</td>
      <td>-0.272599</td>
      <td>-0.144217</td>
      <td>-0.120013</td>
      <td>0.140214</td>
      <td>-0.982843</td>
      <td>-0.666608</td>
      <td>-1.459000</td>
      <td>...</td>
      <td>-0.059659</td>
      <td>-0.049646</td>
      <td>6.575</td>
      <td>43.230625</td>
      <td>284.241359</td>
      <td>1868.886938</td>
      <td>4.98</td>
      <td>24.8004</td>
      <td>123.505992</td>
      <td>615.059840</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.415269</td>
      <td>-0.487722</td>
      <td>-0.593381</td>
      <td>-0.272599</td>
      <td>-0.740262</td>
      <td>0.367166</td>
      <td>0.557160</td>
      <td>-0.867883</td>
      <td>-0.987329</td>
      <td>-0.303094</td>
      <td>...</td>
      <td>-0.143814</td>
      <td>0.071331</td>
      <td>6.421</td>
      <td>41.229241</td>
      <td>264.732956</td>
      <td>1699.850313</td>
      <td>9.14</td>
      <td>83.5396</td>
      <td>763.551944</td>
      <td>6978.864768</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.415272</td>
      <td>-0.487722</td>
      <td>-0.593381</td>
      <td>-0.272599</td>
      <td>-0.740262</td>
      <td>-0.265812</td>
      <td>0.557160</td>
      <td>-0.867883</td>
      <td>-0.987329</td>
      <td>-0.303094</td>
      <td>...</td>
      <td>-0.949544</td>
      <td>-0.340960</td>
      <td>7.185</td>
      <td>51.624225</td>
      <td>370.920057</td>
      <td>2665.060607</td>
      <td>4.03</td>
      <td>16.2409</td>
      <td>65.450827</td>
      <td>263.766833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.414680</td>
      <td>-0.487722</td>
      <td>-1.306878</td>
      <td>-0.272599</td>
      <td>-0.835284</td>
      <td>-0.809889</td>
      <td>1.077737</td>
      <td>-0.752922</td>
      <td>-1.106115</td>
      <td>0.113032</td>
      <td>...</td>
      <td>-0.848901</td>
      <td>-0.823092</td>
      <td>6.998</td>
      <td>48.972004</td>
      <td>342.706084</td>
      <td>2398.257176</td>
      <td>2.94</td>
      <td>8.6436</td>
      <td>25.412184</td>
      <td>74.711821</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.410409</td>
      <td>-0.487722</td>
      <td>-1.306878</td>
      <td>-0.272599</td>
      <td>-0.835284</td>
      <td>-0.511180</td>
      <td>1.077737</td>
      <td>-0.752922</td>
      <td>-1.106115</td>
      <td>0.113032</td>
      <td>...</td>
      <td>-1.026210</td>
      <td>-0.628023</td>
      <td>7.147</td>
      <td>51.079609</td>
      <td>365.065966</td>
      <td>2609.126456</td>
      <td>5.33</td>
      <td>28.4089</td>
      <td>151.419437</td>
      <td>807.065599</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



## Full model R-squared

Check out the R-squared of the full model.


```python
# Your code here
full_model = np.mean(cross_val_score(linreg, df_scaled_new, y, scoring="r2", cv=crossvalidation))
full_model
```




    0.8061116489236934



## Finding the best Lasso regularization parameter

You've learned that, when using Lasso regularization, your coefficients shrink to 0 when using a higher regularization parameter. Now the question is which value we should choose for the regularization parameter. 

This is where the AIC and BIC come in handy! We'll use both criteria in what follows and perform cross-validation to select an optimal value of the regularization parameter alpha of the Lasso estimator.

Read the page here: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html and create a similar plot as the first one listed on the page. 


```python
# Your code here
model_bic = LassoLarsIC(criterion='bic')
model_bic.fit(df_scaled_new, y)
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(df_scaled_new, y)
alpha_aic_ = model_aic.alpha_


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color, linewidth=2, label= name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=2,
                label='alpha for %s ' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'green')
plot_ic_criterion(model_bic, 'BIC', 'blue')
plt.legend()
plt.title('Information-criterion for model selection')
```




    Text(0.5,1,'Information-criterion for model selection')




![png](index_files/index_36_1.png)


## Analyze the final result

Finally, use the best value for regularization parameter according to AIC and BIC and compare the R squared parameters and MSE using train-test-split. Compare with the baseline model.


```python
# Code for baseline model
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

linreg_all = LinearRegression()
linreg_all.fit(X_train, y_train)
print('Training r^2:', linreg_all.score(X_train, y_train))
print('Testing r^2:', linreg_all.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, linreg_all.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, linreg_all.predict(X_test)))
```

    Training r^2: 0.7632796668701112
    Testing r^2: 0.6659539974224437
    Training MSE: 19.324759756564426
    Testing MSE: 30.96213268290194



```python
# code for lasso with alpha from AIC
X_train, X_test, y_train, y_test = train_test_split(df_scaled_new, y)

lasso = Lasso(alpha= model_aic.alpha_) 
lasso.fit(X_train, y_train)
print('Training r^2:', lasso.score(X_train, y_train))
print('Testing r^2:', lasso.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, lasso.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, lasso.predict(X_test)))
```

    Training r^2: 0.8302298227240487
    Testing r^2: 0.7609996233510434
    Training MSE: 14.205620500619458
    Testing MSE: 20.63798229284935



```python
# code for lasso with alpha from BIC
lasso = Lasso(alpha= model_bic.alpha_) 
lasso.fit(X_train, y_train)
print('Training r^2:', lasso.score(X_train, y_train))
print('Testing r^2:', lasso.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, lasso.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, lasso.predict(X_test)))
```

    Training r^2: 0.8292938475138585
    Testing r^2: 0.7659128201888941
    Training MSE: 14.283938782706995
    Testing MSE: 20.213721583462355


## Level Up - Optional

### Create a Lasso Path

From this section, you know that when using lasso, more parameters shrink to zero as your regularization parameter goes up. In Scikit-Learn there is a function lasso_path which visualizes the shrinkage of the coefficients while alpha changes. Try this out yourself!

https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py

### AIC and BIC for subset selection
This notebook shows how you can use AIC and BIC purely for feature selection. Try this code out on our Boston Housing data!

https://xavierbourretsicotte.github.io/subset_selection.html

## Summary

Congratulations! You now know how to create better linear models and how to use AIC and BIC for both feature selection and to optimize your regularization parameter when performing Ridge and Lasso. 
