# <p align="center">Regression

## Preparation : 

### To install Python:
-	Navigate to https://www.anaconda.com/products/distribution and install for your operating system!

- On the Anaconda dashboard, open Spyder. This is an augmented version of python with additional packages that we will utilize

- Initialize your file by importing the data processing library attached as dp

### Read data:

using pandas to read spss file [[1]](#1). (both **Lab1a.sav** and **FredData.sav**)

```py
pandas.read_spss(path, usecols=None, convert_categoricals=True)
```


### Questions Description

The data for problems 1-4 comes from FRED (the St. Louis Federal Reserve Bank). 

## Regression related questions

### 1: Let’s get a basic summary of some of the data:

#### Methods

- Use **pandas.DataFrame.describe** to generate descriptive statistics.
- Use **pandas.DataFrame.skew** to get skewness
- Calculate `SES`   [[2]](#2) using the following formula.

$$S E_{\text {skew }}=\sqrt{\frac{6 \cdot n \cdot(n-1)}{(n-2) \cdot(n+1) \cdot(n+3)}}$$


- Use **pandas.DataFrame.kurt** to generate Kurtosis Statistic

- Calculate `standard error of kurtosis` [[2]](#2) using the following formula.
$$S E_{k}=\sqrt{\frac{\left(4 \cdot N^{2}-1\right) \cdot(N-1)}{(N-2)(N+1)(N+3)}}$$





#### Result
The whole summary we get is :
https://github.com/Mashiro-Yukino/Regression/blob/main/1/summary.csv


### 1.1: Decide which variables are normal?

> Aside: kurtosis and skewness are used to check if the data is normal—both should be close to 0 if the data is normal; kurtosis measures how close to the center the data is, a positive value says the data is clustered more tightly than if it was normal; skewness measures how symmetric the data is, a positive value says the data is skewed right. A rough guideline is the data is ok if the statistic is less than double the standard error. Does it appear that any of the variables are normal using this criteria? If so, list them.

#### Methods

- From the above summary, extract the kurtosis_statistic, skewness_statistic, kurtosis_standard_error, skewness_standard_error for each variables.

- Check if the absolute value of the `kurtosis_statistic` is less than 2 times the `kurtosis_standard_error` and if the absolute value of the skewness statistic is less than 2 times the `skewness_standard_error`. If both conditions are met, we consider that variables is ’Normal’, otherwise, we consider it is 'Not Normal'.


#### Result
The variables **Personal Disposable Income** and **Median Weekly Earnings** appear to be normal.

```
    '''
    Output :

        The following columns are normal:
        PersonalDisposableIncome
        MedianEarnings

    '''
```


### 2: More Summary
There are a few ways to look at the summaries of the data. Let’s use a second here:

#### Method

- Use **pandas.DataFrame.describe** to generate descriptive statistics.
- Calculate the `range` by finding the difference between the min and max values 
- Calculate the length of `missing value` by the difference between the dataframe length and the length of valid value.

#### Result

The whole summary we get is : https://github.com/Mashiro-Yukino/Regression/blob/main/2/further_summary.csv



### 3: Let’s get a few basic graphs
#### 3a: The histogram for one of the variables


##### Method

- Pass in a variable that you want to choose
- Use **pandas.DataFrame.mean**, **pandas.DataFrame.std** to calculate `mean` and `std` of the variable.
- draw histogram by using **matplotlib.pyplot.hist**, first we first input `df['GDP']` to make a simple Histogram of GDP.
- add the title and axis labels using  **plt.title**, **plt.xlabel**, **plt.ylabel**.
- Use **plt.text** to add mean and std to the histogram.

##### Result

![GDP_histogram](https://user-images.githubusercontent.com/67991315/206505583-89065bef-8f3d-4489-ab2c-631d7d82a6ae.png)


The graph has one main peak at about 3. It is skewed a bit to the right and has no obvious outliers. It does appear to be roughly normal. 

#### 3a: Now choose one more variable and get its histogram.

##### Method
- Repeat what we did earlier, but change `df['GDP']` to another variable. For example, this time we use `df['Loans']`.
- **Remember**, different `Bin Size` can change the shape of the histogram slightly. So if the skewness of the histogram is not obvious, we generally don't need to worry about it. To explain what we represent, my results below will show two different bin values.



##### Result

###### case 1 (bins=23) : 
![CommercialAndIndustrialLoans_histogram](https://user-images.githubusercontent.com/67991315/206506871-2a8f1112-189b-4395-9764-5b36b103e1e7.png)


This graph also has one main peak (at about 10), is skewed a bit to the left, and has no obvious outliers. It appears to be roughly normal.

###### case 1 (bins=20) : 
![CommercialAndIndustrialLoans_histogram](https://user-images.githubusercontent.com/67991315/206509675-96814017-882d-4d01-aac9-feedf8b0c8dd.png)

This time the shape is changed a little bit.



#### 3b: Get a boxplot to compare GDP and GNP:


##### Method
- Use `pandas.DataFrame.boxplot` to draw boxplot and pass `['GDP', 'GNP']` to `column`.
- Add the title and axis labels using  **plt.title**, **plt.xlabel**, **plt.ylabel**.


##### Result

![GDP_GNP_boxplot](https://user-images.githubusercontent.com/67991315/206512056-2fdd3eac-84ab-4639-b131-11f0ea3f7689.png)



#### 3c: Let’s now get a scatterplot for two variables:

##### Method
- Use `matplotlib.pyplot.scatter` to draw scatterplot and pass  `df["GDP"]` and `df["GNP"]` to `x` and `y`.
- Add the title and axis labels using  **plt.title**, **plt.xlabel**, **plt.ylabel**.
- Add grid lines using `plt.grid`.



##### Result

![GDP_GNP_scatterplot](https://user-images.githubusercontent.com/67991315/206512863-6c6172d6-fb67-4f75-9f39-4ceb95a602e0.png)


#### 3c: Choose one more pair of variables and get a scatterplot for them. Hand in these two graphs.

##### Method
- Repeat what we did earlier, but change `df["GDP"]` and `df["GNP"]` to another variables. For example, this time we use `df["EmploymentRate"]` and `df["UnemploymentRate"]`.

##### Result

![EmploymentRate_UnemploymentRate_scatterplot](https://user-images.githubusercontent.com/67991315/206513631-7494a474-ae90-41f2-ad1e-d8a32d3c5ba0.png)

### 4: Let’s try some regression.

First we will do multiple linear regression with the Deficit as the dependent variable. We’ll try this using two
methods:

#### 4a: First get a table of the pairwise correlations to see which variables are strongly correlated:
##### Method
- Ignore the `Date` variable.
- Get `Pearson correlation coefficient` and `p-value` to  find the correlation between each of the two variables using **scipy.stats.pearsonr**.
- Using **Python min()** to find `N`.
- For each cell of the table, use **for loop** to combine needed informations for each cell in the table with **'/'**.
The final format is `Pearson Correlation/ Sig. (2-tailed)/ N`.

##### Result

The whole summary we get is : https://github.com/Mashiro-Yukino/Regression/blob/main/4/pairwise_correlation_with_p_value.csv




#### 4b: The first method will decide which variables make sense to use in the regression: it might make sense to not use some because of what they are, while others have a strong relation with other variables so it doesn’t make sense to use all of them.

##### Method (ANOVA part)
- Use **statsmodels.regression.linear_model.OLS** and pass `Dependent Variable` into `y` and the other variables you chose into the `x`. In this case we choose `Deficit` as Dependent Variable and `GDP`, `GovernmentExpenditure`, `PersonalDisposableIncome`, `CPI` and `UnemploymentRate` as Independent variables.
    
- Get summary by using **statsmodels.stats.anova.anova_lm**.
- Calculate `Total Sum of Squares`, `Total Degrees of Freedom` using **sum()** function.
- Calculate the remaining values by referring to the image below [[3]](#3).
    
<img width="511" alt="6F268922-24DA-4F77-9124-3EE0058CAEDA" src="https://user-images.githubusercontent.com/67991315/206726131-162879d0-e195-4d57-ab95-f735cee01977.png">


##### Result (ANOVA part)
    
The whole summary we get is : https://github.com/Mashiro-Yukino/Regression/blob/main/4/b/anova_table.csv
    
    
    
    
##### Method (Coefficients part)
- Extract coef, std err, t, and sig. information from **statsmodels .summary()**.
- Calculate the Standardized Coefficients using the following formula.

![stb](https://user-images.githubusercontent.com/67991315/206727738-c0bc8b49-720e-4cfe-8be3-fff604f4b4ae.png)



##### Result (Coefficients part)
    
The whole summary we get is : https://github.com/Mashiro-Yukino/Regression/blob/main/4/Deficit_anova_Coefficient.csv 

    

## References : 
<a id="1">[1]</a> 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_spss.html


<a id="2">[2]</a> 
https://www.stattutorials.com/EXCEL/EXCEL-DESCRIPTIVE-STATISTICS.html
    
<a id="3">[3]</a> 
https://online.stat.psu.edu/stat415/book/export/html/822
    
    
    
<a id="4">[4]</a> 
https://www.analyticsvidhya.com/blog/2021/03/standardized-vs-unstandardized-regression-coefficient/
