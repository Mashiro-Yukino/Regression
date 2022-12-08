# Regression

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

- Using **pandas.DataFrame.describe** to generate descriptive statistics.
- Using **pandas.DataFrame.skew** to get skewness
- Calculate `SES`   [[2]](#2) using the following formula.

$$S E_{\text {skew }}=\sqrt{\frac{6 \cdot n \cdot(n-1)}{(n-2) \cdot(n+1) \cdot(n+3)}}$$


- Using **pandas.DataFrame.kurt** to generate Kurtosis Statistic

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

- Using **pandas.DataFrame.describe** to generate descriptive statistics.
- Calculate the `range` by finding the difference between the min and max values 
- Calculate the length of `missing value` by the difference between the dataframe length and the length of valid value.

#### Result

The whole summary we get is : https://github.com/Mashiro-Yukino/Regression/blob/main/2/further_summary.csv



### 3: Let’s get a few basic graphs
#### 3a: The histogram for one of the variables


##### Method

- Pass in a variable that you want to choose
- Using **pandas.DataFrame.mean**, **pandas.DataFrame.std** to calculate `mean` and `std` of the variable.
- draw histogram by using **matplotlib.pyplot.hist**, first we first input `df['GDP']` to make a simple Histogram of GDP.
- add the title and axis labels using  **plt.title**, **plt.xlabel**, **plt.ylabel**.
- Using **plt.text** to add mean and std to the histogram.

##### Result

![GDP_histogram](https://user-images.githubusercontent.com/67991315/206498805-287492e9-0cea-4723-a165-2c388fabcd00.png)






## References : 
<a id="1">[1]</a> 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_spss.html


<a id="2">[2]</a> 
https://www.stattutorials.com/EXCEL/EXCEL-DESCRIPTIVE-STATISTICS.html
