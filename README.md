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

### Question 1 : 

#### Methods

Let’s get a basic summary of some of the data

- using **pandas.DataFrame.describe** to generate descriptive statistics.
- using **pandas.DataFrame.skew** to get skewness
- calculate **SES**  [[2]](#2) using the following formula.

$$S E_{\text {skew }}=\sqrt{\frac{6 \cdot n \cdot(n-1)}{(n-2) \cdot(n+1) \cdot(n+3)}}$$


- using **pandas.DataFrame.kurt** to generate Kurtosis Statistic

- calculate **standard error of kurtosis**  [[2]](#2) using the following formula.
$$S E_{k}=\sqrt{\frac{\left(4 \cdot N^{2}-1\right) \cdot(N-1)}{(N-2)(N+1)(N+3)}}$$


#### Result
The summary we get is :
https://github.com/Mashiro-Yukino/Regression/blob/main/1/summary.csv





## References : 
<a id="1">[1]</a> 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_spss.html


<a id="2">[2]</a> 
https://www.stattutorials.com/EXCEL/EXCEL-DESCRIPTIVE-STATISTICS.html
