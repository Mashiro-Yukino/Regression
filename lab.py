#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 11:18:35 2022

@author: mu
"""

# %%

import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
# %%

# if name 'df' is not defined, then read the data
if 'df' not in locals():
    df = pd.read_spss('FredData.sav')


print(df.head())
# %%
# make directory
if not os.path.exists('summary'):
    os.mkdir('summary')

if not os.path.exists('GDP_AND_GNP'):
    os.mkdir('GDP_AND_GNP')


# %%
# print all the data
print(df)

# %%


# %%
# question 1

#
# make a new folder called summary
# make summary of all data and save it to a txt file and save it to the summary folder
df.describe().to_csv('summary/summary.txt', sep='\t', encoding='utf-8')

# make a correlation matrix and save it to a txt file and save it to the summary folder
corrmat = df.corr()
sns.heatmap(corrmat, vmax=.8, square=True)
plt.savefig('summary/correlation.png', bbox_inches='tight', dpi=300)


# %%
# question 3a
#       Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
#                   |-----:-----|
#   o      |--------|     :     |--------|    o  o
#                   |-----:-----|
# flier             <----------->            fliers
#                         IQR
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html#matplotlib.pyplot.boxplot
# using matplotlib to draw two boxplots -- GDP and GNP

df.boxplot(column=['GDP', 'GNP'], vert=False)

# give a title as "GDP vs GNP", and set the dpi to 300
plt.title('GDP vs GNP')


# save the figure as a png file, named as GDP_GNP_boxplot.png
plt.savefig('GDP_AND_GNP/GDP_GNP_boxplot.png', dpi=300)


# %%
# question 3b
# draw the scatter plot of GDP and GNP, gdp is on the x-axis, gnp is on the y-axis
plt.scatter(df['GDP'], df['GNP'])

# give a title as "GDP vs GNP", and write the x-axis and y-axis labels
plt.title('GDP vs GNP')
plt.xlabel('GDP')
plt.ylabel('GNP')

# save the figure as a png file, named as GDP_GNP_scatterplot.png
plt.savefig('GDP_AND_GNP/GDP_GNP_scatterplot.png', dpi=300)


# %%

# check missing values in all columns
print(df.isnull().sum())

# display the number of missing values in bar chart in percentage
# first find the total number of rows
total_rows = df.shape[0]

print(total_rows)

# find the number of missing values in each column in percentage
missing_values = df.isnull().sum() / total_rows

# plot the bar chart， set the title as "Missing Values in Each Column"
# from high to low
missing_values.sort_values(ascending=False).plot(kind='bar', title='Missing Values in Each Column',
                                                 color=sns.color_palette('coolwarm', len(missing_values)))

# y-axis label is "Percentage of Missing Values"
plt.ylabel('Percentage of Missing Values')

# x axis label is "Column Name"
plt.xlabel('Column Name')

# create a folder called "missing_values" if it does not exist
# save the figure as a png file, named as missing_values.png in the missing_values folder

if not os.path.exists('missing_values'):
    os.mkdir('missing_values')


plt.savefig('missing_values/missing_values.png', bbox_inches='tight', dpi=300)


# %%

sns.displot(df['GDP'])

# save the figure as a png file, named as GDP_distribution_before_normalization.png
# in the GDP_AND_GNP folder
plt.savefig('GDP_AND_GNP/GDP_distribution_before_normalization.png',
            bbox_inches='tight', dpi=300)

# find all non-finite values in the GDP column
print(df['GDP'].isin([float('inf'), float('-inf'), float('nan')]))

# print all values in the GDP column
print(df['GDP'])

# make a copy of the GDP column, remove all non-finite values
GDP_copy = df['GDP'].copy()
GDP_copy = GDP_copy[~GDP_copy.isin(
    [float('inf'), float('-inf'), float('nan')])]

# %%

#  draw the Probability plot of GDP_copy
stats.probplot(GDP_copy, plot=plt)
plt.title('Probability plot of GDP')


plt.savefig('GDP_AND_GNP/GDP_probability_plot.png',
            bbox_inches='tight', dpi=300)


# %%


sns.displot(df['GNP'])

# %%
# make a new folder called "pictures" to store all the pictures
if not os.path.exists('pictures'):
    os.mkdir('pictures')

# scan all the subfolders in the current folder, if the file is a png file
# copy them to the pictures folder
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.png'):
            os.system(f'cp {root}/{file} pictures/{file}')
