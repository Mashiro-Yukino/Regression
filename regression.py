import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

import numpy as np
import seaborn as sns

FRED = 'FredData.sav'
LAB = 'lab1a.sav'


def read_file(filename):
    # pip install pyreadstat
    df = pd.read_spss(filename)

    return df


# ------- 1 ----------

def standard_error_of_skewness_for_whole_dataset(data):
    ses_dict = {}
    length_of_columns = len(data.columns)
    for i in range(length_of_columns):
        ses_dict[data.columns[i]
                 ] = formula_for_standard_error_of_skewness(data, i)

    return ses_dict


def formula_for_standard_error_of_skewness(data, index):
    # formula for standard error of skewness
    # https://www.stattutorials.com/EXCEL/EXCEL-DESCRIPTIVE-STATISTICS.html
    # SES=sqrt(6*N*(N-1)/(N-1)*(N+1)*(N+3))

    # find the length of the column (only count the non-NaN values)
    length_for_specific_column = len(data.iloc[:, index].dropna())
    ses_square = (6 * length_for_specific_column * (length_for_specific_column - 1)) / (
        (length_for_specific_column - 2) * (length_for_specific_column + 1) * (length_for_specific_column + 3))
    ses = ses_square ** 0.5

    # store as the four decimal places
    ses = round(ses, 3)

    return ses


def standard_error_of_kurtosis_for_whole_dataset(data):
    sec_dict = {}
    length_of_columns = len(data.columns)

    for i in range(length_of_columns):
        sec_dict[data.columns[i]
                 ] = formula_of_standard_error_of_kurtosis(data, i)

    return sec_dict


def formula_of_standard_error_of_kurtosis(data, index):
    # formula for standard error of skewness
    # https://www.stattutorials.com/EXCEL/EXCEL-DESCRIPTIVE-STATISTICS.html
    # the standard error for kurtosis is =SQRT(4*(N^2-1)*V_skew / ((N-3)*(N+5)))

    # find the length of the column (only count the non-NaN values)
    length_for_specific_column = len(data.iloc[:, index].dropna())

    # find the variance (squared standard error) of the skewness statistic for specific column
    skewness_statistic = (6 * length_for_specific_column * (length_for_specific_column - 1)) / (
        (length_for_specific_column - 2) * (length_for_specific_column + 1) * (length_for_specific_column + 3))

    # get the name of the column
    column_name = data.columns[index]
    sek_square = (4 * (length_for_specific_column ** 2 - 1) * skewness_statistic) / (
        (length_for_specific_column - 3) * (length_for_specific_column + 5))
    sek = sek_square ** 0.5

    # store as the four decimal places
    sek = round(sek, 3)

    return sek


def make_summary(df):
    # make summary statistics
    summary = df.describe()

    # also add Skewness-Statistic, Skewness-standard error, Kurtosis-Statistic, Kurtosis-standard error

    summary.loc['Skewness Statistic'] = df.skew()

    # add standard error of skewness
    ses = standard_error_of_skewness_for_whole_dataset(df)
    summary.loc['Skewness standard error'] = ses

    summary.loc['Kurtosis Statistic'] = df.kurt()

    # add standard error of kurtosis
    sek = standard_error_of_kurtosis_for_whole_dataset(df)
    summary.loc['Kurtosis standard error'] = sek

    # drop the 25%, 50%, 75% rows
    summary = summary.drop(['25%', '50%', '75%'])

    # add Valid N (listwise)

    # create a folder named 'output' if it doesn't exist
    if not os.path.exists('1'):
        os.makedirs('1')

    # store summary statistics as csv
    summary.to_csv('1/summary.csv')

    return summary


def decide_normality(summary, fred_data):
    '''
    Output :

        The following columns are normal:
        PersonalDisposableIncome
        MedianEarnings

    '''
    normal_dict = {}
    # for each column, print the mean and standard deviation
    for i in range(len(summary.columns)):
        column_name = summary.columns[i]

        # find the kurtosis and skewness statistic for the column
        kurtosis_statistic = summary.loc['Kurtosis Statistic', column_name]
        skewness_statistic = summary.loc['Skewness Statistic', column_name]

        kurtosis_standard_error = summary.loc['Kurtosis standard error', column_name]
        skewness_standard_error = summary.loc['Skewness standard error', column_name]

        # if the statistic is less than double the standard error, then it is considered normal
        if abs(kurtosis_statistic) < 2 * kurtosis_standard_error and abs(
                skewness_statistic) < 2 * skewness_standard_error:
            normal_dict[column_name] = 'Normal'
        else:
            normal_dict[column_name] = 'Not Normal'

    # print the normal column names
    print("The following columns are normal: ")
    for key, value in normal_dict.items():
        if value == 'Normal':
            print(key)


# ------- 2 ----------

def further_analysis(df):
    # make the summary of 25% and 75% quantile
    summary = df.describe()

    # range equals max - min
    range = summary.loc['max'] - summary.loc['min']

    # total row number

    # find valid and missing N
    valid_n = summary.loc['count']
    missing_n = len(df) - valid_n

    # only keep the 25%, 50%, and 75% rows
    summary = summary.loc[['25%', '50%', '75%']]
    summary.loc['Range'] = range

    # find the median
    median = summary.loc['50%']

    summary.loc['Median'] = median

    summary.loc['Range'] = range
    summary.loc['Valid N'] = valid_n
    summary.loc['Missing N'] = missing_n

    # save the summary as csv in 2 folder
    if not os.path.exists('2'):
        os.makedirs('2')
    summary.to_csv('2/further_summary.csv')


#  ------- 3 ----------

def gdp_histogram(df):
    # make a histogram of GDP
    # find the mean, standard deviation, and valid N
    mean = df['GDP'].mean()
    std = df['GDP'].std()
    # n only counts the non-NaN values
    n = len(df['GDP'].dropna())

    # make a histogram
    plt.hist(df['GDP'], bins=20, color='blue', edgecolor='black', alpha=0.5)

    # add the title and axis labels
    plt.title('Simple Histogram of GDP(% change from previous quarter)')
    plt.xlabel('GDP(% change from previous quartor)')
    plt.ylabel('Frequency')

    # add grid
    plt.grid(True)

    # add text on the right side of the histogram, saying the mean and standard deviation and number of GDP
    plt.text(20, 50, 'mean = ' + str(round(mean, 3)) + "\n" + ' std = ' +
             str(round(std, 4)) + "\n" + ' n = ' + str(n), fontsize=12)

    # save the histogram as png
    if not os.path.exists('3'):
        os.makedirs('3')
    plt.savefig('3/GDP_histogram.png', dpi=300, bbox_inches='tight')

    # show the histogram
    plt.show()


def commercial_and_industrial_loans_histogram(df):
    # make a histogram of commercial and industrial loans
    # find the mean, standard deviation, and valid N
    mean = df['Loans'].mean()
    std = df['Loans'].std()
    # n only counts the non-NaN values
    n = len(df['Loans'].dropna())

    # make a histogram
    plt.hist(df['Loans'], bins=20,
             color='blue', edgecolor='black', alpha=0.5)

    # add the title and axis labels
    plt.title('Simple Histogram of Commercial and Industrial Loans')
    plt.xlabel('Commercial and Industrial Loans')
    plt.ylabel('Frequency')

    # add grid
    plt.grid(True)

    # add text on the right side of the histogram, saying the mean and standard deviation and number of GDP
    plt.text(45, 40, 'mean = ' + str(round(mean, 3)) + "\n" + ' std = ' +
             str(round(std, 4)) + "\n" + ' n = ' + str(n), fontsize=12)

    # save the histogram as png
    if not os.path.exists('3'):
        os.makedirs('3')
    plt.savefig('3/CommercialAndIndustrialLoans_histogram.png',
                dpi=300, bbox_inches='tight')

    # show the histogram
    plt.show()


def gdp_gnp_boxplot(df):
    # make a boxplot of GDP and GNP
    # find the mean, standard deviation, and valid N

    df.boxplot(column=['GDP', 'GNP'], vert=False)

    # give a title as "GDP vs GNP", and set the dpi to 300
    plt.title('GDP vs GNP')

    # add the title and axis labels
    plt.title('Boxplot of GDP and GNP')
    plt.xlabel('GDP and GNP')
    plt.ylabel('Frequency')

    # save the boxplot as png
    if not os.path.exists('3'):
        os.makedirs('3')
    plt.savefig('3/GDP_GNP_boxplot.png', dpi=300, bbox_inches='tight')

    # show the boxplot
    plt.show()


def scatter_plot(df, attribute1, attribute2):
    # make a scatter plot of two attributes
    # find the mean, standard deviation, and valid N
    mean1 = df[attribute1].mean()
    std1 = df[attribute1].std()
    n1 = len(df[attribute1].dropna())

    mean2 = df[attribute2].mean()
    std2 = df[attribute2].std()
    n2 = len(df[attribute2].dropna())

    # make a scatter plot
    plt.scatter(df[attribute1], df[attribute2], color='blue', alpha=0.5)

    # add the title and axis labels
    plt.title('Scatter plot of ' + attribute1 + ' and ' + attribute2)
    plt.xlabel(attribute1)
    plt.ylabel(attribute2)

    # add grid
    plt.grid(True)

    # save the scatter plot as png
    if not os.path.exists('3'):
        os.makedirs('3')
    plt.savefig('3/' + attribute1 + '_' + attribute2 + '_scatterplot.png',
                dpi=300, bbox_inches='tight')

    # show the scatter plot
    plt.show()


#  ------- 4 ----------

def pairwise_correlation(df):
    # find the pairwise correlation
    correlation = df.corr()

    # save the correlation as csv in 4 folder
    if not os.path.exists('4'):
        os.makedirs('4')
    correlation.to_csv('4/pairwise_correlation.csv')

    return correlation


def pearsonr_p_value_table(df):
    # make a new table, ignore the Date column
    p_value_table = pd.DataFrame(index=df.columns[1:], columns=df.columns[1:])

    # fill the table
    for attribute1 in df.columns:
        if attribute1 != 'Date':
            for attribute2 in df.columns:
                # if the attribute is the same, the p-value is blank
                if attribute1 == attribute2:
                    p_value_table[attribute1][attribute2] = ' '
                elif attribute2 != 'Date':
                    p_value_table[attribute2][attribute1] = (round(pearsonr_cal(
                        df, attribute1, attribute2), 3))

    # combine this table with the correlation table, same cell using / to separate
    correlation = pairwise_correlation(df)
    correlation = correlation.round(3)
    correlation = correlation.astype(str)
    p_value_table = p_value_table.astype(str)
    for attribute1 in df.columns:
        for attribute2 in df.columns:
            if attribute1 != 'Date' and attribute2 != 'Date':
                correlation[attribute1][attribute2] = correlation[attribute1][attribute2] + \
                    '/' + p_value_table[attribute1][attribute2]

    # and combine the correlation table with the minimum valid N from two attributes, separated by /
    for attribute1 in df.columns:
        for attribute2 in df.columns:
            if attribute1 != 'Date' and attribute2 != 'Date':
                correlation[attribute1][attribute2] = correlation[attribute1][attribute2] + \
                    '/' + str(min(len(df[attribute1].dropna()),
                                  len(df[attribute2].dropna())))

    # save the correlation as png in 4 folder
    if not os.path.exists('4'):
        os.makedirs('4')
    correlation.to_csv('4/pairwise_correlation_with_p_value.csv')


def pearsonr_cal(df, attribute1, attribute2):
    data_attribute1 = df[attribute1].dropna()
    data_attribute2 = df[attribute2].dropna()

    # calculate Pearson's correlation p-value if they are the same length
    # if they are not the same length, make them the same length
    if len(data_attribute1) == len(data_attribute2):
        correlation, p_value = stats.pearsonr(data_attribute1, data_attribute2)

    else:
        # make them the same length
        data_attribute1 = data_attribute1[:len(data_attribute2)]
        data_attribute2 = data_attribute2[:len(data_attribute1)]

        correlation, p_value = stats.pearsonr(data_attribute1, data_attribute2)

    return p_value


def linear_regression_6(df, attribute1, list_independent):
    # cite1 from https://online.stat.psu.edu/stat415/book/export/html/822

    # do n-way anova test, dependent_variable is the dependent variable and independent_variables are GDP, GovernmentExpenditure, PersonalDisposableIncome, CPI, UnemploymentRate
    anova_results = ols(
        attribute1 + ' ~ GDP + GovernmentExpenditure + PersonalDisposableIncome + CPI + UnemploymentRate',
        data=df).fit()

    # print anova table

    # anova_resultss = anova_lm(anova_results)

    print(anova_results.summary())

    anova_summary = anova_lm(anova_results)
    print('\nANOVA results')
    print(anova_summary)

    # cite2 from https://www.statsmodels.org/dev/anova.html
    # the total sum of squares is the sum of sum_sq in the anova table
    total_sum_of_squares = anova_summary['sum_sq'].sum()

    # the total degrees of freedom is the sum of df in the anova table
    total_degrees_of_freedom = anova_summary['df'].sum()

    table = sm.stats.anova_lm(anova_results, typ=2)

    # only get the residual sum of squares result
    residual_sum_of_squares = table['sum_sq'][5]
    residual_df = table['df'][5]
    residual_mean_square = (residual_sum_of_squares / residual_df)

    # calculate sum of squares regression
    sum_of_squares_regression = total_sum_of_squares - residual_sum_of_squares

    # calculate degrees of freedom regression
    degrees_of_freedom_regression = total_degrees_of_freedom - residual_df

    # calculate mean square regression
    mean_square_regression = sum_of_squares_regression / degrees_of_freedom_regression

    # cite from : https://online.stat.psu.edu/stat415/book/export/html/822
    # calculate f-statistic
    f_regression = mean_square_regression / residual_mean_square

    # calculate p-value
    p_value = stats.f.sf(
        f_regression, degrees_of_freedom_regression, residual_df)

    # make anova table, the attributes are sum_sq, df, Mean Square, F, the rows are the Regression, Residual, Total
    anova_table = pd.DataFrame(index=['Regression', 'Residual', 'Total'], columns=[
        'sum_sq', 'df', 'Mean Square', 'F', 'P'])

    # fill the table
    anova_table['sum_sq']['Regression'] = sum_of_squares_regression
    anova_table['df']['Regression'] = degrees_of_freedom_regression
    anova_table['Mean Square']['Regression'] = mean_square_regression
    anova_table['F']['Regression'] = f_regression

    anova_table['sum_sq']['Residual'] = residual_sum_of_squares
    anova_table['df']['Residual'] = residual_df
    anova_table['Mean Square']['Residual'] = residual_mean_square

    anova_table['sum_sq']['Total'] = total_sum_of_squares
    anova_table['df']['Total'] = total_degrees_of_freedom

    anova_table['P']['Regression'] = p_value

    # save the anova table as png in 4 folder
    if not os.path.exists('4/b'):
        os.makedirs('4/b')
    anova_table.to_csv('4/b/anova_table.csv')

    anova_summary = anova_results.summary()

    # only take the coef, std err, t, and sig. columns from the summary
    anova_summary = anova_summary.tables[1]

    coef = []
    std_err = []
    t = []
    sig = []

    # calculate standardized_coef_beta
    standardized_coef_beta = []

    # get the standardized_coef_beta

    for i in range(1, 7):
        coef.append(anova_summary.data[i][1])
        std_err.append(anova_summary.data[i][2])
        t.append(anova_summary.data[i][3])
        sig.append(anova_summary.data[i][4])

    # make a new table
    anova_table = pd.DataFrame(
        index=["Constant"] + list_independent, columns=['coef', 'std err', 't', 'sig.'])

    # cite from : https://www.analyticsvidhya.com/blog/2021/03/standardized-vs-unstandardized-regression-coefficient/

    # skip the 0th element, which is the constant
    for i in range(1, 6):
        standardized_coef_beta.append(
            float(coef[i]) * (df[list_independent[i - 1]].std() / df[attribute1].std()))

    # fill the table
    anova_table['coef'] = coef
    anova_table['std err'] = std_err
    anova_table['t'] = t
    anova_table['sig.'] = sig

    # the standardized_coef_beta is the last column, the first element is blank
    anova_table['standardized_coef_beta'] = [' '] + standardized_coef_beta

    # save the table as csv in 4 folder
    if not os.path.exists('4'):
        os.makedirs('4')

    anova_table.to_csv('4/' + attribute1 + '_anova_Coefficient.csv')

    # get the standardized deviation of the gdp and deficit
    gdp_std = df['GDP'].std()
    deficit_std = df[attribute1].std()
    # print("gdp_std", gdp_std)
    # print("deficit_std", deficit_std)

    # # get the standardized_coef_beta for gdp and deficit
    # standardized_coef_beta_for_gdp = 0.145 * (gdp_std / deficit_std)

    # print("standardized_coef_beta_for_gdp", standardized_coef_beta_for_gdp)

    # # get the standardized_coef_beta for government expenditure and deficit
    # print("standardized_coef_beta",
    #       standardized_coef_beta)


def linear_regression_c(df, attribute1, list_independent):
    # cite1 from https://online.stat.psu.edu/stat415/book/export/html/822

    # do n-way anova test, dependent_variable is the dependent variable and independent_variables are GDP, GovernmentExpenditure, PersonalDisposableIncome, CPI, UnemploymentRate
    anova_results = ols(attribute1 + ' ~ ' +
                        ' + '.join(list_independent), data=df).fit()

    # get the anova table
    anova_summary = anova_lm(anova_results)

    # cite2 from https://www.statsmodels.org/dev/anova.html
    # the total sum of squares is the sum of sum_sq in the anova table

    total_sum_of_squares = anova_summary['sum_sq'].sum()

    # the total degrees of freedom is the sum of df in the anova table
    total_degrees_of_freedom = anova_summary['df'].sum()

    table = sm.stats.anova_lm(anova_results, typ=2)

    # only get the residual sum of squares result
    residual_sum_of_squares = table['sum_sq'][len(list_independent)]
    residual_df = table['df'][len(list_independent)]
    residual_mean_square = (residual_sum_of_squares / residual_df)

    # calculate sum of squares regression
    sum_of_squares_regression = total_sum_of_squares - residual_sum_of_squares

    # calculate degrees of freedom regression
    degrees_of_freedom_regression = total_degrees_of_freedom - residual_df

    # calculate mean square regression
    mean_square_regression = sum_of_squares_regression / degrees_of_freedom_regression

    # cite from : https://online.stat.psu.edu/stat415/book/export/html/822
    # calculate f-statistic
    f_regression = mean_square_regression / residual_mean_square

    # calculate p-value
    p_value = stats.f.sf(
        f_regression, degrees_of_freedom_regression, residual_df)

    # make anova table, the attributes are sum_sq, df, Mean Square, F, the rows are the Regression, Residual, Total
    anova_table = pd.DataFrame(index=['Regression', 'Residual', 'Total'], columns=[
        'sum_sq', 'df', 'Mean Square', 'F', 'P'])

    # fill the table
    anova_table['sum_sq']['Regression'] = sum_of_squares_regression
    anova_table['df']['Regression'] = degrees_of_freedom_regression
    anova_table['Mean Square']['Regression'] = mean_square_regression
    anova_table['F']['Regression'] = f_regression

    anova_table['sum_sq']['Residual'] = residual_sum_of_squares
    anova_table['df']['Residual'] = residual_df
    anova_table['Mean Square']['Residual'] = residual_mean_square

    anova_table['sum_sq']['Total'] = total_sum_of_squares
    anova_table['df']['Total'] = total_degrees_of_freedom

    anova_table['P']['Regression'] = p_value

    # save the anova table as png in 4 folder
    if not os.path.exists('4/c'):
        os.makedirs('4/c')

    anova_table.to_csv('4/c/' + attribute1 + '_anova_table.csv')

    anova_summary = anova_results.summary()

    # only take the coef, std err, t, and sig. columns from the summary
    anova_summary = anova_summary.tables[1]

    coef = []
    std_err = []
    t = []
    sig = []

    # calculate standardized_coef_beta
    standardized_coef_beta = []

    # get the standardized_coef_beta

    for i in range(1, len(list_independent) + 2):
        coef.append(anova_summary.data[i][1])
        std_err.append(anova_summary.data[i][2])
        t.append(anova_summary.data[i][3])
        sig.append(anova_summary.data[i][4])

    # make a new table
    anova_table = pd.DataFrame(
        index=["Constant"] + list_independent, columns=['coef', 'std err', 't', 'sig.'])

    # cite from : https://www.analyticsvidhya.com/blog/2021/03/standardized-vs-unstandardized-regression-coefficient/

    # skip the 0th element, which is the constant
    for i in range(1, len(list_independent) + 1):
        standardized_coef_beta.append(
            float(coef[i]) * (df[list_independent[i - 1]].std() / df[attribute1].std()))

    # fill the table
    anova_table['coef'] = coef
    anova_table['std err'] = std_err
    anova_table['t'] = t
    anova_table['sig.'] = sig

    # the standardized_coef_beta is the last column, the first element is blank
    anova_table['standardized_coef_beta'] = [' '] + standardized_coef_beta

    # save the table as csv in 4 folder
    if not os.path.exists('4/c'):
        os.makedirs('4/c')

    anova_table.to_csv('4/c/' + attribute1 + '_initial_anova_Coefficient.csv')

    # return dictionary with each attribute and corresponding sigma
    # initialize the dictionary
    dict_sigma = {}

    # get the sigma from the anova table

    # fill the dictionary
    for i in range(1, len(list_independent) + 1):
        dict_sigma[list_independent[i - 1]] = float(anova_table['sig.'][i])

    return dict_sigma


def anova_model(df, attribute1, list_independent):
    # cite1 from https://online.stat.psu.edu/stat415/book/export/html/822

    # do n-way anova test, dependent_variable is the dependent variable and independent_variables are GDP, GovernmentExpenditure, PersonalDisposableIncome, CPI, UnemploymentRate
    anova_results = ols(attribute1 + ' ~ ' +
                        ' + '.join(list_independent), data=df).fit()

    # get the anova table
    anova_summary = anova_lm(anova_results)

    # cite2 from https://www.statsmodels.org/dev/anova.html
    # the total sum of squares is the sum of sum_sq in the anova table

    total_sum_of_squares = anova_summary['sum_sq'].sum()

    # the total degrees of freedom is the sum of df in the anova table
    total_degrees_of_freedom = anova_summary['df'].sum()

    table = sm.stats.anova_lm(anova_results, typ=2)

    # only get the residual sum of squares result
    residual_sum_of_squares = table['sum_sq'][len(list_independent)]
    residual_df = table['df'][len(list_independent)]
    residual_mean_square = (residual_sum_of_squares / residual_df)

    # calculate sum of squares regression
    sum_of_squares_regression = total_sum_of_squares - residual_sum_of_squares

    # calculate degrees of freedom regression
    degrees_of_freedom_regression = total_degrees_of_freedom - residual_df

    # calculate mean square regression
    mean_square_regression = sum_of_squares_regression / degrees_of_freedom_regression

    # cite from : https://online.stat.psu.edu/stat415/book/export/html/822
    # calculate f-statistic
    f_regression = mean_square_regression / residual_mean_square

    # calculate p-value
    p_value = stats.f.sf(
        f_regression, degrees_of_freedom_regression, residual_df)

    # make anova table, the attributes are sum_sq, df, Mean Square, F, the rows are the Regression, Residual, Total
    anova_table = pd.DataFrame(index=['Regression', 'Residual', 'Total'], columns=[
        'sum_sq', 'df', 'Mean Square', 'F', 'P'])

    # fill the table
    anova_table['sum_sq']['Regression'] = sum_of_squares_regression
    anova_table['df']['Regression'] = degrees_of_freedom_regression
    anova_table['Mean Square']['Regression'] = mean_square_regression
    anova_table['F']['Regression'] = f_regression

    anova_table['sum_sq']['Residual'] = residual_sum_of_squares
    anova_table['df']['Residual'] = residual_df
    anova_table['Mean Square']['Residual'] = residual_mean_square

    anova_table['sum_sq']['Total'] = total_sum_of_squares
    anova_table['df']['Total'] = total_degrees_of_freedom

    anova_table['P']['Regression'] = p_value

    # save the anova table as png in 4 folder
    if not os.path.exists('4/c'):
        os.makedirs('4/c')

    anova_table.to_csv('4/c/' + attribute1 + '_final_anova_table.csv')

    anova_summary = anova_results.summary()

    # only take the coef, std err, t, and sig. columns from the summary
    anova_summary = anova_summary.tables[1]

    coef = []
    std_err = []
    t = []
    sig = []

    # calculate standardized_coef_beta
    standardized_coef_beta = []

    # get the standardized_coef_beta

    for i in range(1, len(list_independent) + 2):
        coef.append(anova_summary.data[i][1])
        std_err.append(anova_summary.data[i][2])
        t.append(anova_summary.data[i][3])
        sig.append(anova_summary.data[i][4])

    # make a new table
    anova_table = pd.DataFrame(
        index=["Constant"] + list_independent, columns=['coef', 'std err', 't', 'sig.'])

    # cite from : https://www.analyticsvidhya.com/blog/2021/03/standardized-vs-unstandardized-regression-coefficient/

    # skip the 0th element, which is the constant
    for i in range(1, len(list_independent) + 1):
        standardized_coef_beta.append(
            float(coef[i]) * (df[list_independent[i - 1]].std() / df[attribute1].std()))

    # fill the table
    anova_table['coef'] = coef
    anova_table['std err'] = std_err
    anova_table['t'] = t
    anova_table['sig.'] = sig

    # the standardized_coef_beta is the last column, the first element is blank
    anova_table['standardized_coef_beta'] = [' '] + standardized_coef_beta

    # save the table as csv in 4 folder
    if not os.path.exists('4/c'):
        os.makedirs('4/c')

    anova_table.to_csv('4/c/' + attribute1 + '_final_anova_Coefficient.csv')

    # return dictionary with each attribute and corresponding sigma
    # initialize the dictionary
    dict_sigma = {}

    # get the sigma from the anova table

    # fill the dictionary
    for i in range(1, len(list_independent) + 1):
        dict_sigma[list_independent[i - 1]] = float(anova_table['sig.'][i])

    return dict_sigma


def continue_drop_until_all_p_value_less_than_0_05(dict_sigma, df, attribute1, list_independent):
    # get the max p-value
    max_p_value = max(dict_sigma.values())

    while max_p_value > 0.05:
        # drop the attribute with the max p-value
        for key, value in dict_sigma.items():
            if value == max_p_value:
                df = df.drop(columns=[key])
                list_independent.remove(key)
                dict_sigma = anova_model(df, attribute1, list_independent)

        # get the max p-value
        max_p_value = max(dict_sigma.values())

    return dict_sigma, df, attribute1, list_independent


def univraite_regression(df, dependent, fixed_factor):
    # find dependent and independent variables's location
    dependent_location = df.columns.get_loc(dependent)
    fixed_factor_location_0 = df.columns.get_loc(fixed_factor[0])
    fixed_factor_location_1 = df.columns.get_loc(fixed_factor[1])

    print("dependent_location", dependent_location)
    print("fixed_factor_location_0", fixed_factor_location_0)
    print("fixed_factor_location_1", fixed_factor_location_1)

    # drop na
    dependent = df.iloc[:, dependent_location].dropna()

    fixed_factor_0 = df.iloc[:, fixed_factor_location_0]
    fixed_factor_1 = df.iloc[:, fixed_factor_location_1]

    # quantify fixed_factor_0, go through each element in fixed_factor_0, put corresponding number in the list
    # if it's Male, then change it to 1, if it is Female, then change it to 2
    sex_list = []
    for i in range(len(fixed_factor_0)):
        if fixed_factor_0.iloc[i] == "Male":
            sex_list.append(1)
        elif fixed_factor_0.iloc[i] == "Female":
            sex_list.append(2)
        else:
            sex_list.append(-1)

    time_chd = []
    for i in range(len(dependent)):
        time_chd.append(df.iloc[i, dependent_location])

    hyperten = []
    for i in range(len(fixed_factor_1)):
        if fixed_factor_1.iloc[i] == 'No':
            hyperten.append(0)
        elif fixed_factor_1.iloc[i] == 'Yes':
            hyperten.append(1)
        else:
            hyperten.append(-1)

    # combine all the list (time_chd, hyperten, sex_list) into a list of list
    list_of_list = [time_chd, hyperten, sex_list]

    # transpose the list of list
    list_of_list = list(map(list, zip(*list_of_list)))

    # change it as array
    test_array = np.array(list_of_list)

    # https://stackoverflow.com/questions/53460099/how-to-do-a-regression-starting-from-a-list-of-list-of-elements

    dataframe = pd.DataFrame(test_array, columns=[
        'time_chd', 'hyperten', 'sex'])

    # perform two-way anova
    # cite : https://www.statology.org/two-way-anova-python/
    model = ols(
        'time_chd ~ C(hyperten) + C(sex) + C(hyperten):C(sex)', data=dataframe).fit()
    result = sm.stats.anova_lm(model, type=2)

    # save the result as csv in 5 folder
    if not os.path.exists('5'):
        os.makedirs('5')

    # only get 4th C(hyperten):C(sex) row
    sex1_and_hyperten = result.iloc[2]

    g = sns.lmplot(x="sex", y="time_chd", data=dataframe, hue="hyperten")

    # y-limit is 15 - 21
    g.set(ylim=(15, 21))

    # x-tick is 1 and 2
    g.set(xticks=[1, 2])

    # my x-tick label
    my_xticklabels = ['Male', 'Female']

    # set the x-tick label
    g.set_xticklabels(my_xticklabels)

    # my legend label
    my_legend_labels = ['No', 'Yes']

    # set the legend label to my legend label
    g._legend.set_title('Incident Hypertension')

    # set the legend label
    for t, l in zip(g._legend.texts, my_legend_labels):
        t.set_text(l)

    # move the legend to the right
    g._legend.set_bbox_to_anchor([1.15, 0.5])

    # set the title of the graph
    # cife from : https://stackoverflow.com/questions/46307941/how-can-i-add-title-on-seaborn-lmplot
    ax = plt.gca()
    ax.set_title("Estimated Marginal Means of Time (years) to CHD")

    # save the graph as png
    plt.savefig('5/Estimated Marginal Means of Time (years) to CHD.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    # make a new table
    anova_table = pd.DataFrame(index=['Hyperten', 'Sex', 'Hyperten * Sex', 'Error'], columns=[
        'df', 'sum_sq', 'mean_sq', 'F', 'sig.'])

    # get the sum_sq, df, F, PR(>F) from the result
    for i in range(4):
        anova_table.iloc[i, 0] = result.iloc[i, 0]
        anova_table.iloc[i, 1] = result.iloc[i, 1]
        anova_table.iloc[i, 2] = result.iloc[i, 2]
        anova_table.iloc[i, 3] = result.iloc[i, 3]
        anova_table.iloc[i, 4] = result.iloc[i, 4]

    # save the table as csv
    anova_table.to_csv('5/ANOVA_table.csv')


def build_Boxplot(lab_data, y_axis, x_axis, cluster):
    # get all the data
    data = lab_data

    # get the data of y_axis
    data_y = data[y_axis]

    # get the data of x_axis
    data_x = data[x_axis]

    # get the data of cluster
    data_cluster = data[cluster]

    # draw the boxplot
    sns.boxplot(x=data_x, y=data_y, hue=data_cluster)

    # save the graph as png
    plt.savefig('5/Boxplot.png', dpi=300, bbox_inches='tight')

    # set the title of the graph
    plt.title(
        "Clustered Boxplot of Time (years) to CHD by sex, exam 1 by Incident Hypertension")

    plt.show()


# --- 6 ----
def binary_logistic(df, dependent, all_the_factors):
    # cite from : https://www.geeksforgeeks.org/logistic-regression-using-statsmodels/

    #
    all_thing = df[all_the_factors]
    # print all the covariates

    # if any row has null value, delete that row in dataframe
    all_thing = all_thing.dropna()

    # quantify the covariates, if the value is "No", set it as 0, if the value is "Yes", set it as 1
    # if the value is 'Male', set it as 1, if the value is 'Female', set it as 2
    # first, create a new dataframe
    new_df = pd.DataFrame()

    changed_factors = ["prevstrk3", 'sex1', 'cursmoke1', 'diabetes1', 'bpmeds1',
                       'prevchd1',
                       'prevap1', 'prevmi1', 'prevhyp1']

    # then copy all_thing to new_df, if if the value is "No", set it as 0, if the value is "Yes", set it as 1
    # if the value is 'Male', set it as 1, if the value is 'Female', set it as 2
    for i in all_thing:
        if i in changed_factors:
            new_df[i] = all_thing[i].map(
                {'No': 0, 'Yes': 1, 'Male': 1, 'Female': 2})
        else:
            new_df[i] = all_thing[i]

    # the dependent variable is the first column of new_df
    y = new_df[dependent]

    # the independent variable is the rest of the columns of new_df
    x = new_df.drop(dependent, axis=1)

    # build the logistic regression model
    model = sm.Logit(y, x).fit()

    # save the summary of the model as txt in 6 folder
    if not os.path.exists('6'):
        os.makedirs('6')

    with open('6/summary.txt', 'w') as f:
        f.write(str(model.summary()))

    # using the predict function to predict the dependent variable, and save it in a table
    # make a new table
    prediction_table = pd.DataFrame(index=['Predicted No', 'Predicted Yes', 'Overall Percentage'], columns=[
        'Observed No', 'Observed Yes', 'Percentage correct'])

    # get the predicted probability
    prediction = model.predict(x)

    # get the chi-square value
    # cite from : https://byjus.com/maths/chi-square-test/
    # using the formula : chi-square = sum((observed - expected)^2 / expected)
    # observed is the number of observed value
    # expected is the number of expected value
    # in this case, the expected value is the number of predicted value
    chi_square = 0
    for i in range(len(prediction)):
        chi_square += (y.iloc[i] - prediction.iloc[i]
                       ) ** 2 / prediction.iloc[i]

    df = len(all_the_factors) - 1

    # get the p-value
    p_value = 1 - stats.chi2.cdf(chi_square, df)

    # make a new table named Omnibus Tests of Model Coefficients
    Omnibus_table = pd.DataFrame(index=['Chi-square', 'Degree of freedom', 'P-value'], columns=[
        'Value'])

    # get the chi-square value
    Omnibus_table.iloc[0, 0] = chi_square

    # get the p-value
    Omnibus_table.iloc[2, 0] = p_value

    # get the degree of freedom
    Omnibus_table.iloc[1, 0] = df

    # save the table as csv
    if not os.path.exists('6'):
        os.makedirs('6')

    Omnibus_table.to_csv('6/Omnibus Tests of Model Coefficients.csv')

    # if the predicted probability is greater than 0.5, set it as 1, if the predicted probability is less than 0.5, set it as 0
    prediction = [1 if i > 0.5 else 0 for i in prediction]

    # get the observed value
    observed_value = y

    # save the result in the table
    prediction_table.iloc[0, 0] = sum(
        [1 if i == 0 and j == 0 else 0 for i, j in zip(prediction, observed_value)])

    prediction_table.iloc[0, 1] = sum(
        [1 if i == 0 and j == 1 else 0 for i, j in zip(prediction, observed_value)])

    prediction_table.iloc[1, 0] = sum(
        [1 if i == 1 and j == 0 else 0 for i, j in zip(prediction, observed_value)])

    prediction_table.iloc[1, 1] = sum(
        [1 if i == 1 and j == 1 else 0 for i, j in zip(prediction, observed_value)])

    # save [0, 2] as the [0,0] divided by the sum of [0,0] and [1,0]
    prediction_table.iloc[0, 2] = prediction_table.iloc[0, 0] / \
        (prediction_table.iloc[0, 0] + prediction_table.iloc[0, 1])

    # save [1, 2] as the [1,1] divided by the sum of [1,1] and [1,0]. if the sum of [1,1] and [1,0] is 0, set it as 0
    if prediction_table.iloc[1, 0] + prediction_table.iloc[1, 1] == 0:
        prediction_table.iloc[1, 2] = 0
    else:
        prediction_table.iloc[1, 2] = prediction_table.iloc[1, 1] / \
            (prediction_table.iloc[1, 0] + prediction_table.iloc[1, 1])

    # save [2, 3] as the sum of [0,0] and [1,1] divided by the sum of [0,0], [0,1], [1,0], [1,1]
    prediction_table.iloc[2, 2] = (prediction_table.iloc[0, 0] + prediction_table.iloc[1, 1]) / \
                                  (prediction_table.iloc[0, 0] + prediction_table.iloc[0, 1] +
                                   prediction_table.iloc[1, 0] + prediction_table.iloc[1, 1])

    # save the table as csv
    if not os.path.exists('6'):
        os.makedirs('6')

    prediction_table.to_csv('6/prediction_table.csv')


# ----- main ---------


def main():
    fred_data = read_file(FRED)
    lab_data = read_file(LAB)

    # ---- 1 -----

    # make summary statistics
    summary = make_summary(fred_data)

    decide_normality(summary, fred_data)

    # ---- 2 ----
    further_summary = further_analysis(fred_data)

    # #  ------- 3 ----------
    # gdp_histogram(fred_data)
    # commercial_and_industrial_loans_histogram(fred_data)
    # gdp_gnp_boxplot(fred_data)
    # scatter_plot(fred_data, "GDP", "GNP")
    # scatter_plot(fred_data, "EmploymentRate", "UnemploymentRate")

    # # ------ 4 ------------
    # # -- a --
    # pairwise_correlation(fred_data)
    # pearsonr_p_value_table(fred_data)

    # -- b --
    # linear_regression_6(fred_data, "Deficit", [
    #     "GDP", "GovernmentExpenditure", "PersonalDisposableIncome", "CPI", "UnemploymentRate"])

    # -- c --
    dict_sigma = linear_regression_c(fred_data, "Deficit", [
        "GDP", "Loans", 'Imports', 'GNP', 'Workeroutput', "GovernmentExpenditure", "PersonalDisposableIncome",
        'RentalVacancyRate', "CPI", 'BondYield', 'EmploymentRate', "UnemploymentRate", 'MedianEarnings', 'TEDSpread',
        'ManufacturingOutput'])

    # drop_biggest_p_value
    continue_drop_until_all_p_value_less_than_0_05(dict_sigma, fred_data, "Deficit", [
        "GDP", "Loans", 'Imports', 'GNP', 'Workeroutput', "GovernmentExpenditure", "PersonalDisposableIncome",
        'RentalVacancyRate', "CPI", 'BondYield', 'EmploymentRate', "UnemploymentRate", 'MedianEarnings', 'TEDSpread',
        'ManufacturingOutput'])

    # dict_sigma = linear_regression_c(fred_data, "Deficit", [
    #     "GDP", "Loans", 'RentalVacancyRate', 'BondYield', 'EmploymentRate',  'MedianEarnings',
    #     'ManufacturingOutput'])

    # ---- 5 ------
    univraite_regression(
        lab_data, "timechd", ['sex1', 'hyperten'])

    build_Boxplot(lab_data, "timechd", 'sex1', 'hyperten')

    # ----- 6 -------
    # binary_logistic(lab_data, "prevstrk3", ["prevstrk3", 'sex1', 'totchol1', 'age1', 'sysbp1', 'diabp1', 'cursmoke1',
    #                                         'cigpday1', 'bmi1', 'diabetes1', 'bpmeds1', 'heartrte1', 'glucose1',
    #                                         'prevchd1',
    #                                         'prevap1', 'prevmi1', 'prevhyp1'])

    binary_logistic(lab_data, "prevstrk3", [
        "prevstrk3", 'sex1', 'age1', 'sysbp1', 'cursmoke1', 'bmi1'])


if __name__ == '__main__':
    main()
