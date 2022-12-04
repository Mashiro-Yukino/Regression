import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import math

from scipy.stats.mstats import zscore

from statsmodels.stats.anova import anova_lm

FRED = 'FredData.sav'


def read_file(filename):
    # pip install pyreadstat
    df = pd.read_spss('FredData.sav')

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

    unindependent_attribute = ['GDP', 'GovernmentExpenditure',
                               'PersonalDisposableIncome', 'CPI', 'UnemploymentRate']

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
        index=['Constant', 'GDP', 'GovernmentExpenditure',
               'PersonalDisposableIncome', 'CPI', 'UnemploymentRate'],
        columns=[
            'coef', 'std err', 't', 'sig.'])

    # cite from : https://www.analyticsvidhya.com/blog/2021/03/standardized-vs-unstandardized-regression-coefficient/

    # skip the 0th element, which is the constant
    for i in range(1, 6):
        standardized_coef_beta.append(
            float(coef[i]) * (df[unindependent_attribute[i - 1]].std() / df[attribute1].std()))

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


# ----- main ---------


def main():
    fred_data = read_file(FRED)

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
    linear_regression_6(fred_data, "Deficit", [
                        "GDP", "GovernmentExpenditure", "PersonalDisposableIncome", "CPI", "UnemploymentRate"])


if __name__ == '__main__':
    main()
