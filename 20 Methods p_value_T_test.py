import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Define the features and target
X = df.drop(['Status'], axis=1)
y = df['Status']

# Label encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)
# Convert X to numeric
X = X.apply(pd.to_numeric)


# 1. Compute the mean of each column
means = X.mean()

# 2. Compute the median of each column
medians = X.median()

# 3. Compute the mode of each column
modes = X.mode()

# 4. Compute the standard deviation of each column
stds = X.std()

# 5. Compute the variance of each column
variances = X.var()

# 6. Compute the minimum value of each column
mins = X.min()

# 7. Compute the maximum value of each column
maxs = X.max()

# 8. Compute the range of each column
ranges = maxs - mins

# 9. Compute the skewness of each column
skews = X.skew()

# 10. Compute the kurtosis of each column
kurtoses = X.kurtosis()

# 11. Compute the correlation between each pair of columns
correlations = X.corr()

# 12. Compute the covariance between each pair of columns
covariances = X.cov()


# 13. Compute the p-value of each column using t-test against the target variable
p_values = [stats.ttest_ind(X[col], y).pvalue for col in X.columns]

# 14. Compute the t-statistic of each column using t-test against the target variable
t_stats = [stats.ttest_ind(X[col], y).statistic for col in X.columns]

# 15. Compute the effect size of each column using Cohen's d with t-test against the target variable
effect_sizes = [abs((X[col].mean() - y.mean()) / np.sqrt((X[col].std() ** 2 + y.std() ** 2) / 2)) for col in X.columns]

# 16. Compute the p-value of each column using ANOVA against the target variable
anova_p_values = [stats.f_oneway(X[col], y).pvalue for col in X.columns]

# 17. Compute the F-statistic of each column using ANOVA against the target variable
f_stats = [stats.f_oneway(X[col], y).statistic for col in X.columns]

# 18. Compute the p-value of each column using Pearson's chi-squared test against the target variable
chi2_p_values = [stats.chi2_contingency(pd.crosstab(X[col], y))[1] for col in X.columns]

# 19. Compute the chi-squared statistic of each column using Pearson's chi-squared test against the target variable
chi2_stats = [stats.chi2_contingency(pd.crosstab(X[col], y))[0] for col in X.columns]

# 20. Compute the mutual information of each column with the target variable
mutual_infos = [mutual_info_regression(X[[col]], y)[0] for col in X.columns]

# Print the results for each statistical measure
# Print the results for each statistical measure
print('Means:\n', means)
print('\nMedians:\n', medians)
print('\nModes:\n', modes)
print('\nStandard Deviations:\n', stds)
print('\nVariances:\n', variances)
print('\nMinimums:\n', mins)
print('\nMaximums:\n', maxs)
print('\nRanges:\n',ranges)
print('\nSkewnesses:\n', skews)
print('\nKurtoses:\n', kurtoses)
print('\nCorrelations:\n', correlations)
print('\nCovariances:\n', covariances)
print('\nT-test p-values:\n', p_values)
print('\nT-test t-statistics:\n', t_stats)
print('\nEffect Sizes (Cohen\'s d):\n', effect_sizes)
print('\nANOVA p-values:\n', anova_p_values)
print('\nANOVA F-statistics:\n', f_stats)
print('\nChi-squared p-values:\n', chi2_p_values)
print('\nChi-squared statistics:\n', chi2_stats)
print('\nMutual Information:\n', mutual_infos)