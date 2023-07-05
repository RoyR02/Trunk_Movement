import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Load ADNI dataset as a Pandas dataframe
adni_df = pd.read_csv('AD_train.csv')

# Encode the Status variable
le = LabelEncoder()
adni_df['Status'] = le.fit_transform(adni_df['Status'])

# Print the number of samples and features in the dataset
print("Number of samples: ", adni_df.shape[0])
print("Number of features: ", adni_df.shape[1] - 1) # Exclude the Status variable

# Compute descriptive statistics for each feature
for col in adni_df.columns[:-1]: # Exclude the Status variable
    stat_dict = {
        'mean': adni_df[col].mean(),
        'median': adni_df[col].median(),
        'std_dev': adni_df[col].std(),
        'min': adni_df[col].min(),
        '25%': adni_df[col].quantile(0.25),
        '50%': adni_df[col].quantile(0.50),
        '75%': adni_df[col].quantile(0.75),
        'max': adni_df[col].max()
    }
    print(f"\n{col} statistics:")
    for key, value in stat_dict.items():
        print(f"{key}: {value:.2f}")

# Perform t-test between Normal and AD groups for Status feature
group1 = adni_df[adni_df['Status'] == 0]['Status']
group2 = adni_df[adni_df['Status'] == 1]['Status']
t_stat, p_val = stats.ttest_ind(group1, group2)
print(f"\nT-test results for Status feature: t-statistic = {t_stat:.2f}, p-value = {p_val:.2f}")