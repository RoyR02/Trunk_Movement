import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load ADNI dataset as a Pandas dataframe
adni_df = pd.read_csv('AD_train.csv')

# Encode the Status variable
#le = LabelEncoder()
#adni_df['Status'] = le.fit_transform(adni_df['Status'])

# Set the style and colors for the plots
sns.set_style('darkgrid')
sns.set_palette('muted')

# Plot histograms of each feature for each group
for col in adni_df.columns[:-1]: # Exclude the Status variable
    fig, ax = plt.subplots()
    sns.histplot(data=adni_df, x=col, hue='Status', kde=True, ax=ax)
    ax.set_title(f"Histogram of {col}")
    plt.savefig(f"{col}_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()
    

# Set the style and colors for the plots
sns.set_style('whitegrid')
sns.set_palette('Set2')

# Plot histograms of each feature for each group
for col in adni_df.columns[:-1]: # Exclude the Status variable
    fig, ax = plt.subplots()
    sns.histplot(data=adni_df, x=col, hue='Status', kde=True, ax=ax)
    ax.set_title(f"Histogram of {col}")
    plt.savefig(f"{col}_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()

# Plot boxplots of each feature for each group
for col in adni_df.columns[:-1]: # Exclude the Status variable
    fig, ax = plt.subplots()
    sns.boxplot(data=adni_df, x='Status', y=col, ax=ax)
    ax.set_title(f"Boxplot of {col}")
    plt.savefig(f"{col}_boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()

# Compute the correlation matrix between features
corr_matrix = adni_df.corr()

# Plot the correlation matrix as a heatmap
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Correlation Matrix")
plt.savefig("correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# Compute the pairplot between features
sns.pairplot(adni_df, hue='Status')
plt.savefig("pairplot.png", dpi=300, bbox_inches='tight')
plt.show()

# Compute the violin plot of each feature for each group
for col in adni_df.columns[:-1]: # Exclude the Status variable
    fig, ax = plt.subplots()
    sns.violinplot(data=adni_df, x='Status', y=col, ax=ax)
    ax.set_title(f"Violin plot of {col}")
    plt.savefig(f"{col}_violinplot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
# Plot scatterplots of each feature against the Status variable
for col in adni_df.columns[:-1]: # Exclude the Status variable
    fig, ax = plt.subplots()
    sns.scatterplot(data=adni_df, x=col, y='Status', ax=ax)
    ax.set_title(f"Scatter plot of {col} vs. Status")
    plt.savefig(f"{col}_scatterplot.png", dpi=300, bbox_inches='tight')
    plt.show()
    