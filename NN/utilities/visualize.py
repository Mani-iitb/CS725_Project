import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 


def countplot(df, column):
    plt.figure(figsize=(10, 8))
    sns.countplot(data=df, x=column)
    plt.xticks(rotation=90)
    plt.title(f'A histogram of column')
    plt.show()


def correlation(df):
    # general correlation matrix 
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize= (10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature correlation matrix')
    plt.show()