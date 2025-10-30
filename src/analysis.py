import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_analysis(df, save_path='static'):
    """Performs descriptive analysis of the dataset."""
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Descriptive statistics
    stats = df.describe()

    # Visualizations
    # Histograms for numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_features:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.savefig(os.path.join(save_path, f'{col}_histogram.png'))
        plt.close()

    # Bar charts for categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        plt.figure()
        sns.countplot(y=col, data=df)
        plt.title(f'Bar Chart of {col}')
        plt.savefig(os.path.join(save_path, f'{col}_barchart.png'))
        plt.close()

    return stats
