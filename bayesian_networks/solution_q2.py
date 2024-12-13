from sklearn.model_selection import train_test_split
import pandas as pd

# read the dataframe
diabetes_df = pd.read_csv("/Users/artemmiyy/Desktop/442_projects/bayesian_networks/Naive-Bayes-Classification-Data.csv")

# stratified split (70% train, 30% test)
diabetes_train, diabetes_test = train_test_split(diabetes_df, test_size=0.3,
                                                 stratify=diabetes_df['diabetes']
                                                )

