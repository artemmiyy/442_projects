from sklearn.model_selection import train_test_split
import pandas as pd

# read the dataframe
diabetes_df = pd.read_csv("/Users/artemmiyy/Desktop/442_projects/bayesian_networks/Naive-Bayes-Classification-Data.csv")

# stratified split (70% train, 30% test)
diabetes_train, diabetes_test = train_test_split(diabetes_df, test_size = 0.3,
                                                 stratify = diabetes_df['diabetes']
                                                )


if __name__ == "__main__":
  # Question 2.1
  # 2.1.1
  p_y = diabetes_train['diabetes'].value_counts(normalize = True).to_dict()
  y_grouped = diabetes_train.groupby('diabetes')

  # 2.1.2
  p_x1_given_y = dict()
  for y_val, group in y_grouped:
      total_y = len(group)
      counts_x1 = group['glucose'].value_counts()
      p_x1_given_y[y_val] = (counts_x1 / total_y).to_dict()

  # 2.1.3
  p_x2_given_y = dict()
  for y_val, group in y_grouped:
      total_y = len(group)
      counts_x2 = group['bloodpressure'].value_counts()
      p_x2_given_y[y_val] = (counts_x2 / total_y).to_dict()

  print("Question 2.1")
  print("The values are represented as a dictionary with 0 (no diabetes) and 1 (diabetes) act as keys")
  print("P(Y) =", p_y)
  print("P(X1 | Y) =", p_x1_given_y)
  print("P(X2 | Y) =", p_x2_given_y)
  print()

  # Question 2.2
  # P(Y | X1, X2)
  def compute_p_y_given_x1_x2(x1, x2, p_y, p_x1_given_y, p_x2_given_y):
    y_values = p_y.keys()
    
    # numerator and denominator
    numerators = dict()
    for y_val in y_values:
        # P(X1|Y=y_val)
        p_x1 = p_x1_given_y[y_val].get(x1, 0.0)
        # P(X2|Y=y_val)
        p_x2 = p_x2_given_y[y_val].get(x2, 0.0)
        
        numerators[y_val] = p_y[y_val] * p_x1 * p_x2

    denominator = sum(numerators.values())

    if denominator == 0: return {y_val: 1 / len(y_values) for y_val in y_values}

    # Compute conditional probability for each y
    p_y_given_x1_x2 = {y_val: num / denominator for y_val, num in numerators.items()}
    return p_y_given_x1_x2
  
  results = list()
  for i, row in diabetes_test.iterrows():
      x1 = row['glucose']
      x2 = row['bloodpressure']
      p_cond = compute_p_y_given_x1_x2(x1, x2, p_y, p_x1_given_y, p_x2_given_y)
      predicted_y = max(p_cond, key = p_cond.get)
      
      results.append({
          'glucose': x1,
          'bloodpressure': x2,
          'P(Y = 0 | X1, X2)': p_cond.get(0, 0.0),
          'P(Y = 1| X1, X2)': p_cond.get(1, 0.0),
          'Predicted Y': predicted_y,
          'Actual Y': row['diabetes']
      })

  predictions = pd.DataFrame(results)
  print("Question 2.2")
  print(predictions)