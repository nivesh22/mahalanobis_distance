import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import mahalanobis

def calculate_mahalanobis_distance(df, threshold=3):
  """
  Calculates Mahalanobis distance for each data point and flags anomalies.

  Args:
      df: Pandas dataframe containing columns 'source', 'currency', 'family', 'date', and 'volume'.
      threshold: The threshold for Mahalanobis distance to consider a point anomalous (default=3).

  Returns:
      A new dataframe with additional columns 'mahalanobis_distance' and 'anomaly'.
  """
  # One-hot encode categorical columns
  categorical_cols = ['source', 'currency', 'family']
  encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
  encoded_df = pd.concat([df[['date', 'volume']], encoder.fit_transform(df[categorical_cols])], axis=1)

  # Get the mean vector and covariance matrix
  mean_vector = encoded_df.mean(axis=0)
  covariance_matrix = encoded_df.cov()

  # Invert the covariance matrix (handle singularity if needed)
  try:
    inv_cov = covariance_matrix.linalg.inv()
  except np.linalg.LinAlgError:
    # Add a small value to the diagonal to avoid singularity
    inv_cov = covariance_matrix.add(np.diag(np.repeat(1e-6, len(covariance_matrix))))

  # Calculate Mahalanobis distance and flag anomalies
  mahalanobis_distances = []
  anomalies = []
  for _, row in encoded_df.iterrows():
    data_point = row.values
    distance = mahalanobis(data_point, mean_vector, inv_cov)
    mahalanobis_distances.append(distance)
    anomaly = distance > threshold
    anomalies.append(anomaly)

  # Add Mahalanobis distance and anomaly flag as new columns
  df['mahalanobis_distance'] = mahalanobis_distances
  df['anomaly'] = anomalies

  return df

# Example usage
df = pd.DataFrame({'source': ['A', 'B', 'A', 'C', 'A'],
                   'currency': ['USD', 'EUR', 'USD', 'USD', 'EUR'],
                   'family': ['X', 'Y', 'X', 'Z', 'X'],
                   'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
                   'volume': [100, 200, 150, 250, 350]})

df = calculate_mahalanobis_distance(df.copy())
print(df)
