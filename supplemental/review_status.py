import pandas as pd

### Load the CSV file into a DataFrame
csv_path = "c:\\Users\\Paige\\Projects\\data\\alderaan_results\\ecc-all-LC\\ecc-all-LC.csv"
df = pd.read_csv(csv_path)

### Specify the review values
review_values = ['Minor Issues', 'Major Issues', 'Critical Issues']

### Filter rows where review is in the specified values
filtered_df = df[df['review'].isin(review_values)]

### Extract unique koi_ids from the filtered DataFrame
unique_koi_ids = filtered_df['koi_id'].unique()

### Print or use the koi_ids list as needed
print("KOI IDs corresponding to review values:")
print(unique_koi_ids)