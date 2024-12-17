 import pandas as pd
 # Load the dataset
 file_path = '/content/mobile_year_2022_quarter_03.csv'
 data = pd.read_csv(file_path)
 # Display the first few rows to inspect the structure
 print("Preview of the Dataset:")
 print(data.head())
 # Check basic information about the dataset
 print("\nDataset Info:")
 print(data.info())

 # Standardize column names for ease of use (lowercase and replace spaces with underscores)
 data.columns = data.columns.str.strip().str.replace(' ', '_').str.lower()
 print("\nStandardized Column Names:")
 print(data.columns)


 # Check for missing values
 print("\nMissing Values Before Cleaning:")
 print(data.isnull().sum())
 # Fill numeric columns with their mean value if missing
 numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
 data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
 # Recheck missing values
 print("\nMissing Values After Cleaning:")
 print(data.isnull().sum())


 # Convert rank columns to integers (if applicable)
 rank_cols = ['rank_upload', 'rank_download', 'rank_latency']
 for col in rank_cols:
 if col in data.columns:
 data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
 print("\nData Types After Adjustment:")
 print(data.dtypes)

    # Remove duplicates from the dataset
 data = data.drop_duplicates()
 print("\nNumber of Records After Removing Duplicates:", len(data))

    # Clip numeric columns at the 99th percentile to handle outliers
 outlier_cols = ['avg_u_kbps', 'avg_d_kbps', 'avg_lat_msavg']
 for col in outlier_cols:
 if col in data.columns:
 upper_limit = data[col].quantile(0.99)
 data[col] = data[col].clip(upper=upper_limit)
 print("\nOutliers Handled for Columns:", outlier_cols)

    # Filter data for India
 india_data = data[data['name'].str.lower() == 'india']
 print("\nIndia's Data:")
print(india_data)

# Calculate global averages for numeric columns
 global_averages = data.mean(numeric_only=True)
 print("\nGlobal Averages:")
 print(global_averages)

 # Create a comparison DataFrame
 comparison = pd.DataFrame({
 'India': india_data.mean(numeric_only=True),
 'Global Average': global_averages
 }).dropna()
 print("\nComparison of India vs Global Averages:")
 print(comparison)

 # Display all column names in the dataset to verify the correct names
 print("\nColumns in the dataset:")
 print(data.columns)

 # Clean and convert columns to numeric values
 data['avg._avg_u_kbps'] = data['avg._avg_u_kbps'].str.replace(',', '').astype(float)
 data['avg._avg_d_kbps'] = data['avg._avg_d_kbps'].str.replace(',', '').astype(float)
 data['avg_lat_ms'] = data['avg_lat_ms'].str.replace(',', '').astype(float)
 # Verify the conversion
 print(data[['avg._avg_u_kbps', 'avg._avg_d_kbps', 'avg_lat_ms']].head())

 # Group data by country and calculate mean for relevant metrics
 country_metrics = data.groupby('name')[['avg._avg_u_kbps', 'avg._avg_d_kbps', 'avg_lat_ms']].mean()
 # Display the aggregated metrics
 print("\nCountry Metrics (Mean Values):")
 print(country_metrics.head())

 # Extract India's metrics for comparison
 india_metrics = country_metrics.loc['India']
 print("\nIndia's Metrics:")
 print(india_metrics)

 # Add a column for the difference compared to India
 comparison_df = country_metrics.copy()
 comparison_df['Difference_Upload'] = comparison_df['avg._avg_u_kbps'] - india_metrics['avg._avg_u_kbps']
 comparison_df['Difference_Download'] = comparison_df['avg._avg_d_kbps'] - india_metrics['avg._avg_d_kbps']
 comparison_df['Difference_Latency'] = comparison_df['avg_lat_ms'] - india_metrics['avg_lat_ms']
 # Sort data for visualization (optional)
 comparison_df = comparison_df.sort_values(by='Difference_Download', ascending=False)
 print("\nComparison DataFrame:")
 print(comparison_df.head())

# Bar plot of upload speed comparison
 import matplotlib.pyplot as plt
 plt.figure(figsize=(10, 6))
 comparison_df['Difference_Upload'].sort_values().plot(kind='bar', color='blue', alpha=0.7)
 plt.title('Difference in Upload Speeds (Compared to India)')
 plt.ylabel('Difference in Upload Speed (Kbps)')
 plt.xlabel('Country')
 plt.tight_layout()
 plt.show()


 import matplotlib.pyplot as plt
 # Bar plot of upload speed comparison
 plt.figure(figsize=(10, 6))
 comparison_df['Difference_Upload'].sort_values().plot(kind='bar', color='blue', alpha=0.7)
 plt.title('Difference in Upload Speeds (Compared to India)')
 plt.ylabel('Difference in Upload Speed (Kbps)')
 plt.xlabel('Country')
 plt.tight_layout()
 plt.show()

 # Line plot for download speeds
 plt.figure(figsize=(12, 6))
 country_metrics['avg._avg_d_kbps'].sort_values().plot(kind='line', marker='o', color='green')
 plt.axhline(y=india_metrics['avg._avg_d_kbps'], color='red', linestyle='--', label="India")
 plt.title('Average Download Speeds Across Countries')
 
plt.ylabel('Download Speed (Kbps)')
 plt.xlabel('Country')
 plt.legend()
 plt.tight_layout()
 plt.show()

 # Scatter plot: Download speed vs Latency
 plt.figure(figsize=(8, 6))
 plt.scatter(country_metrics['avg._avg_d_kbps'], country_metrics['avg_lat_ms'], alpha=0.6, label='Other Countries')
 plt.scatter(india_metrics['avg._avg_d_kbps'], india_metrics['avg_lat_ms'], color='red', label='India', s=100)
 plt.title('Download Speed vs Latency')
 plt.xlabel('Download Speed (Kbps)')
 plt.ylabel('Latency (ms)')
 plt.legend()
 plt.tight_layout()
 plt.show()

# Bar chart to compare upload, download, and latency differences
 comparison_df[['Difference_Upload', 'Difference_Download', 'Difference_Latency']].head(10).plot(
 kind='bar', figsize=(12, 8), alpha=0.75
 )
 plt.title('Comparison of Metrics Against India')
 plt.ylabel('Difference (Compared to India)')
 plt.xlabel('Country')
 plt.legend(loc='upper right')
 plt.tight_layout()
 plt.show()

 import seaborn as sns
 # Create a heatmap for the correlations between metrics
 plt.figure(figsize=(12, 8))
 sns.heatmap(
 country_metrics.corr(),
 annot=True,
 cmap='coolwarm',
 fmt='.2f'
 )
 plt.title('Correlation Between Metrics Across Countries')
 plt.tight_layout()
 plt.show()


from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
 # Select features and target variable
 X = country_metrics[['avg._avg_u_kbps', 'avg_lat_ms']]  # Upload speed and latency as features
 y = country_metrics['avg._avg_d_kbps']  # Download speed as target variable
 # Split data into training and testing sets (80% train, 20% test)
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 # Standardize the features (important for most models)
 scaler = StandardScaler()
 X_train_scaled = scaler.fit_transform(X_train)
 X_test_scaled = scaler.transform(X_test)
 from sklearn.ensemble import RandomForestRegressor
 # Initialize and train the Random Forest model
 model = RandomForestRegressor(n_estimators=100, random_state=42)

 plt.figure(figsize=(10, 6))
 plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
 plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
 plt.title('Predicted vs Actual Download Speed')
 plt.xlabel('Actual Download Speed (Kbps)')
 plt.ylabel('Predicted Download Speed (Kbps)')
 plt.legend()
 plt.tight layout() 
 plt.tight_layout()
 plt.show()

 # Calculate residuals
 residuals = y_test - y_pred
 # Plot residuals
 plt.figure(figsize=(10, 6))
 plt.scatter(y_pred, residuals, alpha=0.7)
 plt.axhline(y=0, color='red', linestyle='--', label="Zero Residual Line")
 plt.title('Residuals vs Predicted Download Speed')
 plt.xlabel('Predicted Download Speed (Kbps)')
 plt.ylabel('Residuals (Actual - Predicted)')
 plt.legend()
 plt.tight_layout()
 plt.show()
# Get feature importances from the trained model
 importances = model.feature_importances_
 # Plot the feature importances
 plt.figure(figsize=(8, 6))
 plt.barh(X.columns, importances, color='green')
 plt.title('Feature Importance (Upload Speed and Latency)')
 plt.xlabel('Importance')
 plt.ylabel('Feature')
 plt.tight_layout()
 plt.show()
