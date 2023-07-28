import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load and Preprocess the Dataset
# Load your dataset here
data = pd.read_csv('water_potability.csv')  # Replace 'water_quality_dataset.csv' with your dataset filename

# Drop rows with missing values (you can handle missing values differently if needed)
data.dropna(inplace=True)

# Split the dataset into features (X) and the target variable (y)
X = data.drop('Potability', axis=1)
y = data['Potability']

# Step 2: Data Scaling and Splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Model Training
# Use a Random Forest Classifier as an example. You can try other classifiers as well.
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 4: Save the trained model to a .pkl file
model_filename = 'model.pkl'
joblib.dump(rf_classifier, model_filename)
