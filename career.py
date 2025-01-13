#CODE:
# Importing required modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load the dataset from the provided URL
url = 'dataset'
df = pd.read_csv(url)

# Drop columns that are completely empty (have no valid values)
df = df.dropna(axis=1, how='all')

# Handle missing data for numerical columns (using mean for imputation)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
numerical_imputer = SimpleImputer(strategy='mean')  # Can also use 'median' if preferred
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

# Handle missing data for categorical columns (using mode for imputation)
categorical_cols = df.select_dtypes(include=['object']).columns
categorical_imputer = SimpleImputer(strategy='most_frequent')  # Using 'most_frequent' to fill missing categorical values
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Convert categorical columns to numerical using Label Encoding
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Preprocess the data
X = df.drop(columns=['Career'])  # Drop the "Career" column from the features
y = df['Career']  # Target is the "Career" column

# Display the features and their count
print("Feature Names:")
print(X.columns.tolist())
print("\nNumber of features:", X.shape[1])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Function to predict career based on user input
def predict_career(user_input):
    # Convert user input to DataFrame with the same feature names
    user_input_df = pd.DataFrame([user_input], columns=X.columns)
    prediction_encoded = classifier.predict(user_input_df)[0]
    
    # Decode the prediction back to the original career label
    predicted_career = label_encoders['Career'].inverse_transform([prediction_encoded])[0]
    
    # Generate reasoning based on input features
    feature_values = user_input_df.values.flatten()
    reasons = []
    for feature, value in zip(X.columns, feature_values):
        reasons.append(f"{feature}: {value}")
    
    # Adding feature importance to reasoning
    importance = classifier.feature_importances_
    important_features = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)[:5]  # Top 5 important features
    importance_reasons = [f"Top 5 important features:\n"]
    for feature, imp in important_features:
        importance_reasons.append(f"{feature}: Importance={imp:.4f}")
    
    reasoning = "Prediction based on the following features:\n" + "\n".join(reasons) + "\n" + "\n".join(importance_reasons)
    
    return predicted_career, reasoning, important_features

# Example: Replace with appropriate values based on your dataset
example_input = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]  # Replace with actual feature values

# Ensure the input length matches the number of features 
assert len(example_input) == X.shape[1], f"Expected {X.shape[1]} features, but got {len(example_input)}"

# Predict the career and provide reasoning
predicted_career, reasoning, important_features = predict_career(example_input)

# Predict for the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

# Output the predicted career and the reasoning
print(f"Predicted Career: {predicted_career}")
print(reasoning)
print("Accuracy: ", acc*100,"%")

# Plot feature importance
features, importance = zip(*important_features)

# Create a scatter plot of the top 5 important features
plt.figure(figsize=(10, 6))
plt.scatter(features, importance, color='blue')
plt.title('Top 5 Important Features')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()
