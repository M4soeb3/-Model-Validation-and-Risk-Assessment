from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Split the data into features and target
X = data.drop(columns=['Risk'])
y = data['Risk']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
