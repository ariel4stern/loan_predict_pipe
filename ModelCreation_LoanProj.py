import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

train = pd.read_csv("train-LoanProj.csv")
#print(train.columns)
#print(test.columns)


X = train[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Married','Credit_History']]
y = train['Loan_Status'].map({'Y':1, 'N':0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#The file test-LoanProj.csv does not have the label column ,so I did this for real score

numeric_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
categorical_features = ['Married','Credit_History']

numeric_pipeline = Pipeline([           #pipe for std scale of numeric_features
    ('imputer', SimpleImputer(strategy='mean')),         #nan replacement strategy - mean
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([    #pipe for OnHotEncode scale of categorical_features
    ('imputer', SimpleImputer(strategy='most_frequent')),      #nan replacement strategy - most frequent
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([      #preprocessor for pipe model numeric_features and categorical_features each by their pipe scaler
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

pipe_loan_model = Pipeline([
    ('preprocess', preprocessor),
    ('model', SVC(kernel='rbf', probability=True))
])

pipe_loan_model.fit(X_train, y_train)          #fit the x after the preprocessor of the pipe model

y_pred = pipe_loan_model.predict(X_test)
confusion_matrix_table = confusion_matrix(y_test, y_pred)


scores = cross_val_score(pipe_loan_model,X,y,cv=5,scoring='accuracy') # cross validation scoring

model_data = {
    "model": pipe_loan_model,
    "confusion_matrix_table": confusion_matrix_table
}

print(f"Confusion Matrix Score:\n{confusion_matrix_table}")
print(f"scores:\n{scores}")
print("mean score: ",scores.mean())

model_name = str('loan_svm_model.joblib')

# Save the model to disk
joblib.dump(model_data,model_name)
print(f"Model  -- {model_name} --  saved successfully!")


