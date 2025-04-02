import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

data = pd.read_csv("sold_listings.csv")

X = data[['title', 'category', 'category_details', "description", 'size']]
y = data['price']

preprocessor = ColumnTransformer(transformers=[
    ('title_tfidf', TfidfVectorizer(max_features=100), 'title'),
    ('desc_tfidf', TfidfVectorizer(max_features=300), 'description'),

    ('cat', OneHotEncoder(handle_unknown='ignore'), ['category', 'category_details', 'size'])
], remainder='drop')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

joblib.dump(pipeline, "price_recommendation_model.pkl")