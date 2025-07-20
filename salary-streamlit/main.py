import streamlit as st

st.title("AI powered Salary Prediction")
st.subheader("get accurate salary predictions based on your professional profile")

st.subheader("Personal Information")
Gender=st.selectbox("Gender:",["Male","Female","other"])
st.write("Your gender is",Gender)
age=st.number_input("Age",min_value=18,max_value=65,value=25,step=1)

st.write("Your Age is:", age)

st.subheader("Professional Details")
education= st.selectbox("Education Level",["High School","Bachelor","Master","PHD"])

Job_title=st.selectbox("Job Title",["developer","Data Scientist","Manager","Designer","Engineer","Other"])

st.subheader("Experience")
experience=st.slider("years of Experience",0,50,5)

if experience<2:
    level="Entry Level"
elif experience<5:
    level="Junior Level"
elif experience<10:
    level="Mid-Junior Level"
elif experience<20:
    level="Senior Level"
else:
    level="Expert Level"

st.info(f" {level}")    
 
st.subheader("Career Insights")
if experience>=5:
    st.success("Experienced professional-Strong market position has been brewed")
else:
    st.warning("Still building profile- Plenty of room to grow")

import pandas as pd
 
st.button("Model evaluation")

file = st.file_uploader("upload your file",type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Data Preview")
    st.dataframe(df)




import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("employee_salary_dataset.csv")  # Replace with your CSV file

# Title
st.title("📊 Model Evaluation Table with Graphs")

# Prepare features and target
X = df.drop(['Salary', 'predicted salary', 'EmployeeID'], axis=1)
y = df['Salary']

# Preprocess categorical columns
categorical_cols = ['Education', 'JobTitle', 'Location']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Fit model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create evaluation DataFrame
df_eval = X_test.copy()
df_eval['Actual Salary'] = y_test.values
df_eval['Predicted Salary'] = y_pred.round(0).astype(int)

# Show table
st.subheader("📋 Actual vs Predicted Table")
st.dataframe(df_eval.reset_index(drop=True), use_container_width=True)

# Bar Chart
st.subheader("📊 Bar Chart: Actual vs Predicted")
chart_data = df_eval[['Actual Salary', 'Predicted Salary']].reset_index(drop=True)
chart_data.index.name = 'Sample Index'

st.bar_chart(chart_data)

# Scatter Plot
st.subheader("📈 Scatter Plot: Predicted vs Actual")
fig, ax = plt.subplots()
sns.scatterplot(x='Actual Salary', y='Predicted Salary', data=df_eval, ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')  # Diagonal line
ax.set_title("Actual vs Predicted Salary")
st.pyplot(fig)

    


