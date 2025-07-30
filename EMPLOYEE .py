#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv(r"C:\Users\hp\Downloads\adult 3.csv")


# In[3]:


data.head(10)


# In[4]:


data.shape # dimension of the dataset


# In[5]:


data.head()


# In[6]:


data.tail() # last seven rows in it


# In[7]:


# check is there is null values in it
data.isna()


# In[8]:


print( data.occupation.value_counts())# categories of data 


# In[9]:


print(data.gender.value_counts())


# In[10]:


print(data.gender.value_counts())


# In[11]:


print(data.education.value_counts())


# In[12]:


print(data.education.value_counts())


# In[13]:


print(data.workclass.value_counts())


# In[14]:


print(data.age.value_counts())


# In[15]:


print(data.age.value_counts())


# In[16]:


import pandas as pd 


# In[17]:


data = pd.read_csv(r"C:\Users\hp\Downloads\adult 3.csv")


# In[18]:


print(data.head())              # Check the DataFrame's structure
print(data.columns)            # Make sure 'occupation' is listed
print(data['occupation'].unique())  # See what's inside that column

data.occupation.replace({'?': 'Others'}, inplace=True)
# In[19]:


print(data.occupation.value_counts())

data.workclass.replace({'?': 'Others'}, inplace=True)

# In[20]:


print(data.workclass.value_counts())


# In[21]:


data = data[data['workclass']!= 'Without-pay']
data = data[data['workclass']!= 'Never-worked']


# In[22]:


data = data[data['workclass']!= 'Without-pay']
data = data[data['workclass']!= 'Never-worked']


# In[23]:


print(data.workclass.value_counts())


# In[ ]:





# In[24]:


print(data.education.value_counts()) # creading this table


# In[25]:


data.drop(columns=['education'],inplace =True)


# In[26]:


data


# In[27]:


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


import pandas as pd


# In[30]:


data = pd.read_csv(r"C:\Users\hp\Downloads\adult 3.csv")


# In[31]:


import matplotlib.pyplot as plt
plt.boxplot(data['age'])
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import pandas as pd


workclass_counts = data['workclass'].value_counts()

plt.bar(workclass_counts.index, workclass_counts.values)
plt.title('Distribution of Workclass')
plt.xlabel('Workclass')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[33]:


# label encoding
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital-status']=encoder.fit_transform(data['marital-status'])               #Label Encoding (Part of Data Preprocessing)
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])
data['race']=encoder.fit_transform(data['race'])
data['gender']=encoder.fit_transform(data['gender'])
data['native-country']=encoder.fit_transform(data['native-country'])


# In[34]:


# label encoding
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital-status']=encoder.fit_transform(data['marital-status'])               #Label Encoding (Part of Data Preprocessing)
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])
data['race']=encoder.fit_transform(data['race'])
data['gender']=encoder.fit_transform(data['gender'])
data['native-country']=encoder.fit_transform(data['native-country'])


# In[35]:


data.shape


# In[36]:


data=data[(data['age']<=75)&(data['age']>=17)]


# In[37]:


plt.boxplot(data['age'])
plt.show()


# In[38]:


plt.boxplot(data['capital-gain'])
plt.show()


# In[39]:


plt.boxplot(data['educational-num'])
plt.show()


# In[40]:


data=data[(data['educational-num']<=16)&(data['educational-num']>=5)]


# In[41]:


plt.boxplot(data['educational-num'])
plt.show()


# In[42]:


data


# In[ ]:


x # input data


# In[ ]:


y # output


# In[ ]:


from sklearn.preprocessing import LabelEncoder   #import libarary
encoder=LabelEncoder()                       #create object
data['workclass']=encoder.fit_transform(data['workclass']) #7 categories   0,1, 2, 3, 4, 5, 6,
data['marital-status']=encoder.fit_transform(data['marital-status'])   #3 categories 0, 1, 2
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])      #5 categories  0, 1, 2, 3, 4
data['race']=encoder.fit_transform(data['race'])  
data['gender']=encoder.fit_transform(data['gender'])    #2 catogories     0, 1
data['native-country']=encoder.fit_transform(data['native-country'])


# In[ ]:


x = data.drop(columns=['income'])
y  = data['income']                    


# In[ ]:


x = data.drop(columns=['income'])


# In[ ]:


print(x.dtypes)


# In[ ]:


x = x.drop(columns=['education'])


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=2, stratify=y)


# In[ ]:


xtrain


# In[ ]:


ytest


# In[ ]:


xtest


# In[ ]:


ytrain


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
# using KNearest Neighbour Classification
knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain)
predict = knn.predict(xtest) # predicted value
predict


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score = (ytest , predict) # checking the accuracy score of the modelt accuracy_score


# In[ ]:


ytrain = ytrain.apply(lambda x: 1 if x == '>50K' else 0)


# In[ ]:


y


# In[ ]:


ytest = ytest.apply(lambda x: 1 if x == '>50K' else 0)


# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(xtrain, ytrain)           # Train the model
y_pred = lr.predict(xtest)      # Predict on test data
print(y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score

# Convert predicted probabilities to class labels (0 or 1)
y_pred_labels = [1 if prob >= 0.5 else 0 for prob in y_pred]


accuracy_score(ytest, y_pred_labels)


# In[ ]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier ( solver = 'adam' , hidden_layer_sizes = (5,2) , random_state = 2 , max_iter =2000)
clf.fit(xtrain, ytrain)
predict2 = clf.predict(xtest)
predict2


# In[ ]:


# checking the accuracy through this model 
from sklearn.metrics import accuracy_score
accuracy_score = (ytest , predict2) # checking the accuracy score of the modelt accuracy_score


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


# In[49]:


import matplotlib.pyplot as plt
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[50]:


import matplotlib.pyplot as plt
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print(" Saved best model as best_model.pkl")


# In[47]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Assuming your data is loaded into a DataFrame called 'data'
# Step 1: Encode categorical target column 'income' to numeric
data['income'] = LabelEncoder().fit_transform(data['income'])  # '>50K' becomes 1, '<=50K' becomes 0

# Step 2: Drop non-numeric columns from features if needed (or encode them too)
x = data.drop(columns=['income'])
x = x.select_dtypes(include=['int64', 'float64'])  # Keep only numeric features

y = data['income']  # Target

# 🚀 Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 4: Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

# Step 5: Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Step 6: Pick best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Step 7: Save the best model
joblib.dump(best_model, "best_model.pkl")
print("Saved best model as best_model.pkl")


# In[52]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nimport joblib\n\n# Load the trained model\nmodel = joblib.load("best_model.pkl")\n\nst.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")\n\nst.title("💼 Employee Salary Classification App")\nst.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")\n\n# Sidebar inputs (these must match your training feature columns)\nst.sidebar.header("Input Employee Details")\n\n# ✨ Replace these fields with your dataset\'s actual input columns\nage = st.sidebar.slider("Age", 18, 65, 30)\neducation = st.sidebar.selectbox("Education Level", [\n    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"\n])\noccupation = st.sidebar.selectbox("Job Role", [\n    "Tech-support", "Craft-repair", "Other-service", "Sales",\n    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",\n    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",\n    "Protective-serv", "Armed-Forces"\n])\nhours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)\nexperience = st.sidebar.slider("Years of Experience", 0, 40, 5)\n\n# Build input DataFrame ( must match preprocessing of your training data)\ninput_df = pd.DataFrame({\n    \'age\': [age],\n    \'education\': [education],\n    \'occupation\': [occupation],\n    \'hours-per-week\': [hours_per_week],\n    \'experience\': [experience]\n})\n\nst.write("### 🔎 Input Data")\nst.write(input_df)\n\n# Predict button\nif st.button("Predict Salary Class"):\n    prediction = model.predict(input_df)\n    st.success(f" Prediction: {prediction[0]}")\n\n# Batch prediction\nst.markdown("---")\nst.markdown("#### 📂 Batch Prediction")\nuploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")\n\nif uploaded_file is not None:\n    batch_data = pd.read_csv(uploaded_file)\n    st.write("Uploaded data preview:", batch_data.head())\n    batch_preds = model.predict(batch_data)\n    batch_data[\'PredictedClass\'] = batch_preds\n    st.write(" Predictions:")\n    st.write(batch_data.head())\n    csv = batch_data.to_csv(index=False).encode(\'utf-8\')\n    st.download_button("Download Predictions CSV", csv, file_name=\'predicted_classes.csv\', mime=\'text/csv\')\n\n')
import streamlit as st
st.title("Employee Salary Prediction")


# In[53]:




# In[55]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r"C:\Users\hp\Downloads\adult 3.csv")

# Encode categorical columns
encoder = LabelEncoder()
for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
    data[col] = encoder.fit_transform(data[col].astype(str))

# Define inputs and target
X = data.drop(columns=['income'])
y = data['income']

# Streamlit UI
st.title("Employee Income Prediction App")

# Show raw data
if st.checkbox("Show raw data"):
    st.write(data.head())

# Display boxplot
st.subheader("Boxplot of Educational Number")
fig, ax = plt.subplots()
ax.boxplot(data['educational-num'])
ax.set_title("Educational Number Distribution")
ax.set_ylabel("Value")
st.pyplot(fig)


# In[ ]:




