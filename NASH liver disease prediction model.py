#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression - Predict if a person will develop Nonalcoholic steatohepatitis(NASH)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The choosen algorithm
from sklearn.linear_model import LogisticRegression

# Spliting the data to train and test data
from sklearn.model_selection import train_test_split

# Feature selection and Standardisation
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv(r'C:\Users\ivanr\OneDrive\Desktop\Main(2nd sem) Special Problem\Datasets\Liver Detection Data\Main Data Frame\steatotest_hepatitis_dataset.csv')
print(len(df))
df.head(n = 10)


# # Data Preprocessing

# ### Classify Cirrhosis and Fibrosis as NASH

# 1 is positive in NASH and 0 is Negative in NASH

# In[3]:


df['steatosis_score_0'] = df['steatosis_score_0'].map({0:0,1:1,2:1,3:1}) 
df = df.rename(columns = {"steatosis_score_0": "NASH"})
df.head(n=10)


# # Check data imbalance

# In[4]:


temp = df['NASH'].value_counts()
temp_df = pd.DataFrame({'NASH':temp.index,'values': temp.values})
print(sns.barplot(x= 'NASH', y= 'values', data = temp_df))


# # Data Analysis

# In[6]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), cmap="RdBu", annot=True)
plt.show()


# # Explore the linearity of features based on the heatmap

# triglycerides_0 and vldl_0

# In[7]:


plt.scatter(df["triglycerides_0"], df["vldl_0"])
plt.xlabel('triglycerides_0')
plt.ylabel('vldl_0')
plt.title('triglycerides_0 and vldl_0')
plt.show()


# ldl_0 and total_cholesterol_0

# In[8]:


plt.scatter(df["ldl_0"], df["total_cholesterol_0"])
plt.xlabel('total_cholesterol_0')
plt.ylabel('ldl_0 ')
plt.title('total_cholesterol_0 and ldl_0')
plt.show()


# neutrophils_0 and lymphocytes_0

# In[9]:


plt.scatter(df["ldl_0"], df["total_cholesterol_0"])
plt.xlabel('neutrophils_0')
plt.ylabel('lymphocytes_0 ')
plt.title('neutrophils_0 and lymphocytes_0')
plt.show()


# # Person's Correlation

# In[10]:


corr, p_value = pearsonr(df['triglycerides_0'], df['vldl_0'])
print(f"Pearson correlation coefficient of triglycerides_0 and vldl_0: {corr:.2f}")

corr, p_value = pearsonr(df['ldl_0'], df['total_cholesterol_0'])
print(f"Pearson correlation coefficient of ldl_0 and total_cholesterol_0: {corr:.2f}")

corr, p_value = pearsonr(df['neutrophils_0'], df['lymphocytes_0'])
print(f"Pearson correlation coefficient of neutrophils_0 and lymphocytes_0: {corr:.2f}")


# # Drop the redundant data and data that didn't affect our model

# In[11]:


columns_to_drop = ["triglycerides_0","ldl_0","neutrophils_0","patient_id","patient_id.1",
                  'wbc_0','monocytes_0', 'creatinine_0','basophils_0','lymphocytes_0']
                  
for col in columns_to_drop:
    del df[col]


# # Checkpoint to preprocess data

# In[12]:


df_preprocess = df.copy()
df_preprocess.head(n = 5)


# # Select Input and target

# In[13]:


data_with_target = df_preprocess.copy()


# In[14]:


targets = data_with_target.iloc[:, 0]


# In[15]:


unscaled_inputs = data_without_target = df.iloc[:, 1:]


# # Standardize the data

# In[16]:


from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

    def __repr__(self):
        return f"CustomScaler(columns={self.columns}, mean_={self.mean_}, var_={self.var_})"


# In[17]:


unscaled_inputs.columns.values


# In[18]:


columns_to_scale = ['age_0', 'afp_0', 'dcp_0','total_cholesterol_0', 'hdl_0', 
                    'vldl_0','albumin_0', 'alkaline_phos_0', 'sgpt_0', 'sgot_ast_0',
                    'total_bilirubin_0','eosinophils_0', 'platelets_0', 'inr_0']


# In[19]:


steatosis_scaler = CustomScaler(columns_to_scale)


# In[20]:


steatosis_scaler.fit(unscaled_inputs)


# In[21]:


scaled_inputs = steatosis_scaler.transform(unscaled_inputs)


# # Split the data into train & test and shuffle

# In[22]:


x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, shuffle = True, random_state = 20)


# # Model Formulation

# ### Logistic Regression

# In[23]:


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


# In[24]:


log_reg.score(x_train, y_train)


# In[25]:


log_reg.score(x_test, y_test)


# # Probability Prediction of the Test Data Set

# In[26]:


# Assuming the arrays x_test and y_test are already defined
y_pred = log_reg.predict(x_test)
y_prob = log_reg.predict_proba(x_test)[:, 1]

# Create DataFrame with test data, predictions, and probabilities
results_df = pd.DataFrame(x_test, columns=[ 'hbsag_status_1', 'gender_1', 'age_0', 'afp_0', 'dcp_0','total_cholesterol_0', 'hdl_0', 
                                            'vldl_0','albumin_0', 'alkaline_phos_0', 'sgpt_0', 'sgot_ast_0',
                                           'total_bilirubin_0','eosinophils_0', 'platelets_0', 'inr_0'])

# Assuming the arrays x_test and y_test are already defined
y_prob = log_reg.predict_proba(x_test)[:, 1]
y_pred = log_reg.predict(x_test)



# Concatenate the dataframes horizontally
result_df = pd.concat([results_df, y_test], axis=1)
results_df['Actual'] = y_test
results_df['Probability'] = y_prob
results_df['Prediction'] = y_pred



results_df


# results_df[['age_0', 'Probability', 'Prediction']].to_csv('age_prediction.csv', index=False)
# results_df[['gender_1','sgpt_0','Probability', 'Prediction']].to_csv('sgpt_prediction.csv', index=False)
# 

# ### Manually check accuracy

# In[27]:


model_outputs = log_reg.predict(x_test)
np.sum((model_outputs == y_test))/ model_outputs.shape[0]


# # Finding the intercept and coefficients

# In[28]:


log_reg.intercept_


# In[29]:


log_reg.coef_


# In[30]:


feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame (columns=["Feature name"], data = feature_name)
summary_table["Coefficients"] = np.transpose(log_reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ["Intercept", log_reg.intercept_[0]]
summary_table = summary_table.sort_index()


# # Interpreting the Coeffients

# In[31]:


summary_table["Odds_ratio"] = np.exp(summary_table.Coefficients)
summary_table.sort_values('Odds_ratio', ascending = False)
summary_table

