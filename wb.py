#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import joblib
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Decision system for combined hepatocellular-carcinoma distant metastasis: A retrospective observational study based on machine learning')
st.title('Decision system for combined hepatocellular-carcinoma distant metastasis: A retrospective observational study based on machine learning')

#%%set varixgbles selection
st.sidebar.markdown('## Varixgbles')

N =  st.sidebar.selectbox("Node",('No', 'Yes'),index=0)
Surgery = st.sidebar.selectbox("Surgery",('No', 'Yes'),index=0)
Age =  st.sidebar.slider("Age (year)", 5,95,value = 65, step = 1)
Grade = st.sidebar.selectbox("Grade",('Well differentiated; Grade I',
                                      'Moderately differentiated; Grade II',
                                      'Poorly differentiated; Grade III',
                                      'Undifferentiated; aplastic; Grade IV'),index=2)
Primary_Site = st.sidebar.selectbox("Primary site",('Liver','Intrahepatic bile duct','Other'),index=0)
Race = st.sidebar.selectbox("Race", ('White',"Black","Other (American Indian, Asian/Pacific Islander)"), index = 0)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'No':0,
       'Yes':1,
       'Well differentiated; Grade I':1,
       'Moderately differentiated; Grade II':2,
       "Poorly differentiated; Grade III":3,
       "Undifferentiated; aplastic; Grade IV":4,
       "Liver":1,
       'Intrahepatic bile duct':2,
       'Other':3,
       'White':1,
       "Black":2,
       "Other (American Indian, Asian/Pacific Islander)":3
}
N =map[N]
Surgery = map[Surgery]
Grade = map[Grade]
Primary_Site = map[Primary_Site]
Race = map[Race]

# 数据读取，特征标注
#%%load model
xgb_model = joblib.load(r'D:\厦门大学\合作\刘荣强\混合型肝癌机器学习\xgb_model.pkl')

#%%load data
hp_train = pd.read_excel(r"F:\合作者\刘荣强\混合型肝癌SEER联合医院数据\SEER_data_2000_2020.xlsx", sheet_name="train")
features = ['N',
            'Surgery',
            'Age',
            'Grade',
            'Primary_Site',
            'Race']
target = "M"
y = np.array(hp_train[target])
sp = 0.5

is_t = (xgb_model.predict_proba(np.array([[N,Surgery,Age,Grade,Primary_Site,Race]]))[0][1])> sp
prob = (xgb_model.predict_proba(np.array([[N,Surgery,Age,Grade,Primary_Site,Race]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Group'
else:
    result = 'Low Risk Group'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Group':
        st.balloons()
    st.markdown('## Probability of High Risk group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[N,Surgery,Age,Grade,Primary_Site,Race]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = xgb_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of XGB model')
    fig, ax = plt.subplots(figsize=(12, 6))
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of XGB model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8))
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of XGB model')
    xgb_prob = xgb_model.predict(X)
    cm = confusion_matrix(y, xgb_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of XGB model")
    disp1 = plt.show()
    st.pyplot(disp1)

