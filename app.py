#Loading Libraries

import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')
# st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Dentsu-logo_black.svg/2560px-Dentsu-logo_black.svg.png', width=250)
#---------------------------------#
# Model building
def build_model(df):
    #df = df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('**Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    st.markdown('**Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)

    # Build lazy model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

    st.subheader('Model Performance Outputs')

    st.write('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    st.subheader('Plot of Model Performance (Test Dataset)')


    with st.markdown('**R-squared**'):
        # Tall
        predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
        ax1.set(xlim=(0, 1))
    #st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'R2 plot.pdf'), unsafe_allow_html=True)

    with st.markdown('**RMSE (capped at 100)**'):
        # Tall
        predictions_test["RMSE"] = [100 if i > 50 else i for i in predictions_test["RMSE"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
    # st.markdown(imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'RSME plot.pdf'), unsafe_allow_html=True)

    with st.markdown('**Calculation time**'):
        # Tall
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
    # st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'Calculation Time Plot.pdf'), unsafe_allow_html=True)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#

#st.image("https://assets-eu-01.kc-usercontent.com/ac1144d1-44c7-0116-3e50-561fc1db4e3c/3fa8fd5d-19a1-4482-b118-ac0d2950cc8d/merkle-dentsu.png")
st.write("""
# ML Models Comparative App

This app uses *lazypredict* library for building several ML models at once and understanding the outputs like Model Performance- R2 & RSME.


""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""

""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**Uploaded dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        Diabetes dataset
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Diabetes dataset is used as the example.')
        st.write(df.head(5))

#         build_model(df)
