import streamlit as st

import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt

from pycaret.classification import *




# -------------------------------------------------------------------
# SETTING UP THE APP INTERFACE and READ INPUT DATA
# -------------------------------------------------------------------

# st.title("External Stability of MSE Walls | Reliability Analysis")
st.write("# Liquefaction potential based on CPT test results")
st.write("## Classification models")
st.markdown(
    "A web app to determine if the soil will liquefy based on CPT test results. It uses different classification models "
)

st.sidebar.header("Input parameters")

D_input = st.sidebar.slider(
    "Depth (m)",
    min_value=1.0,
    max_value=15.0,
    value= 5.0,
    #(6.0, 10.0),
    step=0.1,
    format="%.1f",
    )

qc_input = st.sidebar.slider(
    "Cone resistance (MPa)",
    min_value=0.9,
    max_value=25.0,
    value= 10.0,
    step=0.1,
    format="%.1f",
    )

Rf_input = st.sidebar.slider(
    "Rf (%)",
    min_value=0.1,
    max_value=5.0,
    value= 2.0,
    step=0.1,
    format="%.1f",
    )
S1_input = st.sidebar.slider(
    "Effective vertical sttress (kPa)",
    min_value=20.0,
    max_value=300.0,
    value= 50.0,
    step=1.0,
    format="%.0f",
    )

S2_input = st.sidebar.slider(
    "Total vertical sttress (kPa)",
    min_value=20.0,
    max_value=300.0,
    value= 50.0,
    step=1.0,
    format="%.0f",
    )

amax_input = st.sidebar.slider(
    "Max. ground accel. (g)",
    min_value=0.08,
    max_value=0.8,
    value= 0.4,
    step=0.01,
    format="%.02f",
    )

mw_input = st.sidebar.slider(
    "Earthquake magnitude, Mw",
    min_value=6.0,
    max_value=7.6,
    value= 6.5,
    step=0.1,
    format="%.1f",
    )


input_data = {'Values': [D_input, qc_input, Rf_input, S1_input, S2_input, 
               amax_input, mw_input]} 

# index to show on screen
input_index = ['Depth (m)',
              'Cone resistance (MPa)', 
              'Rf (%)', 
              'Effective vertical stress (kPa)',
              'Total vertical stress (kPa)',
              'Max. ground accel. (g)',
              'Earthquake magnitude (Mw)'
              ]
              
input_df = pd.DataFrame(input_data, index=input_index)


st.subheader("Summary of Input parameters")
st.table(input_df)


# use this index, as model is trained with this index
index = ['D', 'q_c', 'R_f', 'S1', 'S2', 'a_max', 'M_w']
df = pd.DataFrame(input_data, index).T



m0 = load_model('final0___5_models___niter_100___19.10.2021')
m1 = load_model('final1___5_models___niter_100___19.10.2021')
m2 = load_model('final2___5_models___niter_100___19.10.2021')
m3 = load_model('final3___5_models___niter_100___19.10.2021')
m4 = load_model('final4___5_models___niter_100___19.10.2021')
m5 = load_model('final5___5_models___niter_100___19.10.2021')

# -------------------------------------------------------------------
# LOAD MODEL AND PREDICT
# -------------------------------------------------------------------

def run_prediction(df, model):
    pred = predict_model(model, df)
    label = float(pred.Label)
    score = float(pred.Score)
    if label == 0:
        score = 1.0 - score
    if label == 1:
        text_label = 'YES'
    else:
        text_label = 'NO'
    
    return label, text_label, score


def compile_predictions(df, m0, m1, m2, m3, m4, m5):
    
    text_labels = []
    labels = []
    scores = []
    
    models = [m0, m1, m2, m3, m4, m5]
    no_of_models = len(models)
    progress_bar = st.progress(0)
    
    for i, m in enumerate(models):
        label, text_label, score = run_prediction(df, m)
        labels.append(label)
        text_labels.append(text_label)
        scores.append(score)
        progress_bar.progress(int((i+1)/no_of_models * 100))
        if i+1 == no_of_models:
            st.write("Calculation completed. Scroll down to see results and graphs")
    
    return labels, text_labels, scores



def process_test_df(df):
    
    df_tmp = df.copy()
    df_tmp['rd'] = np.where(df_tmp['D']<=9.15, 1.0 - 0.00765*df_tmp['D'], 1.174 - 0.0267*df_tmp['D'])
    df_tmp['MSF'] = (df_tmp['M_w']/7.5)**(-2.56)
    df_tmp['CSR'] = 0.65*df_tmp['S2']/df_tmp['S1']*df_tmp['a_max']*df_tmp['rd']/df_tmp['MSF']
    df_tmp['q_c1N'] = (df_tmp['q_c']*1000.0/100.0)/((df_tmp['S1']/100.0)**0.5)
    
    # df_tmp.drop(['rd', 'MSF'], axis=1, inplace=True)
    
    CSR = float(df_tmp.CSR)
    qc1N = float(df_tmp.q_c1N)
    
    return CSR, qc1N
    



def set_printing_defaults():
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib as mpl

    # PRINTING DEFAULTS

    SMALL_SIZE  = 10     #  8
    MEDIUM_SIZE = 12     # 10
    BIGGER_SIZE = 14     # 12
    
    plt.rc('font', family='calibri') # set font type [sans serif, serif, Tahoma, DejaVu Sans, Lucida Grande, Verdana]
    plt.rc('font', size=SMALL_SIZE)            # controls default text sizes
    
    plt.rc('axes', titlesize=SMALL_SIZE)       # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)      # fontsize of the x and y labels
    plt.rc('axes', linewidth=0.5)
    # plt.rc('axes', labelweight='bold')       # fontsize of the x and y labels
    
    plt.rc('legend', fontsize=SMALL_SIZE)      # legend fontsize
    
    
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='in')  # fontsize of the tick labels ('in', 'out', 'inout')
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='in')  # fontsize of the tick labels ('in', 'out', 'inout')
    plt.rc('xtick',   top='False')     # top tick   ('True' or 'False')
    plt.rc('ytick', right='False')     # right tick ('True' or 'False')
    
    
    plt.rc('figure',dpi=1200)  # figure dpi
    plt.rc('figure', titlesize=BIGGER_SIZE)    # fontsize of the figure title
    
    plt.rc('savefig',dpi=1200) # saved figure's dpi
    
    plt.rc('grid', color='0.5', ls='-', lw=0.5)  # grid properties  ls=':'
    
    mpl.rcParams['legend.frameon'] = 'True'     # TURNOFF LEGEND BORDER
    
    
    # change axes color
    import matplotlib as mpl
    
    grey = '#808080'
    
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['axes.edgecolor'] = grey
    mpl.rcParams['xtick.color'] = grey
    mpl.rcParams['ytick.color'] = grey
    mpl.rcParams['axes.labelcolor'] = "black"

    

def plot_historical_data(test_df):
    
    # import data
    path = 'E:/04 ML Liquefaction/Python code/juang_2003_data_with_CSR.csv'
    # path = 'juang_2003_data.csv'
    dataset = pd.read_csv(path, sep=',')
    
    # get CSR and qc1N from test data
    CSR, qc1N = process_test_df(test_df)
    
    
    # set printing defaults
    set_printing_defaults()
    
    alpha = 0.7
    
    fig,ax = plt.subplots(1, 1, figsize=(5.0, 5.0), dpi=600, facecolor='white')
    
    ax.plot(dataset.loc[dataset['Class']==1]['q_c1N'], 
            dataset.loc[dataset['Class']==1]['CSR'],
            ls='None', marker='o', ms=6.5, mec='red', 
            mew=0.8, mfc='salmon', alpha=alpha, label='Liquified')
    
    ax.plot(dataset.loc[dataset['Class']==0]['q_c1N'], 
            dataset.loc[dataset['Class']==0]['CSR'],
            ls='None', marker='D', ms=5.5, mec='blue', 
            mew=0.8, mfc='royalblue', alpha=alpha, label='Not Liquified')

    ax.plot(qc1N, CSR,
            ls='None', marker='d', ms=10, mec='black', 
            mew=0.8, mfc='gold', alpha=0.8)
    
        
    ax.set_xlabel(r'$q_{c1N}$', fontsize=12)
    ax.set_ylabel('CSR', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), color='black')
    plt.setp(ax.get_yticklabels(), color='black')
    

    ax2 = ax.twinx()
    ax2.plot(np.NaN, np.NaN,
             ls='None', marker='d', ms=7, mec='black',
             mew=0.8, mfc='gold', alpha=1, label='Test data')
    ax2.get_yaxis().set_visible(False)

    ax.legend(title='Historical data', fontsize=10,
              bbox_to_anchor=(1.01,1.0), loc=2, borderaxespad=0.07)
    ax2.legend(bbox_to_anchor=(1.01,0.83), loc=2, borderaxespad=0.07)
    


    
    
    
    st.write(fig)

      
       
# ====================================================
# now run the APP
# ====================================================

submit_1 = st.button("Predict")

if submit_1:

    labels, text_labels, scores = compile_predictions(df, m0, m1, m2, m3, m4, m5)
    
    model_index = ['Random Forest Classifier',
                   'CatBoost Classifier',
                   'Extra Trees Classifier',
                   'Extreme Gradient Boosting',
                   'Gradient Boosting Classifier',
                   'Blended models']

    preds_df = pd.DataFrame({'MODELS': model_index, 'Liquefaction': text_labels,
                              'Probability': scores}) # , index=model_index
    
    st.subheader('Prediction by classification models')
    st.table(preds_df)


    st.subheader("Plot of historical data")
    plot_historical_data(df)
    
    