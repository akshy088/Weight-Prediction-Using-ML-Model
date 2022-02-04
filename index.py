
import streamlit as st
import pandas as pd
# from matplotlib import pyplot as plt
# from plotly import graph_objs as go
# from sklearn.linear_model import LinearRegression
import numpy as np
import time
import pickle

# pickle_in = open("decisionTree.pkl","rb")
# dec_cl = pickle.load(pickle_in)

pickle_in = open("logistics_reg.pkl" ,"rb")
dec_cl = pickle.load(pickle_in)

data = pd.read_csv('weight-height.csv')

# adding title
st.title("Weight Analytics")

# adding image
# st.image("HR-Analytics.jpg",width = 800)

# adding sidebar

nav = st.sidebar.radio("Navigation" ,["Home" ,"Prediction"])

if nav == "Home":

    if st.checkbox("Show Table"):
        st.dataframe(data)

    # if st.checkbox("Show data_dictionary Table"):
    #  st.image("data_dictionary.png",width = 800)




if nav == "Prediction":

    st.header("predict weight")

    weight = st.number_input("Enter your height" ,0.00 ,100.00 ,step = 2.00)

    #    pred = dec_cl.predict([[age,b_t_f,b_t_r,dep,m_s_m,m_s_s,sal,ex,j_i,j_s,w_h]])


    val = np.array([weight])
    val=val.reshape(1, -1)
    pred = dec_cl.predict(val)

    if st.button("Predict"):

        progress = st.progress(0)    # this is for progress bar
        for i in range(100):
            time.sleep(0.001)
            progress.progress( i +1)

        # st.success(pred)

        st.success('The output is {}'.format(pred))
        # st.success(f"Your predicted salary is {round(pred)}")






