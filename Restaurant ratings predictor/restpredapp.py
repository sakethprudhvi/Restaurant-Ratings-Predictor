import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
st.set_page_config(layout = "wide")
scaler = joblib.load("Scaler.pkl")
st.title("Restaurant Rating Prediction App")
st.caption("This app helps you to predict a restaurants review class")
st.divider()
averagecost = st.number_input("Please enter the estimated cost for two", min_value = 50, max_value = 999999, value = 1000, step =200 )
tablebooking  = st.selectbox("Restaurant has table booking?", ["YES","NO"])
onlinedelivery = st.selectbox("Restaurant has online booking?", ["YES","NO"])
pricerange = st.selectbox("What is the price range (1 Cheapest, 4 Most Expensive)", [1,2,3,4])
predictbutton = st.button("Predict the review!")

st.divider()
model = joblib.load("mlmodel.pkl")
bookingstatus = 1 if tablebooking == "YES" else 0
deliverystatus = 1 if tablebooking == "YES" else 0
values= [[averagecost,bookingstatus, deliverystatus, pricerange]]
my_X_values = np.array(values)
X = scaler.transform(my_X_values)
#averagecost_scaled = scaler.transform(averagecost)
#tablebooking = scaler.fit_transform(tablebooking)
#onlinedelivery = scaler.fit_transform(onlinedelivery)
#pricerange = scaler.fit_transform(pricerange)

if predictbutton:
    st.snow()
    prediction = model.predict(X)
    #st.write(prediction)
    #Above 2 Below 2.5 Poor
    #Above 2.5 Below 3.5 Average
    #Above 3.5 Below 4.0 Good
    #Above 4 Below 4.5 Very Good
    #Above 4.5 Excellent
    
    if prediction < 2.5:
        st. write("Poor")   
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")    
    