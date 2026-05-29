import streamlit as st
import plotly.express as px
from src.inference import stream

st.set_page_config(page_title='Real-Time Network Intrusion Detection',layout='wide')
st.title('Real-Time Anomaly Detection Dashboard')
rate=st.sidebar.selectbox('Replay Rate',[100,500,1000],index=0)
limit=st.sidebar.slider('Records',50,1000,200)

if st.button('Run Stream Simulation'):
    df=stream(rate=rate,limit=limit)
    st.metric('Anomalies',int(df['anomaly'].sum()))
    st.metric('Average Latency (ms)',round(df['latency_ms'].mean(),3))
    st.plotly_chart(px.line(df,x='record',y='error',title='Reconstruction Error'))
    st.plotly_chart(px.line(df,x='record',y='latency_ms',title='Inference Latency'))
    st.dataframe(df.tail(25),use_container_width=True)
