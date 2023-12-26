%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OrdinalEncoder , PolynomialFeatures ,RobustScaler
from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error , r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.express as px
pd.options.display.float_format = '{:,.2f}'.format
# Read Data
# _______________________________________________________________________________________________________________________________________________________________________________________________________
df_new = pd.read_csv('new_df.csv',index_col=0)
pd.options.display.float_format ='{:,.2f}'.format
st.set_page_config (page_title = 'Bike Sharing Prediction (Per Houre) 🚲' , layout = "wide" , page_icon = '📊 🚲')
st.title("Bike Sharing Prediction (Per Houre) 🚲")

# Sidebar
brief = st.sidebar.checkbox(":red[Brief about Project]")
Planning = st.sidebar.checkbox(":orange[About Project]")
About_me = st.sidebar.checkbox(":green[About me]")

if brief:
    st.sidebar.header(":red[Brief about Project]")
    st.sidebar.write("""
    * Bike sharing systems are new generation of traditional bike rentals where whole process from membership
      rental and return back has become automatic. 
      Through these systems, user is able to easily rent a bike from a particular position and return back at another position. 
      Currently, there are about over 500 bike-sharing programs around the world which is composed of over 500 thousands bicycles. 
      Bike-sharing programs are becoming more and more popular in recent Today,.
    * :red[So let us see the insights 👀.]
    """)
# ______________________________________________________________________________________________________________________________________________________________________________________________________

if Planning :
    st.sidebar.header(":orange[About Project]")
    st.sidebar.subheader ('BIKE SHARING Prediction Per Houre 📊')
    st.sidebar.write("""
    * This is my Final Project during my Data Science Diploma @Epsilon AI. 
    * In this Project we have 3 Tabes: ('Expolration🔎','Insight💡' , 'Prediction 👨‍👩‍👦‍👦').

    * Data Source:
        1) mendeley : https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
        2) Kaggle : https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset
    """)
    st.sidebar.write("""
    * Data Details:
        * Columns : 16 Features
        * Instance : 17379 Operations
    """)
# ______________________________________________________________________________________________________________________________________________________________________________________________________

if About_me :
    st.sidebar.header(":green[About me]")
    st.sidebar.write("""
    - Osama SAAD
    - Student : Data Scaience & Machine Learning  @Epsilon AI
    - Infor ERP (EAM/M3) key Business.User | Infor DMS, Assets and System Control Supervisor @ Ibnsina Pharma
    - LinkedIn: 
        https://www.linkedin.com/in/ossama-ahmed-saad-525785b2
    - Github : 
        https://github.com/OsamaSamnudi
    """)
# ______________________________________________________________________________________________________________________________________________________________________________________________________

# Tabs
Exploration , Insight , Custom_EDA , Prediction  = st.tabs(['Exploration 🔎','Insight 💡' , 'Custom EDA for User👨‍👩‍👦‍👦' , 'Prediction 📈'])
with Exploration:
    with st.container():
        st.header('Exploration🔎')
        About_Data = st.checkbox(":blue[🔻 About Data]")
        if About_Data:
            st.write("""
            - instant: record index
            - dteday : date
            - season : season (1 = springer & 2 = summer & 3 = fall & 4 = winter)
            - yr : year (0 = 2011 & 1 = 2012)
            - mnth : month ( 1 to 12)
            - hr : hour (0 to 23)
            - holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
            - weekday : day of the week
            - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
            - weathersit :
                - 1) Clear, Few clouds, Partly cloudy, Partly cloudy
                - 2) Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
                - 3) Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
                - 4) Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
            - temp : Normalized temperature in Celsius. The values are divided to 41 (max)
            - atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
            - hum: Normalized humidity. The values are divided to 100 (max)
            - windspeed: Normalized wind speed. The values are divided to 67 (max)
            - casual: count of casual users
            - registered: count of registered users
            - cnt: count of total rental bikes including both casual and registered
            """)
        Data_Sample = st.checkbox(":blue[🔻 Data_Sample]")
        if About_Data:
            st.dataframe(data= df_new.sample(10), use_container_width=False)
#########################################################################################################################################
        Correlation = st.checkbox(":blue[🔻 Correlation]")
        if Correlation: 
            corr = df_new.select_dtypes(include='number').corr()
            IMSHOW = px.imshow(corr , width=1100,height=1100,text_auto=True,title='Correlation' , color_continuous_scale='rdbu')
            st.plotly_chart(IMSHOW , use_container_width=True,theme="streamlit")
#########################################################################################################################################
        Distribution = st.checkbox(":blue[🔻 Distribution]")
        if Distribution:
            st.subheader('✔️ Distribution of Numerical Features : temp , atemp , hum , windspeed , casual , registered , cnt')
####################################################
            col_1 , col_2 = st.columns([10 , 10])
            with col_1:
                st.markdown('✔️ temp Distribution :')
                fig_1 = px.histogram(df_new , x = 'temp'  , text_auto=True , marginal='box')
                st.plotly_chart(fig_1 , use_container_width=True,theme="streamlit")
            with col_2:
                st.markdown('✔️ atemp Distribution :')
                fig_2 = px.histogram(df_new , x = 'atemp' , text_auto=True , marginal='box')
                st.plotly_chart(fig_2 , use_container_width=True,theme="streamlit")
####################################################
            col_3 , col_4 = st.columns([10 , 10])
            with col_3:
                st.markdown('✔️ hum Distribution :')
                fig_3 = px.histogram(df_new , x = 'hum', text_auto=True , marginal='box')
                st.plotly_chart(fig_3 , use_container_width=True,theme="streamlit")
            with col_4:
                st.markdown('✔️ windspeed Distribution :')
                fig_4 = px.histogram(df_new , x = 'windspeed', text_auto=True , marginal='box')
                st.plotly_chart(fig_4 , use_container_width=True,theme="streamlit")
####################################################
            col_5 , col_6 = st.columns([10 , 10])
            with col_5:
                st.markdown('✔️ casual Distribution :')
                fig_5 = px.histogram(df_new , x = 'casual' , text_auto=True , marginal='box')
                st.plotly_chart(fig_5 , use_container_width=True,theme="streamlit")
            with col_6:
                st.markdown('✔️ registered Distribution :')
                fig_6 = px.histogram(df_new , x = 'registered' , text_auto=True , marginal='box')
                st.plotly_chart(fig_6 , use_container_width=True,theme="streamlit")
####################################################
            st.markdown('✔️✔️ Finally Distribution ofTarget : cnt (Count of Rentals)')
            fig_7 = px.histogram(df_new , x = 'cnt' , text_auto=True , marginal='box')
            st.plotly_chart(fig_7 , use_container_width=True,theme="streamlit")
#########################################################################################################################################
        Scatter = st.checkbox(":blue[🔻 Scatter]")
        if Scatter:
            st.subheader('✔️ Scatter for : casual , registered , cnt (vs) season , weathersit , time_preiod')
####################################################
            col_8 , col_9 , col_10 = st.columns([10 , 10 , 10])
            with col_8:
                st.markdown('✔️ (cnt) vs (casual) Grouping_by [season]')
                fig_8 = px.scatter(df_new , x = 'casual' , y = 'cnt' , color = 'season')
                st.plotly_chart(fig_8 , use_container_width=True,theme="streamlit")
            with col_9:
                st.markdown('✔️ (cnt) vs (casual) Grouping_by [weathersit]')
                fig_9 = px.scatter(df_new , x = 'casual' , y = 'cnt' , color = 'weathersit')
                st.plotly_chart(fig_9 , use_container_width=True,theme="streamlit")
            with col_10:
                st.markdown('✔️ (cnt) vs (casual) Grouping_by [time_preiod]')
                fig_10 = px.scatter(df_new , x = 'casual' , y = 'cnt' , color = 'time_preiod')
                st.plotly_chart(fig_10 , use_container_width=True,theme="streamlit")
####################################################
            col_11 , col_12 , col_13= st.columns([10 , 10 , 10])
            with col_11:
                st.markdown('✔️ (cnt) vs (registered) Grouping_by [season]')
                fig_11 = px.scatter(df_new , x = 'registered' , y = 'cnt' , color = 'season')
                st.plotly_chart(fig_11 , use_container_width=True,theme="streamlit")
            with col_12:
                st.markdown('✔️ (cnt) vs (registered) Grouping_by [weathersit]')
                fig_12 = px.scatter(df_new , x = 'registered' , y = 'cnt' , color = 'weathersit')
                st.plotly_chart(fig_12 , use_container_width=True,theme="streamlit")
            with col_13:
                st.markdown('✔️ (cnt) vs (registered) Grouping_by [time_preiod]')
                fig_13 = px.scatter(df_new , x = 'registered' , y = 'cnt' , color = 'time_preiod')
                st.plotly_chart(fig_13 , use_container_width=True,theme="streamlit")

# ______________________________________________________________________________________________________________________________________________________________________________________________________
with Insight:
    with st.container():
        st.subheader("🚩 DATA UNDERSTANDING & INSIGHTS...")
        Totals_Report = []
        for i in ['sum','mean','count']:
            Totals = {'name':['casual' , 'registered' , 'cnt'],
                '2011':[df_new[(df_new.yr == 2011) & (df_new.casual != 0)].casual.agg(i).round(2), 
                        df_new[(df_new.yr == 2011) & (df_new.registered != 0)].registered.agg(i).round(2),
                        df_new[(df_new.yr == 2011) & (df_new.cnt != 0)].cnt.agg(i).round(2) ] ,
                '2012':[df_new[(df_new.yr == 2012) & (df_new.casual != 0)].casual.agg(i).round(2), 
                        df_new[(df_new.yr == 2012) & (df_new.registered != 0)].registered.agg(i).round(2),
                        df_new[(df_new.yr == 2012) & (df_new.cnt != 0)].cnt.agg(i).round(2)] }
            Totals_Report.append(Totals)
        # DataFrames
        Totals_Report_sum = pd.DataFrame(Totals_Report[0])
        Totals_Report_avg = pd.DataFrame(Totals_Report[1]) 
        Totals_Report_count = pd.DataFrame(Totals_Report[2]) 
    col_i_1 , col_i_2 , col_i_3 = st.columns([40 , 40 , 40])
    with col_i_1:
        st.write('✔️ Total Rents')
        fig_S = px.histogram(Totals_Report_sum,x='name',y=['2011','2012'],barmode='group',text_auto=True,width=900,height=500,
                 title = f'Total of Casual, Registred, cnt')
        st.plotly_chart(fig_S , use_container_width=True,theme="streamlit")
    with col_i_2:
        st.write('✔️ Avg Rents')
        fig_V = px.histogram(Totals_Report_avg,x='name',y=['2011','2012'],barmode='group',text_auto=True,width=900,height=500,
                 title = f'Avg of Casual, Registred, cnt')
        st.plotly_chart(fig_V , use_container_width=True,theme="streamlit")
    with col_i_3:
        st.write('✔️ Count Rents')
        fig_C = px.histogram(Totals_Report_count,x='name',y=['2011','2012'],barmode='group',text_auto=True,width=900,height=500,
                 title = f'Count of Casual, Registred, cnt')
        st.plotly_chart(fig_C , use_container_width=True,theme="streamlit")

    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per year")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            msk = df_new.groupby(['yr']).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.pie(msk , names = 'yr' , values = 'casual',title = 'avg (casual) per year' ).update_traces(textinfo='percent+value')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk_15 = df_new.groupby(['yr']).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig_15 = px.pie(msk_15 , names = 'yr' , values = 'registered',title = 'avg (registered) per year' ).update_traces(textinfo='percent+value')
            st.plotly_chart(fig_15 , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby(['yr']).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.pie(msk , names = 'yr' , values = 'cnt' ,title = 'avg (cnt) per year' ).update_traces(textinfo='percent+value')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per season")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            Var = 'season'
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.pie(msk , names = Var , values = 'casual',title = f'avg (casual) per {Var}' ).update_traces(textinfo='percent+value')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.pie(msk , names = Var , values = 'registered',title = f'avg (registered) per {Var}' ).update_traces(textinfo='percent+value')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.pie(msk , names = Var , values = 'cnt' ,title = f'avg (cnt) per {Var}' ).update_traces(textinfo='percent+value')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per mnth")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            Var = 'mnth'
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'casual',title = f'avg (casual) per {Var}',text_auto=True,color='casual')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'registered',title = f'avg (registered) per {Var}',text_auto=True,color='registered')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'cnt' ,title = f'avg (cnt) per {Var}',text_auto=True,color='cnt')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per time_preiod")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            Var = 'time_preiod'
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'casual',title = f'avg (casual) per {Var}',text_auto=True,color='casual')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'registered',title = f'avg (registered) per {Var}',text_auto=True,color='registered')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'cnt' ,title = f'avg (cnt) per {Var}',text_auto=True,color='cnt')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")         
    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per weathersit")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            Var = 'weathersit'
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'casual',title = f'avg (casual) per {Var}',text_auto=True,color='casual')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'registered',title = f'avg (registered) per {Var}',text_auto=True,color='registered')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'cnt' ,title = f'avg (cnt) per {Var}',text_auto=True,color='cnt')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")        
    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per weekday")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            Var = 'weekday'
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'casual',title = f'avg (casual) per {Var}',text_auto=True,color='casual')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'registered',title = f'avg (registered) per {Var}',text_auto=True,color='registered')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'cnt' ,title = f'avg (cnt) per {Var}',text_auto=True,color='cnt')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per hour")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            Var = 'hr'
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'casual',title = f'avg (casual) per {Var}',text_auto=True,color='casual')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'registered',title = f'avg (registered) per {Var}',text_auto=True,color='registered')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'cnt' ,title = f'avg (cnt) per {Var}',text_auto=True,color='cnt')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per workingday")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            Var = 'workingday'
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'casual',title = f'avg (casual) per {Var}',text_auto=True,color='casual')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'registered',title = f'avg (registered) per {Var}',text_auto=True,color='registered')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'cnt' ,title = f'avg (cnt) per {Var}',text_auto=True,color='cnt')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    with st.container():
        st.write("✔️ avg (registered, casual, cnt) per holiday")
        col_i_1 , col_i_2 , col_i_3 = st.columns([30 , 30 , 30])
        with col_i_1:
            Var = 'holiday'
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'casual',title = f'avg (casual) per {Var}',text_auto=True,color='casual')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_2:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'registered',title = f'avg (registered) per {Var}',text_auto=True,color='registered')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            msk = df_new.groupby([Var]).agg({'casual':'mean' , 'registered':'mean', 'cnt':'mean'}).reset_index()
            fig = px.bar(msk , x = Var , y = 'cnt' ,title = f'avg (cnt) per {Var}',text_auto=True,color='cnt')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")  
    # weathersit
    with st.container():
        col_i_1 , col_i_2 , col_i_3 = st.columns([40 , 1 , 40])
        with col_i_1:
            color = 'weathersit'
            msk = df_new[df_new.yr == 2011].groupby(['yr_mnth','weathersit','season','time_preiod','weekday','holiday','mnth']).cnt.mean().reset_index()
            fig = px.histogram(msk,y="cnt",x="yr_mnth",color=color,barmode= 'group',title=f'cnt per month per ({color}) in 2011',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            color = 'weathersit'
            msk = df_new[df_new.yr == 2012].groupby(['yr_mnth','weathersit','season','time_preiod','weekday','holiday','mnth']).cnt.mean().reset_index()
            fig = px.histogram(msk,y="cnt",x="yr_mnth",color=color,barmode= 'group',title=f'cnt per month per ({color}) in 2012',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    # season
    with st.container():
        col_i_1 , col_i_2 , col_i_3 = st.columns([40 , 1 , 40])
        with col_i_1:
            color = 'season'
            msk = df_new[df_new.yr == 2011].groupby(['yr_mnth','weathersit','season','time_preiod','weekday','holiday','mnth']).cnt.mean().reset_index()
            fig = px.histogram(msk,y="cnt",x="yr_mnth",color=color,barmode= 'group',title=f'cnt per month per ({color}) in 2011',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            color = 'season'
            msk = df_new[df_new.yr == 2012].groupby(['yr_mnth','weathersit','season','time_preiod','weekday','holiday','mnth']).cnt.mean().reset_index()
            fig = px.histogram(msk,y="cnt",x="yr_mnth",color=color,barmode= 'group',title=f'cnt per month per ({color}) in 2012',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    # time_preiod
    with st.container():
        col_i_1 , col_i_2 , col_i_3 = st.columns([40 , 1 , 40])
        with col_i_1:
            color = 'time_preiod'
            msk = df_new[df_new.yr == 2011].groupby(['yr_mnth','weathersit','season','time_preiod','weekday','holiday','mnth']).cnt.mean().reset_index()
            fig = px.histogram(msk,y="cnt",x="yr_mnth",color=color,barmode= 'group',title=f'cnt per month per ({color}) in 2011',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            color = 'time_preiod'
            msk = df_new[df_new.yr == 2012].groupby(['yr_mnth','weathersit','season','time_preiod','weekday','holiday','mnth']).cnt.mean().reset_index()
            fig = px.histogram(msk,y="cnt",x="yr_mnth",color=color,barmode= 'group',title=f'cnt per month per ({color}) in 2012',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
            
    with st.container():
        st.subheader("📌 TARGET EXPLORATION & INSIGHTS...")
        col_i_1 , col_i_2 , col_i_3 = st.columns([40 , 1 , 40])
        with col_i_1:
            Var = 'yr'
            msk = df_new.groupby([Var,'workingday'])['cnt'].mean().reset_index()
            fig = px.histogram(msk,y="cnt",x=Var,color='workingday',barmode= 'group',title=f'AVG Rental Count of (({Var})) Grouping By (workingday flag)',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            Var_1 = 'season'
            msk = df_new.groupby([Var_1,'workingday'])['cnt'].mean().reset_index()
            fig = px.histogram(msk,y="cnt",x=Var_1,color='workingday',barmode= 'group',title=f'AVG Rental Count of (({Var_1})) Grouping By (workingday flag)',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        
    with st.container():
        col_i_1 , col_i_2 , col_i_3 = st.columns([40 , 1 , 40])
        with col_i_1:
            Var = 'mnth'
            msk = df_new.groupby([Var,'workingday'])['cnt'].mean().reset_index()
            fig = px.histogram(msk,y="cnt",x=Var,color='workingday',barmode= 'group',title=f'AVG Rental Count of (({Var})) Grouping By (workingday flag)',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            Var_1 = 'weekday'
            msk = df_new.groupby([Var_1,'workingday'])['cnt'].mean().reset_index()
            fig = px.bar(msk,y="cnt",x=Var_1,color='workingday',title=f'AVG Rental Count of (({Var_1})) Grouping By (workingday flag)',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
    with st.container():
        col_i_1 , col_i_2 , col_i_3 = st.columns([40 , 1 , 40])
        with col_i_1:
            Var = 'time_preiod'
            msk = df_new.groupby([Var,'workingday'])['cnt'].mean().reset_index()
            fig = px.histogram(msk,y="cnt",x=Var,color='workingday',barmode= 'group',title=f'AVG Rental Count of (({Var})) Grouping By (workingday flag)',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
        with col_i_3:
            Var_1 = 'weathersit'
            msk = df_new.groupby([Var_1,'workingday'])['cnt'].mean().reset_index()
            fig = px.histogram(msk,y="cnt",x=Var_1,color='workingday',barmode= 'group',title=f'AVG Rental Count of (({Var_1})) Grouping By (workingday flag)',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")    
    with st.container():
        col_i_1 , col_i_2 , col_i_3 = st.columns([40 , 1 , 40])
        with col_i_1:
            Var = 'holiday'
            msk = df_new.groupby([Var,'workingday'])['cnt'].mean().reset_index()
            fig = px.histogram(msk,y="cnt",x=Var,color='workingday',barmode= 'group',title=f'AVG Rental Count of (({Var})) Grouping By (workingday flag)',text_auto=True)
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")

    with st.container():
        st.subheader("📆 TIME PERIOD EXPLORATION & INSIGHTS...")
        Var = 'weathersit'
        msk = df_new.groupby(['yr_mnth',Var])[['cnt']].mean().reset_index().sort_values(['yr_mnth'])
        fig = px.line(msk , x='yr_mnth' , y='cnt',color=Var,title=f"Time Series for ({Var})")
        st.plotly_chart(fig , use_container_width=True,theme="streamlit")  
    with st.container():
        Var_1 = 'time_preiod'
        msk = df_new.groupby(['yr_mnth',Var_1])[['cnt']].mean().reset_index().sort_values(['yr_mnth'])
        fig = px.line(msk , x='yr_mnth' , y='cnt',color=Var_1,title=f"Time Series for ({Var_1})")
        st.plotly_chart(fig , use_container_width=True,theme="streamlit")  
    with st.container():
        Var_1 = 'workingday'
        msk = df_new.groupby(['yr_mnth',Var_1])[['cnt']].mean().reset_index().sort_values(['yr_mnth'])
        fig = px.line(msk , x='yr_mnth' , y='cnt',color=Var_1,title=f"Time Series for ({Var_1})")
        st.plotly_chart(fig , use_container_width=True,theme="streamlit")  
    with st.container():
        Var_1 = 'holiday'
        msk = df_new.groupby(['yr_mnth',Var_1])[['cnt']].mean().reset_index().sort_values(['yr_mnth'])
        fig = px.line(msk , x='yr_mnth' , y='cnt',color=Var_1,title=f"Time Series for ({Var_1})")
        st.plotly_chart(fig , use_container_width=True,theme="streamlit")  
    with st.container(): # 'casual','registered','cnt'
        Var_1 = 'yr'
        msk = df_new.groupby(['hr',Var_1])[['casual']].mean().reset_index().sort_values(['hr'])
        fig = px.line(msk , x='hr' , y='casual',color=Var_1,title=f"Time Series (casual) Hourly per ({Var_1})")
        st.plotly_chart(fig , use_container_width=True,theme="streamlit") 
    with st.container(): # 'casual','registered','cnt'
        Var_1 = 'yr'
        msk = df_new.groupby(['hr',Var_1])[['registered']].mean().reset_index().sort_values(['hr'])
        fig = px.line(msk , x='hr' , y='registered',color=Var_1,title=f"Time Series (registered) Hourly per ({Var_1})")
        st.plotly_chart(fig , use_container_width=True,theme="streamlit") 
    with st.container(): # 'casual','registered','cnt'
        Var_1 = 'yr'
        msk = df_new.groupby(['hr',Var_1])[['cnt']].mean().reset_index().sort_values(['hr'])
        fig = px.line(msk , x='hr' , y='cnt',color=Var_1,title=f"Time Series (cnt) Hourly per ({Var_1})")
        st.plotly_chart(fig , use_container_width=True,theme="streamlit") 

    with st.container():
        st.success("""
        - ✔️ From the above graphs we can see that:
            * Overall: Casual is less than registered in the two years.
                * That can help us in (Operations Capacity, Cost, Budgeting, etc.) Planning to Considerate to increase about 40% to 60% as Casual rate from Register Daily/Monthly/Yearly Rate.
            * Seasons: Fall reach 31% of Avg Total Rents, then summer 27%, Then Winter 26%.
                * Than can help us during distribute the Manpower, Costs, and Purchasing Maintenance/Renewals OPS to be in the right time to avoid any disagreement with Customers requirements.
            * Months: Usually Dec & Jan are the low rents, and from May to Oct are the High rents, but from Feb to Apr + Nov are mean.
                * During the High rents Months: Our resources need to be ready to help customers during the season.
                * During the low rents Months: We can use these months to make the necessary maintenance and renovation of assets.
            * Time Period: 
                * High Period : Afternoon for Casual & Evening for Registered.
                * Low Period : Early Morning & Night for Both.
                    * That will help us to Distribute our resources & Assets & Products to help customers as per Rents count of High/Low periods.
            * Weather sit:
                * Customers usually rents during Clear/Mist Weather sit.
                    * That will let us should follow the weather update.
            * Week Day & Hourly EDA:
                * Casual has High Rents during Weekends (Usually be increasing from 7 AM till 10 AM & From 16 PM till 19 PM).
                * Registered has High Rents during Working Days (Usually be increasing from 10 AM till 20 PM).
                    * That’s mean: registered Customers may use the bikes during going to Work/School, and Casual may use the bikes for fun/sports.
                    * Also this will help us for the distribution type of bikes to help customers and keep satisfaction.""")
# ______________________________________________________________________________________________________________________________________________________________________________________________________
with Custom_EDA:
    with st.container():
        st.subheader('Custom multivariate EDA')
        X_lst = ['Select','yr','season','mnth','weekday','hr','time_preiod','weathersit']
        Y_lst = ['Select','casual','registered','cnt']
        Color_lst = ['Select','holiday','workingday','season','yr','mnth','weekday','time_preiod','weathersit']
        col_1 , col_2 , col_3 = st.columns([10,10,10])
        with col_1:
            x = st.selectbox ('Select X :' , X_lst)
        with col_2:
            y = st.selectbox ('Select Y :' , Y_lst)
        with col_3:
            color = st.selectbox ('Select Color :' , Color_lst)
        Method = st.radio('Calculation Method' , ['mean' , 'sum'], horizontal=True)
        if x == color or x == y or y == color or  x == 'Select' or y =='Select' or  color=='Select':
            st.write(":red[Please Select a value from every list x , y , color]")
        else:
            if Method == 'mean':
                MSK_AVG =df_new.groupby([x,color])[[y]].mean().sort_values(y).reset_index()
                FI_AVG = px.histogram(MSK_AVG , x=x , y=y , color=color,barmode='group',text_auto=True,title= f'{Method} of {x} vs {y}')
                st.plotly_chart(FI_AVG , use_container_width=True,theme="streamlit") 
            else:
                MSK_SUM = df_new.groupby([x,color])[[y]].sum().sort_values(y).reset_index()
                FI_SUM = px.histogram(MSK_SUM , x=x , y=y , color=color,barmode='group',text_auto=True,title= f'{Method} of {x} vs {y}')
                st.plotly_chart(FI_SUM , use_container_width=True,theme="streamlit") 
    with st.container():
        st.subheader('Custom Scatter')
        X_lst_Scatter = ['Select','yr','mnth','hr','holiday','weekday','workingday','temp','atemp','hum','windspeed','casual','registered','cnt']
        Y_lst_Scatter = ['Select','yr','mnth','hr','holiday','weekday','workingday','temp','atemp','hum','windspeed','casual','registered','cnt']
        Color_lst_Scatter = ['Select','yr','season','mnth','weekday','hr','time_preiod','weathersit']
        col_1 , col_2 , col_3 = st.columns([10,10,10])
        with col_1:
            x = st.selectbox ('Select_X_Scatter :' , X_lst_Scatter)
        with col_2:
            y = st.selectbox ('Select_Y_Scatter :' , Y_lst_Scatter)
        with col_3:
            color = st.selectbox ('Select_Color_Scatter :' , Color_lst_Scatter)
        if x == color or x == y or y == color or  x == 'Select' or y =='Select' or  color=='Select':
            st.write(":red[Please Select a value from every list x , y , color]")
        else:
            fig = px.scatter(df_new , x=x , y=y , color=color,title= f'{x} vs {y}')
            st.plotly_chart(fig , use_container_width=True,theme="streamlit")
            
    with st.container():
        st.subheader("Custom EDA ('Distribution')")
        X_Dist_lst = ['Select' ,'temp','atemp','hum','windspeed','casual','registered','cnt']
        X_Dist = st.selectbox ('Select Line X :' , X_Dist_lst)
        if X_Dist == 'Select' :
            st.write(":red[Please Select x]")
        else:
            Dist_Fig = px.histogram(df_new ,x=X_Dist ,marginal='box' , title=f"Distribution of {X_Dist}")
            st.plotly_chart(Dist_Fig , use_container_width=True,theme="streamlit") 
# ______________________________________________________________________________________________________________________________________________________________________________________________________

with Prediction:
    # Load data
    data = pickle.load(open('Deployment_data.pkl', 'rb'))
    
    # Input Data
    with st.container():
        P_col_0, P_col_1,P_col_2, P_col_3  = st.columns([2,10,10,10])
        with P_col_0:
            st.subheader("📈 Prediction")
        with P_col_2:
            min_date = pd.Timestamp("2011-01-01")
            max_date = pd.Timestamp("2100-12-31")
            selected_date = st.date_input('Date' , min_value=min_date , max_value=max_date , value=pd.Timestamp.today())

        col1, col2 , col3  = st.columns([10,10,10])
        with col1:
            Temp = st.slider("Temp" , min(data["Temp"]) , max(data["Temp"]))
            Atemp = st.slider("Atemp" , min(data["Atemp"]) , max(data["Atemp"]))
        with col2:
            Hour = st.slider("Hour" , 0 , 23)
            windspeed = st.slider("Windspeed" , min(data["Windspeed"]) , max(data["Windspeed"]))
        with col3:
            Workingday = st.radio("Workingday" , data["Workingday"])
            hum = st.slider("Hum" , min(data["Hum"]) , max(data["Hum"]))

            def get_season_name(month):
                if month in [12, 1, 2]:
                    return '04_winter'
                elif month in [3, 4, 5]:
                    return '01_springer'
                elif month in [6, 7, 8]:
                    return '02_summer'
                elif month in [9, 10, 11]:
                    return '03_fall'
                else:
                    return 'Invalid Month'
        with st.container():
            # New Data
            col1, col2, col3 = st.columns([20,5,10])
            with col1:
                N_data = pd.DataFrame({
                    "dteday":selected_date,"season":get_season_name(pd.Timestamp(selected_date).month),
                    "mnth":pd.Timestamp(selected_date).month,"weekday":pd.Timestamp(selected_date).dayofweek,
                    "hr":Hour,"workingday":Workingday,
                    "temp":Temp,"atemp":Atemp,
                    "hum":hum,"windspeed":windspeed}, index=[0])

                # Preprocess
                Preprocess = pickle.load(open('processor.pkl', 'rb'))
                N_test = Preprocess.transform(N_data)

                # # Predict
                Predict = pickle.load(open('model.pkl', 'rb'))
                result = Predict.predict(N_test)
                st.write(N_data)
            with col2:
                # # Output
                if st.button("Predict"):
                    st.header(f"result : {int(result)}")
                    st.balloons()
