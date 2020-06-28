# Core Pkg
import streamlit as st
import pickle
from datetime import date

# Load EDA Pkgs
import pandas as pd 
import numpy as np

# Load Data Vis Pkg
import plotly.express as px
import pydeck as pdk

# function to load and cache(faster) the dataset and set mutation to True
@st.cache(allow_output_mutation=True)
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

# load model from pickle after ML
def load_prediction_models(model_file):
	loaded_model = pickle.load(open(model_file,"rb"))
	return loaded_model

# the main or header function
# visitors can select EDA, Interactive Charts, Prediction and About
def main():
    # Menu
    menu = ['EDA','Interactive Charts','Prediction','About']
    choices = st.sidebar.selectbox('Select Menu',menu)

    # load dataframe
    data = load_data('data/data.csv')
    # we established from EDA that year post 2013 is more stable and more reflective of 
    # current market trends
    data2013 = data.query('year_sold >=2013')

    # When 'EDA' is selected at the menu.
    if choices == 'EDA':

        st.title('EDA')

        st.header("Project Title : HDB Price Prediction ML App")
        st.subheader(" Problem Statement")
        st.markdown('How do flat buyers know if they have snatched a good deal?\
        Alternatively, how do flat sellers benchmark their property reasonably?\
        In order to help flat buyers and flat sellers make an informed decision, we decided to find out more about resale flat prices in Singapore.\
        Ultimately, we want to predict the price of resale flats in Singapore.')

        st.markdown('Complete notebook can be found [here](https://nbviewer.jupyter.org/github/andrewng88/hdb/blob/master/2_Exploratory_Data_Analysis.ipynb)')

        st.subheader("The Data")
        st.markdown('Obtained from [Data.gov.sg](http://data.gov.sg/dataset/resale-flat-prices)\
        the dataset is from **1990 to 2019**.')

        if st.checkbox("Show Summary of the Dataset"):
            st.write(data.describe())

        # display overall hdb price trend chart 
        table1=data.groupby("year_sold")["resale_price"].agg(["median"]).reset_index()
        table1.rename(columns={"median": "resale_price"},inplace=True)
        resale_price=px.line(table1,x="year_sold",y="resale_price")
        resale_price.update_layout(title_text='HDB Resale Price trend (1990 - 2019)',template='ggplot2')
        st.plotly_chart(resale_price)

        # chart commentary
        st.markdown('The decline in resale price and sudden surge in the number of units sold following 1997 is due to the 1997\
        [Asian financial crisis](https://www.todayonline.com/singapore/divergent-hdb-resale-private-home-price-trends-will-not-last).\
        With regards to the sharp spike in 2007 is because HDB has stopped Walk-In-Selection and replace it with Sale of Balance Flats\
        which is only twice per year and hence everyone went with the Resale')

        # display overall hdb transactions trend chart 
        table2=data.groupby("year_sold")["resale_price"].count().reset_index()
        table2=table2.rename(columns={"resale_price":"number_of_resale"})
        resale_transaction=px.line(table2,x="year_sold",y="number_of_resale")
        resale_transaction.update_layout(title_text='HDB Resale Transactions between (1990 - 2019)',template='ggplot2')
        st.plotly_chart(resale_transaction)
        
        # chart commentary
        st.markdown('Implementation of the revised [cooling measures](https://www.srx.com.sg/cooling-measures) to cool the residential market from 2010 onwards\
        led to the drop in resale price and low number of units sold during this period.Specifically the lowering of LTV(Loan-To-Value) from \
        90% to 80% - meaning buyers have to pay more initally.')

        # display overall dollar per square meter based on flat type
        data['dollar_per_sq_m'] = data['resale_price']/data['floor_area_sqm']
        table3 = data.groupby(["year_sold",'flat_type'])["dollar_per_sq_m",].agg(["median"]).reset_index()
        table3.rename(columns={"median": "dollar_per_sq_meter"},inplace=True)
        dollar_per_sq_m = px.line(table3,x="year_sold",y="dollar_per_sq_m",color = 'flat_type')
        dollar_per_sq_m.update_layout(title_text='Median Dollar Per Square Meter between 1990 and 2019 based on flat type',template='ggplot2')
        st.plotly_chart(dollar_per_sq_m)

        # chart commentary
        st.markdown('Similar trend if we break down based on flat type, the median went up by two fold from 2007 to 2013 and gradually\
        went down because of additional cooling measures')

        # display overall dollar per square meter based on storey
        table4 = data.groupby(["year_sold",'storey_range'])["dollar_per_sq_m",].agg(["median"]).reset_index()
        table4.rename(columns={"median": "dollar_per_sq_meter"},inplace=True)
        median_storey = px.line(table4,x="year_sold",y="dollar_per_sq_m",color = 'storey_range')
        median_storey.update_layout(title_text='Median Dollar Per Square Meter between 1990 and 2019 based on storey',template='ggplot2')
        st.plotly_chart(median_storey)

        st.markdown('Similar trend if we break down based on storey, but for high storey more than 40, price is still climbing.\
        We can also notice that high rise flats ( > 30 storeys ) starts from around 2005 onwards( less 3 years)')

        st.markdown('**We decided to work with data from 2013**. This is because the 1997 Asian financial crisis is a once off event and does not provide an \
        accurate reflection of the current situation.In addition, with the cooling measures still in place, using data from 2013\
        will ensure consistency in this aspect.')

        st.subheader('Complete notebook can be found [here](https://nbviewer.jupyter.org/github/andrewng88/hdb/blob/master/2_Exploratory_Data_Analysis.ipynb)')

    # When 'Interactive Charts' is selected at the menu.

    if choices == 'Interactive Charts':
        st.title('Interactive Charts')

        # 3D map component
        st.subheader("HDB Transactions Visualized using 3D")
        # from 1990 to 2019, defaults to 2019
        year = st.slider('Year to look at',1990,2019,2019) 
        data = data[data['year_sold'] == year]

        st.markdown("HDB transactions in **%i**" % (year))
        midpoint = (np.average(data["latitude"]), np.average(data["longitude"]))
        st.write(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            #display the mid of SG
            initial_view_state={
                "latitude": midpoint[0],
                "longitude": midpoint[1],
                "zoom": 11,
                "pitch": 50,
            },
            #displays the GPS of each HDB based on year_sold, GPS
            layers=[
                pdk.Layer(
                "HexagonLayer",
                data=data[['year_sold', 'latitude', 'longitude']],
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                radius=100,
                extruded=True,
                pickable=True,
                elevation_scale=4,
                elevation_range=[0, 1000]
                ),
            ],
        ))   

        #displays the Median price by Flat type

        st.subheader('View HDB Median price by Flat type')
        flat_type_values = sorted(list(data['flat_type'].unique()))
        flat_type_values.insert(0,'ALL')
        flat_option = st.selectbox("Flat_type", flat_type_values)
        sort_option = st.radio("Sort by", ("Ascending", "Descending"))
        flat_type_display_text = f'<sup>You selected {flat_option} and {sort_option}</sup>'
        st.markdown(flat_type_display_text,unsafe_allow_html=True)

        sort_option_dict = {'Ascending': False ,'Descending': True, }
        
        if flat_option=='ALL':
            revenue_all= data.groupby(['town'])['resale_price'].median().reset_index().sort_values(by=['resale_price'],ascending=sort_option_dict[sort_option])
            figure_to_plot = revenue_all
        else:
            revenue = data[data['flat_type'] == flat_option]    
            revenue= revenue.groupby(['town'])['resale_price'].median().reset_index().sort_values(by=['resale_price'],ascending=sort_option_dict[sort_option])
            figure_to_plot  = revenue
        fig_median=px.bar(figure_to_plot,x='resale_price',y='town',orientation="h",height=600,template='ggplot2')
        fig_median_title = f'HDB Median price for {flat_option} flats in {sort_option} order'
        fig_median.update_layout(title_text=fig_median_title)
        st.plotly_chart(fig_median)

        #displays the Median price by MRT

        st.subheader('View HDB Median price by MRT')
        mrt_values = sorted(list(data2013['mrt'].unique()))
        mrt_values.insert(0,'ALL')
        mrt_option = st.selectbox("MRT", mrt_values)
        mrt_display_text = f'<sup>You selected {mrt_option}</sup>'
        st.markdown(mrt_display_text,unsafe_allow_html=True)
        
        if mrt_option=='ALL':
            mrt_all= data2013.query('nearest_mrt_distance <1').groupby(['mrt'])['resale_price'].median().reset_index().sort_values(by=['resale_price'])
            fig_median=px.bar(mrt_all,x='resale_price',y='mrt',orientation='h',height=600,template='ggplot2')
            st.write(mrt_all)
        else:
            mrt = data2013[data2013['mrt'] == mrt_option]    
            mrt= mrt.query('nearest_mrt_distance <1').groupby(['mrt' ,'flat_type'])['resale_price'].median().reset_index().sort_values(by=['resale_price']).drop('mrt',axis=1)
            fig_median=px.bar(mrt,x='flat_type',y='resale_price',height=400,template='ggplot2')
            fig_median_title = f'HDB Median price for HDB flats near {mrt_option}'
            fig_median.update_layout(title_text=fig_median_title)
        st.plotly_chart(fig_median)
    
    # When 'Prediction' is selected at the menu.

    if choices == 'Prediction':
        st.subheader('Predictions')
        
        # load the unique database for speed
        df_unique_deploy = load_data('data/df_unique_deploy.csv')

        #obtain Postcode input from end user
        input_postcode = st.text_input("Postcode : ",560216) #560216
        postcode_list=df_unique_deploy['postcode'].unique().tolist()

        # we proceed with HDB transaction prediction, if the postcode is in the list
        if int(input_postcode) in postcode_list:
        
            input_postcode_results = f"Postcode is **{input_postcode}** "

            #auto retrieve the flat_type for selection based on postcode
            flat_type=df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['flat_type'].unique().tolist()
            flat_type = st.selectbox("The flat_type", (flat_type))
            flat_type_results = f"Flat Type is **{flat_type}**."

            #auto retrieve the flat_model for selection based on postcode
            f_model=df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['flat_model'].unique().tolist()
            flat_model = st.selectbox("The flat_model", (f_model))
            flat_model_results = f"Flat Model is **{flat_model}**."

            #auto retrieve town for selection based on postcode
            town = df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['town'].unique()[0]
            town_results = f" and it is located in **{town }** town ."

            #storey requires input from end user as we're not mind reader :P
            storey = st.slider("Storey level : ", 1 , 50,6) #8
            storey_results = f"Storey is **{storey}**."
            
            #auto retrieve floor_area_sqm for selection based on postcode
            area=df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['floor_area_sqm'].unique().tolist()
            floor_area_sqm = st.selectbox("Floor_area_sqm", (area))
            area_results = f"Area is **{floor_area_sqm }**."

            # calculate remaining lease = start year + 99 - current year
            today = date.today()
            year_sold = today.year
            month_sold = today.month
            lease_commence_date = df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['lease_commence_date'].tolist()[0]
            remaining_lease = int(lease_commence_date) + 99 - year_sold
            remaining_lease_results = f"Remaining lease is **{remaining_lease}** years ."

            #auto retrieve nearest_mrt_distance for selection based on postcode
            nearest_mrt_distance=df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['nearest_mrt_distance'].unique().tolist()[0]
            nearest_mrt_distance_results = f"MRT is **{nearest_mrt_distance:.2f}** km away."

            #auto retrieve CBD_distance for selection based on postcode
            CBD_distance=df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['CBD_distance'].unique().tolist()[0]
            cbd_distance_results = f"CBD is **{CBD_distance:.2f}** km away."

            #auto retrieve nearest_mall_distance for selection based on postcode
            nearest_mall_distance=df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['nearest_mall_distance'].unique().tolist()[0]
            nearest_mall_distance_results = f"Nearest Mall is **{nearest_mall_distance:.2f}** km away."

            #auto retrieve nearest_school_distance
            nearest_school_distance=df_unique_deploy[df_unique_deploy['postcode']==int(input_postcode)]['nearest_school_distance'].unique().tolist()[0]
            nearest_school_distance_results = f"Nearest school is **{nearest_school_distance:.2f}** km away."
            
            #condolidate all data for prediction
            sample_data= [[floor_area_sqm, year_sold, month_sold, remaining_lease,
            nearest_mrt_distance, CBD_distance, nearest_mall_distance,
            nearest_school_distance, storey, town, flat_type, flat_model]]

            list_columns = ['floor_area_sqm', 'year_sold', 'month_sold', 'remaining_lease',
            'nearest_mrt_distance', 'CBD_distance', 'nearest_mall_distance',
            'nearest_school_distance', 'storey', 'town', 'flat_type', 'flat_model']

            sample_data = pd.DataFrame(sample_data, columns = list_columns) 

            #load model and predict
            predictor = load_prediction_models('data/ridge.sav')
            predictor.predict(sample_data)

            #display data input
            if st.checkbox('Verbose ON/OFF:'):
                st.markdown(input_postcode_results + town_results)
                st.markdown(flat_type_results)
                st.markdown(flat_model_results)
                st.markdown(storey_results)
                st.markdown(area_results)
                st.markdown(remaining_lease_results)
                st.markdown(nearest_mrt_distance_results)
                st.markdown(cbd_distance_results)
                st.markdown(nearest_mall_distance_results)
                st.markdown(nearest_school_distance_results)
                st.write('Data collated for prediction:')
                st.write(sample_data)       
            
            #prefix $ and convert prediction to int 
            prediction = "{} {}".format('$', int(predictor.predict(sample_data)))
            st.subheader('HDB valuation:')
            st.success(prediction)

            #display other HDB data from the same block
            st.subheader("Other transactions from 2013 onwards(sorted by latest transaction)")
            st.dataframe(data2013[data2013['postcode']==int(input_postcode)].sort_values(by='month', ascending=False)\
            [['resale_price','dollar_per_sq_m','month','flat_type','flat_model','storey_range','lease_commence_date','floor_area_sqm']])

        #message to display if Postcode does not exists
        else:
            st.warning('Please input valid Postcode')

    if choices == 'About':
         st.header('About')
         
         st.subheader('Project by:')
         st.markdown('**Andrew Ng** andrew77@gmail.com')
         st.markdown('https://www.linkedin.com/in/sc-ng-andrew/')
         st.markdown('**Lau Lee Ling** lauleeling13@gmail.com')
         st.markdown('https://www.linkedin.com/in/lauleeling/')
    
if __name__=='__main__':
    main()