import numpy as np
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import plotly.express as px 
import streamlit as st
import sqlite3
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
tasks=st.selectbox('Tasks',['EDA','Customer Segmentation','Churn Analysis','Predictive Modeling','Visualization and Reporting'])
class Tasks:
    def __init__(self) :
        self.database="elmenus.db"
        self.databasegeolocation="elmenus-geolocation.db"
        self.conn_geo=sqlite3.connect(self.databasegeolocation)
        self.conn=sqlite3.connect(self.database)
        self.orders_query = 'select * from orders '
        self.order_df=pd.read_sql_query(self.orders_query,self.conn)
        self.order_df['order_purchase_timestamp']=pd.to_datetime(self.order_df['order_purchase_timestamp'])
        self.order_df['order_purchase_timestamp']=self.order_df['order_purchase_timestamp'].dt.strftime('%Y-%m-%d')
        self.popualr_product_category_query='''select order_products.order_id,order_products.product_id,order_products.product_category_name,order_products.seller_id from (
                                                    select order_items.*,products.product_category_name from order_items left join products
                                                    on order_items.product_id=products.product_id) as order_products
                                                    left join orders
                                                    on orders.order_id=order_products.order_id
                                                    '''
        self.popualr_product_category_df=pd.read_sql_query( self.popualr_product_category_query,self.conn)
        self.geolocation_query='select * from geolocation'
        self.geolocation_df=pd.read_sql_query(self.geolocation_query,self.conn_geo)
        self.customers_query='select * from Customers'
        self.customers_df=pd.read_sql_query(self.customers_query,self.conn)
        self.orders_order_items_query='''select order_items.*,orders.customer_id from order_items left join orders on order_items.order_id=orders.order_id'''
        self.orders_order_items_df=pd.read_sql_query(self.orders_order_items_query,self.conn)
        self.productname_location_query='''select customer_order_product.customer_city,products.product_category_name from (
                            select customer_city,product_id from (
                            select customers.customer_city,orders.order_id from customers left join orders on customers.customer_id=orders.customer_id) as customer_orders
                            inner join order_items
                            on order_items.order_id=customer_orders.order_id)as customer_order_product inner join products
                            on customer_order_product.product_id=products.product_id
                            '''
        self.productname_location_df=pd.read_sql_query(self.productname_location_query,self.conn)
        self.orders_item_product_query='''select order_items_products.*,orders.order_purchase_timestamp from 
                        (select order_items.*,products.product_category_name from order_items left join products on order_items.product_id=products.product_id) as order_items_products
                        left join orders
                        on order_items_products.order_id=orders.order_id'''
        self.orders_item_product_df=pd.read_sql_query(self.orders_item_product_query,self.conn)
    def check_order(self):
        st.subheader('Explore the distribution of orders over time, analyzing trends in order volume and orderstatuses')
        self.order_status_options = ['All Status'] + list(self.order_df['order_status'].unique())
        self.order_status = st.selectbox('', self.order_status_options)
        if self.order_status!='All Status':
            self.order_df=self.order_df[self.order_df['order_status']==self.order_status]
        self.order_df_pivot=self.order_df.pivot_table(values='order_id',index='order_purchase_timestamp',aggfunc='count').reset_index()
        try:
            self.result = seasonal_decompose(self.order_df_pivot['order_id'], model='additive', period=30)
            tabs = ["Actual Data", "Trend", "Seasonal", "Residual"]
            selected_tab = st.radio("Select Chart Types", tabs)
            if selected_tab == "Actual Data":
                fig = px.line(x=self.order_df_pivot['order_purchase_timestamp'], y=self.result.observed, labels={'y': 'Number of Orders', 'x': 'Date'},
                                title='Actual Data')
            elif selected_tab == "Trend":
                fig = px.line(x=self.order_df_pivot['order_purchase_timestamp'], y=self.result.trend, labels={'y': 'Number of Orders', 'x': 'Date'},
                                title='Trend')
            elif selected_tab == "Seasonal":
                fig = px.line(x=self.order_df_pivot['order_purchase_timestamp'], y=self.result.seasonal, labels={'y': 'Number of Orders', 'x': 'Date'},
                                title='Seasonal')
            elif selected_tab == "Residual":
                fig = px.line(x=self.order_df_pivot['order_purchase_timestamp'], y=self.result.resid, labels={'y': 'Number of Orders', 'x': 'Date'},
                                title='Residual')
            st.plotly_chart(fig)
            st.markdown('1-about Trend graphs the most of status shows that there was a very large increase from the beginning of 2017 to the end, but a large fluctuation occurred at the end of 2017')
            st.markdown('2-about Seasonal graphs shows seasonlaity')
            
       
        except:
            st.dataframe(self.order_df)
        self.order_df_status=self.order_df.pivot_table(values='order_id',index=['order_status'],aggfunc='count').reset_index().rename(columns={'order_id':'Total Order'}).sort_values(by='Total Order',ascending=False)
        fig_status=px.histogram(self.order_df_status,x='order_status',y='Total Order',title='Frequaintly')
        st.plotly_chart(fig_status)
        return self.product_seller()
    
    def product_seller(self):
        st.subheader(' The most popular product categories and sellers on the platform')
        self.popualr_product_category_category_name_pivot=self.popualr_product_category_df.pivot_table(values='order_id',index='product_category_name',aggfunc='count').reset_index().sort_values(by='order_id',ascending=False)
        self.popualr_product_category_category_name_pivot.rename(columns={'order_id':'Total Orders'},inplace=True)
        fig=px.histogram(self.popualr_product_category_category_name_pivot,x='product_category_name',y='Total Orders',title=' Popular Categories with Total Order')
        popualr_product_category_seller_pivot=self.popualr_product_category_df.pivot_table(values='order_id',index='seller_id',aggfunc='count').reset_index().sort_values(by='order_id',ascending=False)
        popualr_product_category_seller_pivot.rename(columns={'order_id':'Total Orders'},inplace=True)
        st.plotly_chart(fig)
        fig=px.histogram(popualr_product_category_seller_pivot.head(30),x='seller_id',y='Total Orders',title='Top 20 Sellers ID')
        st.plotly_chart(fig)
        return self.customer_geolocation()

    def customer_geolocation(self):
        st.subheader('3-customer demographics, such as location and purchasing behavior')
        customers_orders_query='''select orders.*,customers.customer_zip_code_prefix,customers.customer_city  from orders left join customers
                                on orders.customer_id=customers.customer_id'''
        self.customers_orders_df=pd.read_sql_query(customers_orders_query,self.conn)
        self.customers_orders_df['order_purchase_timestamp'] = pd.to_datetime(self.customers_orders_df['order_purchase_timestamp'])
        self.geolocation_df.rename(columns={'geolocation_zip_code_prefix':'customer_zip_code_prefix'},inplace=True)
        st.map(self.geolocation_df,latitude='geolocation_lat',longitude='geolocation_lng')
        return self.city()
    def city(self):
        customers_orders_pivot=self.customers_orders_df.pivot_table(values='order_id',index='customer_city',aggfunc='count').reset_index().rename(columns={'order_id':'Total Order'}).sort_values(by='Total Order',ascending=False).head(20)
        fig=px.histogram(customers_orders_pivot,x='customer_city',y='Total Order',title='Top 20 City Making Orders')
        st.plotly_chart(fig)
        st.write('Obviously, São Paulo and Rio de Janeiro are the two most in demand city')
    def customer_segmentation(self):
        orders_order_items_pivot=self.orders_order_items_df.pivot_table(values=['price','product_id'],index=['customer_id'],aggfunc={'price':'sum','product_id':'count'}).reset_index().sort_values(by='price',ascending=False)
        q2 = np.quantile(orders_order_items_pivot['price'], 0.50)
        q3 = np.quantile(orders_order_items_pivot['price'], 0.75)
        bins = [0, q2, q3, float('inf')]
        labels = ['Low', 'Medium', 'High']
        st.write(f'Low-values: between 0 and {q2} price')
        st.write(f'Medium-values: between {q2} and {q3} price')
        st.write(f'High-values: above {q3} price')
        orders_order_items_pivot['Segment Category'] = pd.cut(orders_order_items_pivot['price'], bins=bins, labels=labels, right=False)
        fig=px.histogram(orders_order_items_pivot,x='Segment Category',title=' divide customers based on values',histnorm='percent')
        st.plotly_chart(fig)
        #################
        #city with product name
        
        self.city=st.selectbox('Select City',['All City']+list(self.productname_location_df['customer_city'].unique()))
        if self.city!='All City':
            self.productname_location_df=self.productname_location_df[self.productname_location_df['customer_city']==self.city]
        self.productname_location__pivot=self.productname_location_df.pivot_table(values='customer_city',index='product_category_name',aggfunc='count').reset_index().rename(columns={'customer_city':'Total Order'}).sort_values(by='Total Order',ascending=False)        
        fig=px.histogram(self.productname_location__pivot,x='product_category_name',y='Total Order',title=' divide customers based on values')
        fig.update_layout(width=800, height=600) 
        st.plotly_chart(fig)
    ###############################################################################################
    def predictive_modeling(self):
        st.subheader('Forecast sales')
        self.order_status_options = ['All Status'] + list(self.order_df['order_status'].unique())
        self.order_status = st.selectbox('', self.order_status_options)
        if self.order_status!='All Status':
            self.order_df=self.order_df[self.order_df['order_status']==self.order_status]
        data=self.order_df.pivot_table(values='order_id',index=['order_purchase_timestamp'],aggfunc='count').reset_index()
        if len(data)>50:
            data.rename(columns={'order_purchase_timestamp':'ds','order_id':'y'},inplace=True)
            model=Prophet()
            model.fit(data)
            if len(data)>700:
                period=365
                forecasts = model.make_future_dataframe(periods=period)
            else:
                period=100
                forecasts = model.make_future_dataframe(periods=period)
            predictions = model.predict(forecasts)
            predictions['yhat'] = predictions['yhat'].apply(lambda x: max(x, 0))
            predictions['yhat_lower'] = predictions['yhat_lower'].apply(lambda x: max(x, 0))
            predictions['yhat_upper'] = predictions['yhat_upper'].apply(lambda x: max(x, 0))
            predictions['trend']=predictions['trend'].apply(lambda x:max(x,0))
            fig = px.line(predictions, x='ds', y='yhat', title='Sales Prediction')
            fig.add_scatter(x=predictions['ds'], y=predictions['yhat_lower'], mode='lines', name='Lower Bound')
            fig.add_scatter(x=predictions['ds'], y=predictions['yhat_upper'], mode='lines', name='Upper Bound')
            fig.add_scatter(x=data['ds'], y=data['y'], mode='lines+markers', name='Original Data')
            fig.update_layout(width=800, height=600) 
            st.plotly_chart(fig)    
            st.dataframe(predictions.iloc[:,:5])
            mse = mean_squared_error(data['y'][-1* int(period):], predictions['yhat'][-1 * int(period):])
            st.write('MSE It is a metric used to measure the average squared difference between the estimated values and the actual values.')
            st.write(f"MSE: {mse}")
        else:
            st.dataframe(data)
        return self.predictive_customer_demand()
    def predictive_customer_demand(self):
        self.orders_item_product_df=pd.read_sql_query(self.orders_item_product_query,self.conn)
        product_category=['All Product']+list(self.orders_item_product_df['product_category_name'].unique())
        product_category=st.selectbox('',product_category)
        self.orders_item_product_df['order_purchase_timestamp']=pd.to_datetime(self.orders_item_product_df['order_purchase_timestamp'])
        self.orders_item_product_df['order_purchase_timestamp']=self.orders_item_product_df['order_purchase_timestamp'].dt.strftime('%Y-%m-%d')
        if product_category!='All Product':
            self.orders_item_product_df=self.orders_item_product_df[self.orders_item_product_df['product_category_name']==product_category]
        orders_item_product_pivot=self.orders_item_product_df.pivot_table(values='order_id',index='order_purchase_timestamp',aggfunc='count').reset_index().rename(columns={'order_id':'y','order_purchase_timestamp':'ds'})
        if len(orders_item_product_pivot)>50:
            model=Prophet()
            model.fit(orders_item_product_pivot)
            if len(orders_item_product_pivot)>800:
                period=365
                forecasts = model.make_future_dataframe(periods=period)
            period=30
            forecasts = model.make_future_dataframe(periods=period)
            predictions = model.predict(forecasts)
            predictions['yhat'] = predictions['yhat'].apply(lambda x: max(x, 0))
            predictions['yhat_lower'] = predictions['yhat_lower'].apply(lambda x: max(x, 0))
            predictions['yhat_upper'] = predictions['yhat_upper'].apply(lambda x: max(x, 0))
            predictions['trend']=predictions['trend'].apply(lambda x:max(x,0))
            st.subheader('Predictive Customer demand')
            fig = px.line(predictions, x='ds', y='yhat', title='Predictive Customer demand')
            fig.add_scatter(x=predictions['ds'], y=predictions['yhat_lower'], mode='lines', name='Lower Bound')
            fig.add_scatter(x=predictions['ds'], y=predictions['yhat_upper'], mode='lines', name='Upper Bound')
            fig.add_scatter(x=orders_item_product_pivot['ds'], y=orders_item_product_pivot['y'], mode='lines+markers', name='Original Data')
            fig.update_layout(width=800, height=600) 
            st.plotly_chart(fig)
            st.dataframe(predictions)
            mse = mean_squared_error(orders_item_product_pivot['y'][-1*int(period):], predictions['yhat'][-1*int(period):])
            st.write('MSE It is a metric used to measure the average squared difference between the estimated values and the actual values.')
            st.write(f"MSE: {mse}")
        else:
            st.dataframe(orders_item_product_pivot)
        self.conn.close()
        self.conn_geo.close()
            
     
     

task = Tasks()
if tasks == 'EDA':
    task.check_order()
if tasks=='Customer Segmentation':
    task.customer_segmentation()
if tasks=='Predictive Modeling':
    task.predictive_modeling()
    