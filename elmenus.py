import numpy as np
from prophet import Prophet
import pandas as pd
import plotly.express as px 
import streamlit as st
import sqlite3
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
tasks=st.selectbox('Tasks',['EDA','Churn Analysis','Customer Segmentation','Predictive Modeling','Visualization and Reporting'])

class Tasks:
    def __init__(self):
        self.database = "elmenus.db"
        self.databasegeolocation = "elmenus-geolocation.db"
        self.conn_geo = sqlite3.connect(self.databasegeolocation)
        self.conn = sqlite3.connect(self.database)
        self.orders_query = 'SELECT * FROM orders'
        self.order_df = pd.read_sql_query(self.orders_query, self.conn)
        self.order_df['order_purchase_timestamp'] = pd.to_datetime(self.order_df['order_purchase_timestamp'])
        self.order_df['order_purchase_timestamp'] = self.order_df['order_purchase_timestamp'].dt.strftime('%Y-%m-%d')

        self.popular_product_category_query = '''
            SELECT op.order_id, op.product_id, op.product_category_name, op.seller_id
            FROM (
                SELECT oi.*, p.product_category_name
                FROM order_items oi
                LEFT JOIN products p ON oi.product_id = p.product_id
            ) AS op
            LEFT JOIN orders o ON op.order_id = o.order_id
        '''
        self.popular_product_category_df = pd.read_sql_query(self.popular_product_category_query, self.conn)

        self.geolocation_query = 'SELECT * FROM geolocation'
        self.geolocation_df = pd.read_sql_query(self.geolocation_query, self.conn_geo)

        self.customers_query = 'SELECT * FROM Customers'
        self.customers_df = pd.read_sql_query(self.customers_query, self.conn)

        self.orders_order_items_query = '''
            SELECT oi.*, o.customer_id
            FROM order_items oi
            LEFT JOIN orders o ON oi.order_id = o.order_id
        '''
        self.orders_order_items_df = pd.read_sql_query(self.orders_order_items_query, self.conn)

        self.productname_location_query = '''
            SELECT c.customer_city, p.product_category_name
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            LEFT JOIN products p ON oi.product_id = p.product_id
        '''
        self.productname_location_df = pd.read_sql_query(self.productname_location_query, self.conn)

        self.orders_item_product_query = '''
            SELECT oi.*, p.product_category_name, o.order_purchase_timestamp
            FROM order_items oi
            LEFT JOIN products p ON oi.product_id = p.product_id
            LEFT JOIN orders o ON oi.order_id = o.order_id
        '''
        self.orders_item_product_df = pd.read_sql_query(self.orders_item_product_query, self.conn)
        self.orders_item_product_df=pd.read_sql_query(self.orders_item_product_query,self.conn)
        self.orders_item_product_df['order_purchase_timestamp']=pd.to_datetime(self.orders_item_product_df['order_purchase_timestamp'])
        self.orders_item_product_df['year']=self.orders_item_product_df['order_purchase_timestamp'].dt.year
        self.orders_item_product_df['order_purchase_timestamp']=self.orders_item_product_df['order_purchase_timestamp'].dt.strftime('%Y-%m-%d')


    def check_order(self):
        st.subheader('Explore the distribution of orders over time, analyzing trends in order volume and orderstatuses')
        self.order_status_options = ['All Status'] + list(self.order_df['order_status'].unique())
        self.order_status = st.selectbox('Select Order Status', self.order_status_options)
        if self.order_status!='All Status':
            self.order_df=self.order_df[self.order_df['order_status']==self.order_status]
        self.order_df_pivot=self.order_df.pivot_table(values='order_id',index='order_purchase_timestamp',aggfunc='count').reset_index()
        try:
            self.result = seasonal_decompose(self.order_df_pivot['order_id'], model='additive', period=30)
            tabs = ["Actual Data", "Trend", "Seasonal", "Residual"]
            selected_tab = st.radio("Select Chart Types", tabs)
            if selected_tab == "Actual Data":
                fig = px.line(x=self.order_df_pivot['order_purchase_timestamp'], y=self.result.observed, labels={'y': 'Total Orders', 'x': 'Date'},
                                title='Actual Data')
            elif selected_tab == "Trend":
                fig = px.line(x=self.order_df_pivot['order_purchase_timestamp'], y=self.result.trend, labels={'y': 'Total Orders', 'x': 'Date'},
                                title='Trend')
            elif selected_tab == "Seasonal":
                fig = px.line(x=self.order_df_pivot['order_purchase_timestamp'], y=self.result.seasonal, labels={'y':'Total Orders', 'x': 'Date'},
                                title='Seasonal')
            elif selected_tab == "Residual":
                fig = px.line(x=self.order_df_pivot['order_purchase_timestamp'], y=self.result.resid, labels={'y': 'Total Orders', 'x': 'Date'},
                                title='Residual')
            st.plotly_chart(fig)
            st.markdown("This Charts shows there is an increase in sales from year to year and that sales fluctuate seasonally.")
            
       
        except:
            st.dataframe(self.order_df)
        self.order_df_status=self.order_df.pivot_table(values='order_id',index=['order_status'],aggfunc='count').reset_index().rename(columns={'order_id':'Total Order'}).sort_values(by='Total Order',ascending=False)
        fig_status=px.histogram(self.order_df_status,x='order_status',y='Total Order',title='Frequaintly')
        st.plotly_chart(fig_status)
        return self.product_seller()
    
    def product_seller(self):
        st.subheader(' The most popular product categories and sellers on the platform')
        self.popualr_product_category_category_name_pivot=self.popular_product_category_df .pivot_table(values='order_id',index='product_category_name',aggfunc='count').reset_index().sort_values(by='order_id',ascending=False)
        self.popualr_product_category_category_name_pivot.rename(columns={'order_id':'Total Orders'},inplace=True)
        fig=px.histogram(self.popualr_product_category_category_name_pivot,x='product_category_name',y='Total Orders',title=' Popular Categories with Total Order')
        popualr_product_category_seller_pivot=self.popular_product_category_df.pivot_table(values='order_id',index='seller_id',aggfunc='count').reset_index().sort_values(by='order_id',ascending=False)
        popualr_product_category_seller_pivot.rename(columns={'order_id':'Total Orders'},inplace=True)
        st.plotly_chart(fig)
        fig=px.histogram(popualr_product_category_seller_pivot.head(30),x='seller_id',y='Total Orders',title='Top 20 Sellers ID')
        st.plotly_chart(fig)
        st.subheader('3-customer demographics, such as location and purchasing behavior')
    def customer_geolocation(self):
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
        st.subheader("Notes: values it means the prices that customers paid.")
        orders_order_items_pivot=self.orders_order_items_df.pivot_table(values=['price','product_id'],index=['customer_id'],aggfunc={'price':'sum','product_id':'count'}).reset_index().sort_values(by='price',ascending=False)
        q2 = np.quantile(orders_order_items_pivot['price'], 0.50)
        q3 = np.quantile(orders_order_items_pivot['price'], 0.75)
        bins = [0, q2, q3, float('inf')]
        labels = ['Low', 'Medium', 'High']
        st.write(f'- Low-values: The price between 0 and {q2}.')
        st.write(f'- Medium-values: The price between {q2} and {q3} price')
        st.write(f'- High-values: above {q3} price')
        orders_order_items_pivot['Segment Category'] = pd.cut(orders_order_items_pivot['price'], bins=bins, labels=labels, right=False)
        fig=px.histogram(orders_order_items_pivot,x='Segment Category',title="This chart illustrates the percentage of customers in each category.",histnorm='percent')
        st.plotly_chart(fig)
        #################
        #city with product name
        
        self.city=st.selectbox('Select City',['All City']+list(self.productname_location_df['customer_city'].unique()))
        if self.city!='All City':
            self.productname_location_df=self.productname_location_df[self.productname_location_df['customer_city']==self.city]
        self.productname_location__pivot=self.productname_location_df.pivot_table(values='customer_city',index='product_category_name',aggfunc='count').reset_index().rename(columns={'customer_city':'Total Order'}).sort_values(by='Total Order',ascending=False)        
        fig=px.histogram(self.productname_location__pivot,x='product_category_name',y='Total Order',title='divide customers based on their orders')
        fig.update_layout(width=800, height=600) 
        st.plotly_chart(fig)
    ###############################################################################################
    def predictive_modeling(self):
        st.subheader('Forecast sales')
        self.order_status_options = ['All Status'] + list(self.order_df['order_status'].unique())
        self.order_status = st.selectbox('Select Order Status', self.order_status_options)
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
            fig.add_scatter(x=predictions['ds'], y=predictions['trend'], mode='lines', name='Trend')
            fig.update_layout(width=800, height=600) 
            st.write('The Green line represents trend')
            st.write('The blue line represents forecasting')
            st.plotly_chart(fig)    
            st.dataframe(predictions.iloc[:,:5])
            st.write(f"The blue color represents the forecasted values for the next {period} days.")
            st.write("- This indicates that the company's orders were increasing from 2016 to 2017, then to 2018. However, in 2018, its orders started to decline.")
            mse = mean_squared_error(data['y'][-1* int(period):], predictions['yhat'][-1 * int(period):])
            st.write('MSE It is a metric used to measure the average squared difference between the estimated values and the actual values.')
            st.write(f"MSE: {mse}")
        else:
            st.dataframe(data)
        return self.predictive_customer_demand()
    def predictive_customer_demand(self):
        st.subheader('customer_demand based on product')
        product_category=['All Product']+list(self.orders_item_product_df['product_category_name'].unique())
        product_category=st.selectbox('Select Product name',product_category)
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
            fig.add_scatter(x=orders_item_product_pivot['ds'], y=orders_item_product_pivot['y'], mode='lines', name='Original Data')
            fig.update_layout(width=800, height=600) 
            fig.add_scatter(x=predictions['ds'], y=predictions['trend'], mode='lines', name='Trend')
            st.write('The Green line represents trend')
            st.write('The blue line represents forecasting')
            st.plotly_chart(fig)
            st.write(f"The blue color represents the forecasted values for the next {period} days.")
            mse = mean_squared_error(orders_item_product_pivot['y'][-1*int(period):], predictions['yhat'][-1*int(period):])
            st.write('MSE It is a metric used to measure the average squared difference between the estimated values and the actual values.')
            st.write(f"MSE: {mse}")
        else:
            st.dataframe(orders_item_product_pivot.rename(columns={'y':'Total Orders'}))
        
            
     
    def product_quailty(self):
        st.header('Evaluate the performance')
        st.write('')
        st.subheader("We create KPI's to show our performance.")
        st.write("We create KPI's to show our performance, focusing on the following key points:")
        st.write("- Number of products with increased sales assume 40%.")
        st.write("- delivery speed assume 30%")
        st.write("- Customer satisfaction rate assume 30%.")
        st.write('')
        st.write('')
        st.subheader('1-Product Status: notes we have')
        st.write("- 61.29% of the products increased their sales from 2017 to 2018.")
        st.write("- 38.71% of the products decreased their sales.")
        self.orders_item_product_quailty_pivot=self.orders_item_product_df.pivot_table(values='product_id',index='product_category_name',columns='year',aggfunc='count',fill_value=0).reset_index().rename(columns={'product_id':'Total Order'})
        self.orders_item_product_quailty_pivot['change_2017_2016'] = ((self.orders_item_product_quailty_pivot[2017] - self.orders_item_product_quailty_pivot[2016]) / self.orders_item_product_quailty_pivot[2016]) * 100
        self.orders_item_product_quailty_pivot['change_2018_2017'] = ((self.orders_item_product_quailty_pivot[2018] - self.orders_item_product_quailty_pivot[2017]) / self.orders_item_product_quailty_pivot[2017]) * 100
        self.orders_item_product_quailty_pivot=self.orders_item_product_quailty_pivot[self.orders_item_product_quailty_pivot[self.orders_item_product_quailty_pivot.columns[1]]!=0]
        labels = ['Huage  Decrease', 'Severe Decrease', 'Considerable Decrease', 'No Change', 'Normal Increase', 'Good Increase', 'Strong Increase', 'Significant Increase', 'Remarkable Increase']
        bins = [-float('inf'), -200, -100, -20, 0, 20, 50, 100, 200, float('inf')]
        self.orders_item_product_quailty_pivot['pcg'] = pd.cut(self.orders_item_product_quailty_pivot['change_2018_2017'], bins=bins, labels=labels, right=False)
        select_category=st.selectbox('Select Product Staus',list(self.orders_item_product_quailty_pivot['pcg'].unique()))
        if select_category!='Select':
            st.markdown('Notes : Change it means the percentage of change from one year to the next')
            self.orders_item_product_quailty_pivot_category=self.orders_item_product_quailty_pivot[self.orders_item_product_quailty_pivot['pcg'].str.contains(select_category)==True]
            fig=px.histogram(self.orders_item_product_quailty_pivot_category,x='product_category_name',y='change_2018_2017')
            st.plotly_chart(fig)
            st.dataframe(self.orders_item_product_quailty_pivot_category)
        #self.orders_item_product_quailty_pivot['pcg status']=np.where(self.orders_item_product_quailty_pivot['change_2018_2017']>0,1,0)
        #st.write(self.orders_item_product_quailty_pivot['pcg status'].value_counts(normalize=True))
        #st.write(self.orders_item_product_quailty_pivot['pcg status'].value_counts())
        return self.reviews_order()

    def reviews_order(self):
        #self.reviews_order_query='''select orders.*,order_reviews.review_score from orders left join order_reviews
        #                        on order_reviews.order_id=orders.order_id'''
        #self.reviews_order_df=pd.read_sql_query(self.reviews_order_query,self.conn)
        #self.reviews_order_df['order_delivered_customer_date']=pd.to_datetime(self.reviews_order_df['order_delivered_customer_date'])
        #self.reviews_order_df['order_estimated_delivery_date']=pd.to_datetime(self.reviews_order_df['order_estimated_delivery_date'])    
        #self.reviews_order_df['days diff'] = (self.reviews_order_df['order_delivered_customer_date'] - self.reviews_order_df['order_estimated_delivery_date']).dt.days
        #self.reviews_order_df['On Time Status']=np.where(self.reviews_order_df['days diff']>0,1,0)
        #st.write(self.reviews_order_df['On Time Status'].value_counts(normalize=True))
        st.subheader("2- Delivery Status")
        st.write('')
        st.write("- The percentage of orders delivered on time is 93.44%, and the percentage of delayed orders is 6.56%. This means that orders are almost never delayed.")
        labels = ['On Time', 'Delayed']
        values = [93.44, 6.56]
        fig = px.pie(names=labels, values=values, title='Order Delivery Status')
        st.plotly_chart(fig)
        return self.satisfaction()
    def satisfaction(self):
        st.subheader('3- NPS or Net Promoter Score')
        st.write('is a popular metric used to measure customer satisfaction and loyalty to a brand. It is typically calculated by subtracting the percentage of detractors (customers who express dissatisfaction with the product or service) from the percentage of promoters (customers who are encouraged to promote the product or service). This metric can be used to gauge loyalty and track changes in feedback over time.')
        #label=['detractor','passive','poromoter']
        #bin=[0,2,3,5]
        #self.reviews_order_df=self.reviews_order_df[~self.reviews_order_df['review_score'].isna()]
        #self.reviews_order_df['Customer Status']=pd.cut(self.reviews_order_df['review_score'],bins=bin,labels=label)
        #self.reviews_order_df_pivot=self.reviews_order_df.pivot_table(values='order_id',index='Customer Status',aggfunc='count')
        detractor = 14575
        passive = 8179
        promoter = 76470
        total = detractor + passive + promoter
        nps_score= (promoter - detractor) / total
        st.write("Our NPS Percentage is: {:.2%}".format(nps_score) + " This percentage is low, so we should find the reason.")
        labels = ['detractor', 'passive','Promoter']
        values = [detractor, passive,promoter]
        fig = px.pie(names=labels, values=values, title='NPS Score')
        st.plotly_chart(fig)
        st.write( 'Promoter (Satisfied) (4 or 5)',  'Passive (Not Bad, Not Good) (3)',  'Detractor (Dissatisfied) (1or 2)')        
        ######################KPI's
        st.subheader('Our performance score: ')
        st.write("- product status 24.516.")
        st.write("- Delivery Status 28.032")
        st.write("- NPS Score 18.714")
        st.write('')
        st.markdown("<h1 style='text-align: center; color: red;'>Our Score</h1>", unsafe_allow_html=True)
        st.write("<h2 style='text-align: center;'>71.26%</h2>", unsafe_allow_html=True)
    def Visualization(self):
        st.subheader("Notes :")
        st.write("- There was an increase in orders in 2018 compared to 2017 by 217, which is 19.75%. Additionally, there were significant increases in some products compared to 2017, such as fraldas_higiene, industria_comercio_e_negocios, alimentos, and other products.")
        st.write("- There were some products that experienced a significant decrease in their orders, such as [brinquedos,ferramentas_jardim,cool_stuff]")
        st.write("- The city with the highest Orders is Sao Paulo.")
        st.write("- The majority of orders were made using credit.")
        st.write("- The highest sales ID is 6560211a19b47992c3666cc44a7e94c0.")
        st.write("- The customer satisfaction rate is not good enough,The satisfaction rate is approximately 62%.")
        st.write('')
        st.write('')
        st.subheader("Recommendation :")
        st.markdown("""
            1. offering benefits to credit customers to increase their sales, such as interest-free installment plans and others.
            2. There are some products that have experienced a significant decrease in sales. Therefore, they must be studied carefully, as this could lead to damage to our reputation.
            3. Maintaining focus on the quality of products that have achieved high sales rates.
            4. Developing a marketing strategy for products in other areas to achieve high sales, such as Sao Paulo.
            5. The NPS rate is not good, which means that many customers are not satisfied. Therefore, it is important to speak with these customers to understand the main reasons for their dissatisfaction and improve them.
            6. Engage with the top sales achievers to understand their strategies and share them with others.
            """)





                    
                    
                    
        
        
        
        
        
        
        st.write('')
        url = 'https://app.powerbi.com/view?r=eyJrIjoiN2ExNDAzNjMtMTRiMS00OTg4LWFjNTgtOGE1YmM3MDA3YzJmIiwidCI6ImRmODY3OWNkLWE4MGUtNDVkOC05OWFjLWM4M2VkN2ZmOTVhMCJ9'
        st.markdown(url)

        
        

task= Tasks()
if tasks == 'EDA':
    task.check_order()
    if st.button('Click for Geolocation'):
        task.customer_geolocation()
if tasks=='Customer Segmentation':
    task.customer_segmentation()
if tasks=='Predictive Modeling':
    task.predictive_modeling()
    
if tasks=='Churn Analysis':
    task.product_quailty()
if tasks=='Visualization and Reporting':
    task.Visualization()