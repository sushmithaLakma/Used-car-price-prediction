import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

header = st.beta_container()
with header:
	st.title('Used car price prediction')
	data = pd.read_csv('cardata.csv')

col1, col2 = st.beta_columns(2)

with col1:
	st.write('### Shape of the dataset',data.shape)
	st.text("")

with col2:		
	st.write('## "The problem is to predict the prices of old cars based on certain features."')	

if st.button("Sneak Peek of the dataset"):
		st.write(data.head())

if st.sidebar.checkbox("Show dataset with selected columns"):
    # get the list of columns
    columns = data.columns.tolist()
    st.write("#### Select the columns to display:")
    selected_cols = st.multiselect("", columns)
    if len(selected_cols) > 0:
        selected_df = data[selected_cols]
        st.dataframe(selected_df)

if st.sidebar.checkbox('More info on the distribution of data? (eg.Mean, Std. deviation, IQR)'):
    st.write(data.describe())

st.text("")
current_year = 2021
data['no_of_years'] = current_year - data['Year']
data.drop('Year',axis=1,inplace=True)

# Before Label encoding
#col3_1, col3_2 = st.beta_columns(2)
#with col3_1:
#	st.write(data['Fuel_Type'].unique())
#	st.write(data['Seller_Type'].unique())
#	st.write(data['Transmission'].unique())

# Treating skewness
data['Present_Price'] = np.log(data['Present_Price'])
data['Kms_Driven'] = np.log(data['Kms_Driven'])

le = LabelEncoder()

categorical_columns = [x for x in data.columns if data.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns if x not in ['Car_Name']]

numerical_columns = [x for x in data.columns if data.dtypes[x]!='object']

col2_1, col2_2 = st.beta_columns(2)
with col2_1:
	st.write('##### Independent categorical variables')
	st.write(categorical_columns)

with col2_2:	
    st.write('##### Independent numerical variables')
    st.write(numerical_columns)

data[categorical_columns]=data[categorical_columns].apply(le.fit_transform)
data.drop(['Car_Name'],axis=1,inplace=True)

# After label encoding
#with col3_2:	
#	st.write(data['Fuel_Type'].unique())
#	st.write(data['Seller_Type'].unique())
#	st.write(data['Transmission'].unique())

#st.write(data.head())
features = data.iloc[:,1:]
target = data.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.1,random_state=1)   
st.text('')

#linear_reg = LinearRegression();
#linear_reg.fit(X_train,y_train)

#coefficients = linear_reg.coef_
#st.bar_chart(coefficients)

#score = linear_reg.score(X_train,y_train)
#st.write('### Model score --  ',score)

#predictions = linear_reg.predict(X_test)
#r2_score = r2_score(y_test,predictions)
#st.write('R2 score --  ',r2_score)

st.write('### > Check the estimated price of a used car')
st.text("")
year = st.number_input('Year of purchase',2000,2020)
price = st.number_input('Price of the car when purchased in Lakhs',1,50)
kms = st.number_input('Kilometers driven',500,500000)
owners = st.number_input('Owners till now',0,3)
fuel = st.selectbox('Fuel type',options=['Petrol','Diesel','CNG'])
buyer = st.selectbox('Are you a dealer or an indiviual?',options=['Dealer','Indiviual'])
transmission = st.selectbox('Transmission type',options=['Manual','Automatic'])

# No. of years
years = current_year - year

# Price
price = np.log(price)

# Kms_driven
kms = np.log(kms)

# Fuel 
if(fuel == 'CNG'):
	fuel = 0
elif(fuel == 'Diesel'):
	fuel = 1
else:
	fuel = 2

# Buyer
if(buyer == 'Dealer'):
	buyer = 0
else:
    buyer = 1

# Transmission type
if(transmission == 'Automatic'):
	transmission = 0
else:
	transmission = 1	

rf = RandomForestRegressor(max_depth=7)

rf.fit(X_train,y_train)

score_rf = rf.score(X_train,y_train)
#st.write('### Model score --  ',score_rf)

load_model = pickle.load(open('car_prediction_model.pkl', 'rb'))

predictions = load_model.predict(X_test)

mse = mean_squared_error(y_test,predictions)
mae_score = mean_absolute_error(y_test,predictions)
r2 = r2_score(y_test,predictions)

st.text('')
predicted_price = load_model.predict([[price, kms, fuel, buyer, transmission, owners, years]])
if st.button('Get the estimated price'):
	st.write(round(predicted_price[0],2))

st.text('')
if st.sidebar.checkbox('Metrics'):
	st.write('Mean squared error --  ',mse)#round(mse,2))
	st.write('Mean absolute error --  ',mae_score)#round(mae_score,2))
	st.write('R2 score --  ',r2)#round(r2,2))	

# open a file, where you ant to store the data
#file = open('car_prediction_model.pkl', 'wb')

# dump information to that file
#pickle.dump(rf, file)
