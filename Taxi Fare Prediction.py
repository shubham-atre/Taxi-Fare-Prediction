
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib as matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor


# The NYC Taxi and Limousine commission describes the fare calculation as below.
# <br>(1)The initial charge is $2.50.
# <br>(2)Plus 50 cents per 1/5 mile or 50 cents per 60 seconds in slow traffic or when the vehicle is stopped.
# I will use this information to perform some data cleaning and other analysis activity later.
# <br><br>
# Fistly, I load the data set and specify the column which needs to be parsed as date.

# In[2]:


df_raw =  pd.read_csv('C:/Users/Shubham/Desktop/DSF/all/train.csv', nrows=3000000, parse_dates=["pickup_datetime"])
df_raw.dtypes


# **Data Cleaning (1)** I Check if any row has null value for any of the columns.

# In[3]:


df_raw.isnull().sum()


# I will drop these incomplete records with null values.

# In[4]:


print("Size before: %d" %len(df_raw))
df_raw = df_raw.dropna()
print("Size after: %d" %len(df_raw))


# **Data Cleaning (2)** I Check if a row has the value zero for any of the columns.

# In[5]:


df_raw[(df_raw['fare_amount'] == 0) | 
       (df_raw['pickup_datetime'] == 0) | (df_raw['pickup_longitude'] == 0) | 
       (df_raw['pickup_latitude'] == 0) | (df_raw['dropoff_longitude'] == 0) | 
       (df_raw['dropoff_latitude'] == 0) | (df_raw['passenger_count'] == 0)].count()


# Several rows in the dataset have the value zero in atleast one columns.
# I will drop these incomplete records.

# In[6]:


print("Size before: %d" %len(df_raw))
df_raw = df_raw[(df_raw['fare_amount'] != 0) & 
       (df_raw['pickup_datetime'] != 0) & (df_raw['pickup_longitude'] != 0) & 
       (df_raw['pickup_latitude'] != 0) & (df_raw['dropoff_longitude'] != 0) & 
       (df_raw['dropoff_latitude'] != 0) & (df_raw['passenger_count'] != 0)]
print("Size after: %d" %len(df_raw))


# Let us describe the data set and see if some more outliers can be noticed.

# In[7]:


df_raw.describe()


# **Data Cleaning:**
#     <br><br>(3) The fare_amount has some negative values, these must be incorrect.  drop them.
#     <br>(4) Some coordinates are too much away from the coordinate limit of New York City.
#     <br><br>I will define reasonable limits for Longitudes and Latitudes, removing all outliers beyond the limit.

# In[8]:


print("Size before: %d" %len(df_raw))
df_raw = df_raw[(df_raw['fare_amount'] > 0)]
print("Size after: %d" %len(df_raw))


# In order to remove outliers based on coordinates, I check if it is possible to define a limit based on our test data.

# In[9]:


df_test =  pd.read_csv('C:/Users/Shubham/Desktop/DSF/all/test.csv', parse_dates=["pickup_datetime"])
df_test.dtypes


# In[10]:


df_test.describe()


# The longititude and latitude column seem to be reasonably within the range of New York City.
# I will define a limit for coordinates of our training data based on above tabel
# and remove any records that do not confirm with this range.

# In[11]:


#Range of coordinates
minimum_longitude = -74.263242
maximum_longitude = -72.986532
minimum_latitude = 40.568973
maximum_latitude = 41.709555

print("Size before: %d" %len(df_raw))
df_raw = df_raw[(df_raw.pickup_longitude >= minimum_longitude) & (df_raw.pickup_longitude <= maximum_longitude) &
           (df_raw.pickup_latitude >= minimum_latitude) & (df_raw.pickup_latitude <= maximum_latitude) &
           (df_raw.dropoff_longitude >= minimum_longitude) & (df_raw.dropoff_longitude <= maximum_longitude) &
           (df_raw.dropoff_latitude >= minimum_latitude) & (df_raw.dropoff_latitude <= maximum_latitude)]
print("Size after: %d" %len(df_raw))


# **Data Cleaning :** 
# (5) As per the official site of New York Taxi and Limousine Commision,
# the maximum number of passengers allowed in a cab is 5, with an exception to allow one extra child passenger.
# (Source-http://www.nyc.gov/html/tlc/html/faq/faq_pass.shtml)
#     I am going to drop any rides which involved more than 6 passengers.

# In[12]:


print("Size before: %d" %len(df_raw))
df_raw = df_raw[(df_raw.passenger_count <= 6)]
print("Size after: %d" %len(df_raw))


# In[13]:


#In order to calculate euclidean distance between our coordinates, I am going to define a function
#This Function will take the GPS coordinates, convert them to UTM coordinates using PyPi library
#The UTM coordinates will be used to calculate euclidean distance using SciPy library.
#I have referenced the following threads for this part
#https://gis.stackexchange.com/questions/58530/find-euclidean-distance-in-gps
#https://math.stackexchange.com/questions/738529/distance-between-two-points-in-utm-coordinates

import utm
from scipy.spatial import distance

def euclidean_distance(row):
    utm1 = utm.from_latlon(row.pickup_latitude, row.pickup_longitude)
    utm2 = utm.from_latlon(row.dropoff_latitude, row.dropoff_longitude)
    distance_meters = distance.euclidean((utm1[0], utm1[1]),(utm2[0], utm2[1]))
    distance_miles = (distance_meters/1000)*0.621371
    return distance_miles


# In[14]:


#Creating a new column that contains euclidean distance for rides
df_raw['euclidean_distance'] = df_raw.apply(euclidean_distance, axis=1)


# In[15]:


#Creating a new column that contains hour of day of the pickup for rides
df_raw['pickup_hour'] = df_raw.pickup_datetime.apply(lambda t: t.hour)


# In[16]:


#Creating a new column that contains hour of day of the pickup for rides
df_raw['pickup_day'] = df_raw.pickup_datetime.apply(lambda t: t.weekday())


# **Task 2** (a) Pearson Correlation between Euclidean distance of the ride and the taxi fare.

# In[17]:


df_raw['euclidean_distance'].corr(df_raw['fare_amount'])


# **Task 2** (b) Pearson Correlation between time of day and distance traveled.

# In[18]:


df_raw['pickup_hour'].corr(df_raw['euclidean_distance'])


# **Task 2** (c) Pearson Correlation between time of day and the taxi fare.

# In[19]:


df_raw['pickup_hour'].corr(df_raw['fare_amount'])


# The Euclidean distance and taxi fare show highest correlation amongst all the three pairs. There is a very strong correlation between distance and fare, which makes sense as we know that distance is one of the factors used in fare calculation. On the other hand, time of day does not seem to be correlated either with distance traveled or with taxi fare, at least at this point.

# **Task 3** (a) Plot between Euclidean distance of the ride and the taxi fare.

# In[20]:


sns.lmplot(x='fare_amount', y='euclidean_distance', data=df_raw)


# A linear relationship can be seen in the plot above for most of the data. There seem to be some outliers such as trips of zero distance but very high fares and trips of very long distances with zero fare. These trips are justifiable by some logics such as trips starting and ending at same points, but such trip details won't help our model. So I am going to drop them later. 

# **Task 3** (b) Plot between time of day and distance traveled.

# In[21]:


df_raw.pivot_table('euclidean_distance', index='pickup_hour').plot(figsize=(14,6))
matplotlib.pyplot.ylabel('Distance Traveled');


# No useful relationship could be seen in the above plot, which was expected given the small correlation between the two 

# **Task 3** (c) Plot between time of day and the taxi fare.

# In[22]:


df_raw.pivot_table('fare_amount', index='pickup_hour').plot(figsize=(14,6))
matplotlib.pyplot.ylabel('Fare Amount');


# Again, no useful relationship could be seen in the plot at this point.

# Looking at the correlation table for our data, we see that there no other strong correlations.

# In[23]:


df_raw.corr(method='pearson')


# **Data Cleaning** Let's drop the outliers that we saw in Task 3 plot (a)

# In[24]:


#Dropping trips with zero euclidean distance
print("Size before: %d" %len(df_raw))
df_raw = df_raw[(df_raw.euclidean_distance > 0)]
print("Size after: %d" %len(df_raw))


# We know that the taxi meter starts with an initial charge of 2.5<span>$</span> and adds 50 cents for each 1/5 mile. By that logic a 25 mile trip will cost atleast 65<span>$</span>. But these are the present rates. The data we have is old, so we have to factor that as well. Interestingly, the rates haven't changed many times.There is was a good 7 years gap between the hike in 2005 and 2012. We don't have data before 2005. To be on the safe side we will consider rates before 2012. Before 2012 the taxi meter started with an initial charge of 2<span>$</span> and added 30 cents for each 1/5 mile. By that logic a 25 mile trip will cost atleast 39.5<span>$</span>.I am going to drop any trip longer than 25 miles with a fare less than 39.5<span>$</span>. Also every trip has to be of atleast 2<span>$</span>, if not I drop them as well.

# In[25]:


#Dropping trips with fare below 2$
print("Size before: %d" %len(df_raw))
df_raw = df_raw[(df_raw.fare_amount >= 2)]
print("Size after: %d" %len(df_raw))


# In[26]:


#Dropping trips longer than 25 miles with a fare less than 39.5$
print("Size before: %d" %len(df_raw))
df_raw = df_raw.drop(df_raw[(df_raw.euclidean_distance >=25) & (df_raw.fare_amount <= 39.5)].index)
print("Size after: %d" %len(df_raw))


# **Task (4) & (5)** Let us create some new features from the existing data and see if any of these can be useful for our model.

# In[27]:


#Creating a new column that contains year of the pickup for rides
df_raw['pickup_year'] = df_raw.pickup_datetime.apply(lambda t: t.year)


# As mentioned already the taxi rates have increased sometimes over the years. So it will make sense if the year of the trip becomes one of the features of our model.

# In[28]:


#Creating a new column that contains fare per mile for rides
df_raw['fare_per_mile'] = df_raw.fare_amount / df_raw.euclidean_distance


# Let us plot the per mile fares against hours of a day, for each year in our data set.

# In[29]:


df_raw.pivot_table('fare_per_mile', index='pickup_hour', columns='pickup_year').plot(figsize=(16,6))
matplotlib.pyplot.ylabel('Fare Per Mile (USD)');


# The plot reveals that some trips have unrealistic per mile rates. This may have been caused due to the trips which are less than 1 mile.
# Let us set those trip aside for a while.

# In[30]:


#Dropping trips shorter than one mile$
print("Size before: %d" %len(df_raw))
df_for_plotting = df_raw[(df_raw.euclidean_distance > 1)]
print("Size after: %d" %len(df_for_plotting))


# Let us plot again with this new data set.

# In[31]:


df_for_plotting.pivot_table('fare_per_mile', index='pickup_hour', columns='pickup_year').plot(figsize=(14,6))
matplotlib.pyplot.ylabel('Fare Per Mile (USD)');


# As we can see, there are 3 noticeable group of lines. Each group indicates the years through which the taxi rates might have been the same. Thus depending on the year, different rates apply for trips. So year is going to be one of the feature of our model.

# **Task (6)** Now let us create a simple linear regression model using our two features, eulicdean_distance and pickup_year.

# Defining features of our model.

# In[32]:


lr = LinearRegression()
matrix = np.matrix(df_raw)
X = df_raw[['euclidean_distance', 'pickup_year']].values;
y = df_raw['fare_amount'].values;

X.shape, y.shape


# Let us split the training data and generate the model. Also print the coeficients and intercept.

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = lr.fit(X_train, y_train)
model.coef_, model.intercept_


# Finding the RMSE of our model.

# In[34]:


y_pred = model.predict(X_test)
print('Below is the RMSE for Linear model')
sqrt(mean_squared_error(y_test, y_pred))


# In[35]:


model = lr.fit(X, y)
print('Below are the coeficients for Linear model')
model.coef_, model.intercept_


# Preparing our test data.

# In[36]:


#I have preprocessed the test data and generated the trip distance values using google distance matrix api.
df_test =  pd.read_csv('C:/Users/Shubham/Desktop/test_google.csv', parse_dates=["pickup_datetime"])


# In[37]:


#Creating a new column that contains euclidean distance for rides
df_test['euclidean_distance'] = df_test.apply(euclidean_distance, axis=1)

#Creating a new column that contains year of the pickup for rides
df_test['pickup_year'] = df_test.pickup_datetime.apply(lambda t: t.year)


# In[38]:


matrix = np.matrix(df_raw)
X = df_test[['euclidean_distance', 'pickup_year']].values;

y_pred = model.predict(X)


# Saving the output.

# In[39]:


y_pred = model.predict(X)
output = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': y_pred},
    columns = ['key', 'fare_amount'])
output.to_csv('C:/Users/Shubham/Desktop/submission_linear_3M.csv', index = False)


# **Task 7** There are several data sets available on the internet related to the New York Taxi. Below is the list of some of them.
#     <br>(1) 2017 Yellow Taxi Trip Data - Source: NYC Open data, This dataset includes trip records from all trips completed in yellow taxis from in NYC during 2017.
#     <br>(2) Hourly weather data for the New York City Taxi Trip Duration Challenge - Source:Kaggle
#     <br>(3) Google's Distance Matrix API for "New York City Taxi Trip Duration" challenge - Source: Kaggle
# <br><br>I am going to use an external data set. It is not entirely external. I created a data set using 10k rows from the training data and called the Google matrix api for these rows. Then I used this new data set to build a model which predicts trip duration for larger data set (training data set with 3M rows) 

# In[40]:


#Loading the external data set
df_google =pd.read_csv('C:/Users/Shubham/Desktop/train_google2.csv', parse_dates=["pickup_datetime"])
df_google.dtypes


# Building Random Forest Regressive model using the google data set.

# In[41]:


matrix = np.matrix(df_google)
X = df_google[['pickup_longitude', 'pickup_latitude', 'euclidean_distance']]
y = df_google.google_time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train, y_train)
pred = rfr.predict(X_test)
error = sqrt(mean_squared_error(y_test,pred))
print('Below is the RMSE for Random Forest model for predicting trip time')
error


# Let us train our model for entire data set and predict values.

# In[42]:


rfr.fit(X, y)
matrix = np.matrix(df_raw)
X = df_raw[['pickup_longitude', 'pickup_latitude', 'euclidean_distance']]

df_raw['estimated_google_time'] = rfr.predict(X)


# **Task 8** Now let us build another Random Forest Regressive model for predicting the taxi fare with one more feature that we just created.

# In[43]:


matrix = np.matrix(df_raw)
X = df_raw[['euclidean_distance', 'pickup_year', 'estimated_google_time']]
y = df_raw.fare_amount


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X, y)


# Let us use the model to predict values.

# In[45]:


matrix = np.matrix(df_test)
X = df_test[['euclidean_distance', 'pickup_year', 'google_time']]

y_pred = rfr.predict(X)


# Finally we output the submission file containing fare amounts for the corresponding trips.

# In[46]:


submission = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': y_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('C:/Users/Shubham/Desktop/submission_randomforest_estimatedtime.csv', index = False)

