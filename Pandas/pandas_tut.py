import pandas as pd

city_names = pd.Series(['San Francisco','San Jose','Santa Claura'])
population = pd.Series([852469, 1015785, 485199])
df = pd.DataFrame({'City':city_names , 'Population' : population})
print df
df['Area Square miles'] = pd.Series([46.87,176.53,97.92])
df['Population Density'] = df['Population'] / df['Area Square miles']
df['cond'] = (df['Area Square miles'] > 50) & df['City'].apply(lambda name: name.startswith('San'))
print df
print df.index
#california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
#print california_housing_dataframe.describe()
#print california_housing_dataframe.head()



