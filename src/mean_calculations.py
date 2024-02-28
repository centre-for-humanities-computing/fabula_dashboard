# script for reading in .xlsx files and calculating the mean of the columns with numerical data

# read in the file
import pandas as pd
df = pd.read_excel('data/CHICAGO_MEASURES_FEB24.xlsx')

# calculate the mean of the columns with numerical data
# identify the columns with numerical data
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
# calculate the mean of the columns with numerical data
mean = df[numerical_columns].mean()
# save the mean to a new dataframe with the old column names and one row wth the mean
mean_df = pd.DataFrame(mean, columns=['Mean'])
# transpose the dataframe
mean_df = mean_df.T
mean_df.pop('BOOK_ID', 'LIBRARIES', 'WRITTEN_AS', 'PUBL_DATE', 'PULITZER', 'NBA', )

# save the mean dataframe to a csv file
mean_df.to_csv('data/mean.csv')
