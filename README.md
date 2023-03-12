## Final Project Submission

Please fill out:
* Student name: Mark Ngige Ndung'u
* Student pace: Full time
* Scheduled project review date/time: 13/03/2023 00:00 EAT
* Instructor name: William Okomba
* Blog post URL:


<h1> Introduction

<h3>Problem Statement

In January 2020, Microsoft Inc. ("the Company") decides to enter the movie industry and creates a movie studio. 
The company approaches consultancy XYZ to provide a data-driven understanding of theindustry and characteristics 
of a successful movie.
The Company is after actionable insights to shape their new venture.

<h3> Our Approach

We will first investigate the notion of profit and profit margin within the movie industry.

We will then explore the characteristics of a successful movie and seek to answer the following questions:

**~** What budget should be allocated?

**~** What movie genres are currently performing best?

**~** What is the optimal movie runtime

**~** What should the release date be to maximise success?

**~** What Original Language to consider for Production

<h3> Data Sources

Data for this project was obtained from the following repository and also saved in ZippedData folder in this repository.

The data comes from:
 
 **~** Box Office Mojo
 
 **~** IMDB
 
 **~** Rotten Tomatoes
 
 **~** TheMovieDB.org
 
 **~** The-numbers

<h3> Methodology

The process can be divided into two main parts.

Section 2 focuses on data preparation.
The steps include:

**~** Importing libraries

**~** Reading provided data

**~** Dealing with missing values and cleaning data

**~** Joining datasets

Section 3 focuses on visualisations and insights.
For each characteristic we will be:

**~** Conducting feature engineering where applicable

**~** Creating visualisations

**~** Drawing conclusions

**~** Providing recommendations

Finally section 4 sets out a summary of our findings, key actionable insights and suggested future work.

<h1>Use of EDA for Precise Bussiness Desion-Making

<h4>Import relevant Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import sqlite3
import csv
import matplotlib.pyplot as plt
%matplotlib inline

<h3>Data Preview before analysis

#Importing/Loading file as a Dataframe.
movie_gross_df = pd.read_csv("bom.movie_gross.csv")
movie_gross_df.head(1)

#Check the shape of the Dataframe.
movie_gross_df.shape

#The above code shows it has 3387rows and 5columns.

#Checking if there are missing values in the dataset and also the datatypes.
movie_gross_df.info()

From the .info() method, we learn that Studio, Domestic_gross and Foreign_gross columns have missing values

#Loading file as a Dataframe.
movie_info_df=pd.read_csv("rt.movie_info.tsv", sep='\t')
movie_info_df.head(1)

# Check the shape of the DataFrame.
movie_info_df.shape

#The shape attribute shows us it has 1560rows and 12columns.

#Check for missing values and datatypes.
movie_info_df.info()

In this dataframe, all columns have missing values except for the id column. Some afew entries while others
lacking over three quaters of the data.All this we will sort during data cleaning.

#Importing file as a Dataframe.
reviews_df = pd.read_csv("rt.reviews.tsv", sep ='\t', encoding ='latin')
reviews_df.head(1)

#Check the shape of the DataFrame
reviews_df.shape

#Has 54432rows and 8columns

#check for missing data.
reviews_df.info()

There are missing values in the review, rating, critic and publisher columns.

#Importing/Loading file as a Dataframe.
movies_df = pd.read_csv("tmdb.movies.csv")
movies_df.head(1)

#check the shape and presence of missing values
print(movies_df.shape)
print(movies_df.info())

#It has 26517rows and 10columns as its shape.

Goodnews....whooa!!,the above data set has no missing values.

#Importing/Loading file as a Dataframe.
movie_budgets_df = pd.read_csv("tn.movie_budgets.csv")
movie_budgets_df.head(1)

#Check for the shape and presence of missing values.
print(movie_budgets_df.shape)
print(movie_budgets_df.info())

#It has 5782rows and 6columns.

Again....Hurray!!!.The dataframe has no missing values.

<h2>Retrieving and Preview Data from SQlite Database

<h4>(1.)Connecting to the database

#Connect to the database
conn = sqlite3.connect("im.db")

#Create a cursor object
cursor = conn.cursor()


<h4>(2.)View list of tables

# Execute a query to get a list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# Fetch all the results as a list of tuples
tables = cursor.fetchall()

# Print the list of tables
for table in tables:
    print(table[0])


From the above query we are able to view the list of tables present in the Database.This enables us to know where to access
which data from and help to create relation between the above Datasets and the Database contents.

<h4>(3.) Selecting data from the tables for preview.

pd.read_sql("""SELECT * FROM movie_basics;""", conn).head(2)

pd.read_sql("""SELECT * FROM directors;""", conn).head(2)

pd.read_sql("""SELECT * FROM known_for;""", conn).head(2)

pd.read_sql("""SELECT * FROM movie_akas;""", conn).head(2)

pd.read_sql("""SELECT * FROM movie_ratings;""", conn).head(2)

pd.read_sql("""SELECT * FROM persons;""", conn).head(2)

pd.read_sql("""SELECT * FROM principals;""", conn).head(2)

pd.read_sql("""SELECT * FROM writers;""", conn).head(2)

Now that we have veiwed the tables in the database, we are able to identify the Primary keys, Foreign keys,the columns present and view any missing values. If we were to check for columns only, a list comprenssion could have been more efficient and tidy.


<h1> Data Preparation and Cleaning 

After previewing the data and knowing it's contents, it time to make the data usable for attaining an uptimum level of accuracy. We could use it as it is, but there is a rule saying "Garbage In Garbage Out" and Garbage results are bad for
strategic Business decisions.
I will be cleaning all of the above datasets so that they can be ready for the relevant analysis.

<h5> (1..) movie_gross Datafame

#Start by removing of duplicates in the dataframe
movie_gross_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

# Act on missing values.
#Calculate the percentage of missing values in the relevant columns
perc_miss1 = (movie_gross_df.isna().sum() / len(movie_gross_df)) * 100
perc_miss1

We get a result show the percentages of data missing in each Column. I get to decide on how to deal with the
missing values based on the percentage and the impact of my choosen method.

For studio removing the offending rows is the suitable option,same to the domestic_gross
column.For the foreign_gross column since it lacks Almost 40% of it's data i wil replace 
with the mean value since dropping leads to remaining with 60% of the available data.

#Droping rows with missing values in the Studio and Domestic_gross columns
movie_gross_df.dropna(axis=0, subset=['studio', 'domestic_gross'], inplace=True)

# Convert the foreign_gross to an integer from an object to be able to calculate mean.
movie_gross_df['foreign_gross'] = pd.to_numeric(movie_gross_df['foreign_gross'], errors='coerce').fillna(0).astype(int)
# Calculate mean value
# Calculate the mean of non-zero values in the column
mean_value = round(movie_gross_df.loc[movie_gross_df['foreign_gross'] != 0, 'foreign_gross'].mean())

# Replace the 0 values with the mean value
movie_gross_df.loc[movie_gross_df['foreign_gross'] == 0, 'foreign_gross'] = mean_value
#check if there is still missing Values
movie_gross_df.info()

<h5> (2..) movie_info_df

#Start by removing of duplicates in the dataframe
movie_info_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

# Act on missing values.
#Calculate the percentage of missing values in the relevant columns
perc_miss2 = (movie_info_df.isna().sum() / len(movie_info_df)) * 100
perc_miss2

The dataframe has missing values an all columns except the id column. It also has plenty of **Irrelevant Data Columns**.
After choosing the relevant columns to be used for analysis, as part of the data cleaning i will drop most columns some 
missing between 65% - 78% of the data. In alignment with the Business problem i hope to solve, i will only be using the Id, rating, genre and Runtime columns.

#Drop the Irrelevant Columns
movie_info_df.drop(columns=['synopsis', 'director','writer','theater_date', 'dvd_date', 
                            'currency', 'box_office', 'studio'], inplace=True)

# Drop rows with mising values since the remaining columns have a maximum of 2% missing.
movie_info_df.dropna(subset=['rating', 'genre', 'runtime'], inplace=True)

<h5>(3..)reviews_df

#Droping all The Duplicates keeping ony the first copy.
reviews_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
reviews_df.info()

# Act on missing values.
#Calculate the percentage of missing values in the relevant columns
perc_miss3 = (reviews_df.isna().sum() / len(reviews_df)) * 100
perc_miss3

According to the percentage of missing values, the best way to clean is to drop the rows with missing values.This being the 
the largest dataframe it will still have enough data for analysis.

#Droping rows with missing values
reviews_df.dropna(subset=['review', 'rating', 'critic', 'publisher'], inplace=True)
reviews_df.info()

<h5> (4..)movies_df

#Droping all The Duplicates keeping ony the first copy.
movies_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
movies_df.info()

The Dataframe is clean and ready for analysis, it does not contain any missing values.

<h5>(5..)movie_budgets_df

#Drop any duplicates available
movie_budgets_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
movie_budgets_df.info()

This Dataframe is now ready for analysis

<h3> Analysis and Visualizations

<h6> Domestic Vs Foreign markets over the years.

This dataset will be used to find the trend of the gross values over the years.
It will help reason which markets are more profitable.

# group the dataframe by years 
grouped_movie_gross = movie_gross_df.groupby('year')


#Plot a linegraph of year vs amount in billion dollars of both domstic and foreign gross.
# Set up the Seaborn figure
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the two line graphs
sns.lineplot( movie_gross_df['year'], movie_gross_df['domestic_gross'], data=movie_gross_df, ax=ax, label='Domestic_gross')
sns.lineplot(movie_gross_df['year'], movie_gross_df['foreign_gross'], data=movie_gross_df, ax=ax, label='Foreign_gross')

# Set axis labels and title
ax.set_xlabel('year')
ax.set_ylabel('Amount in Billion dollars')
ax.set_title('Foreign and Domestic gross ')

# Add a legend
ax.legend()

#Saving the plot as an image
fig.savefig('year vs gross.jpg', bbox_inches='tight', dpi=150)

# Show the plot
plt.show();

<h4> Runtime for movies.

We will use the movie_info_df and some tables in the imdb for this analysis


# Strip the 'min' string from the 'runtime' column and convert to numeric values
movie_info_df['runtime'] = pd.to_numeric(movie_info_df['runtime'].str.replace(' minutes', ''))

# Calculate the mean of the 'runtime' column
mean_runtime = movie_info_df['runtime'].mean()

# Print the average runtime
print(mean_runtime)


We have a average runtime of 104 minutes.
Now we will find the distribution using top 150 movies and their runtime to find a suitable reccommendation.

# sort the data with rating and pick top 150 movies.
#Filter the dataframe to only include rows where the 'rating' column is 'R'
filtered_df = movie_info_df.loc[movie_info_df['rating'] == 'R']

# Sort the filtered dataframe by the 'rating' column in descending order
sorted_df = filtered_df.sort_values('rating', ascending=False)

# Pick the top 150 rows
top_150 = sorted_df[:150]
#reset index
top_150 = top_150.reset_index(drop=True)

print(top_150)

# create a bar plot of movie runtimes
plt.bar(top_150['runtime'], top_150.index, color='blue', alpha=0.5,)

# set axis labels and title
plt.xlabel('Runtime')
plt.ylabel('Movie index')
plt.title('Movie Runtimes Distribution')

plt.savefig('Movie Runtime.png', bbox_inches='tight')
# display the plot
plt.show()

<h5> Genres

The goal is to determine the top selling genres currently.

#Identify the diffrent genres
movie_info_genres = list(set(movie_info_df['genre'].str.split('|').sum()))

# Count the number of occurrences of each string in the 'genre' column
genre_counts = []
count = 0
stripped_top_150 = top_150['genre'].str.split('|').sum()
#initialize an empty dictionary to store the results
genre_dict = {}

# Loop over the items in list_a
for item in movie_info_genres:
    # Count the number of occurrences of the item in list_b
    count = stripped_top_150 .count(item)
    # Add the item and count to the results dictionary
    genre_dict[item] = count
    
    sorted_genre = sorted(genre_dict.items(), key=lambda x: x[1], reverse=True)

# Print the results
for item, count in sorted_genre:
    print(f'{item}: {count}')
        
        

#PLOT a bar graph of the distribution of genre in top gross best 150 movies

# Extract the keys and values from the list
keys = [item[0] for item in sorted_genre]
values = [item[1] for item in sorted_genre]

# Create a horizontal bar graph
fig, ax = plt.subplots()
ax.barh(keys, values)

# Add axis labels and a title
plt.xlabel('Count')
plt.ylabel('Genre')
plt.title('Top Performing Genre')

plt.savefig('Top performing genres.png', bbox_inches='tight')
# Show the plot
plt.show()

<h5> Budget

I will be doing analysis of the production budget for cool fun-loved movies.

movie_budgets_df.head()

# Sort the values by the production budget

# Convert the 'production_budget' column to numeric values
movie_budgets_df['production_budget'] = movie_budgets_df['production_budget'].str.replace('[\$,]', '', regex=True).astype(int)

# Sort the DataFrame by the values of the 'production_budget' column in descending order
movie_budgets_df_sorted = movie_budgets_df.sort_values(by='production_budget', ascending=False)
#Pick top 100 rows
top_100 = movie_budgets_df_sorted[:100]
# Pick the top 100 rows
top_1o0 = sorted_df[:100]
#reset index
top_100 = top_100.reset_index(drop=True)


# Convert the 'worldwide_gross' column to numeric values
movie_budgets_df['worldwide_gross'] = movie_budgets_df['worldwide_gross'].str.replace('[\$,]', '', regex=True).astype(float)

#find  maximum  production budget
maximum_production_budget = movie_budgets_df['production_budget'].max()
print('maximum_production_budget:' ,maximum_production_budget)

average_pruduction_budget = movie_budgets_df['production_budget'].mean()
print('average_pruduction_budget', average_pruduction_budget)

#Calcuate max earning by a movie 
maximum_worldwide_gross =  movie_budgets_df['worldwide_gross'].max()
print('maximum_worldwide_gross:', maximum_worldwide_gross)

#Calculate Profits
profit = maximum_worldwide_gross - maximum_production_budget
print('profit:',profit)

<h5> Release date

Here i will be analysing the appropriate date for releasing movies, months to be precise.

movies_df.head()

#Sort the data with the popularity column
movies_df_sorted = movies_df.sort_values(by='popularity', ascending=False)

# convert date column to datetime format
movies_df_sorted['release_date'] = pd.to_datetime(movies_df_sorted['release_date'])

# extract month information into a new column
movies_df_sorted['month'] = movies_df_sorted['release_date'].dt.month

# compute popularity by month using mean
popularity_by_month = movies_df_sorted.groupby('month')['popularity'].mean()

# create a bar plot
popularity_by_month.plot.bar(x='month', y='popularity', rot=0)
plt.xlabel('Month')
plt.ylabel('Popularity')
plt.title('Release Months vs Popularity')
plt.savefig('Release Months vs Popularity', bbox_inches='tight')
plt.show()

<h5> Original language

I will be analysing the popularity of a movie based on the Original Language of movie.

# Using the sorted dataframe according to popularity.
movies_df_sorted.head()

# get unique languages from the 'original_language' column
unique_languages = movies_df_sorted['original_language'].unique()

# print the unique languages
print(unique_languages)


#Get the number of movie in particular languages
counts = movies_df_sorted['original_language'].value_counts(normalize=False).loc[unique_languages]
sorted_counts = counts.sort_values(ascending=False)

#Sort and get the top 10 languages
first_10 = sorted_counts.head(10)

first10_language = first_10.index

# plot a horizontal bar graph of the items in the Series
first_10.plot(kind='barh')
plt.xlabel('Num of movies')
plt.ylabel('Languages')
plt.title('languages vs num of movies')
plt.savefig('languages vs num of movies', bbox_inches='tight')
plt.show()

# filter the DataFrame to only include the languages in the list
df_filtered = movies_df_sorted[movies_df_sorted['original_language'].isin(first10_language)]

# create a scatter plot between the popularity column and the separate list of languages
plt.scatter(df_filtered['popularity'], df_filtered['original_language'], color='blue', alpha=0.5)

# set the axis labels and title
plt.xlabel('Popularity')
plt.ylabel('Language')
plt.title('Language Popularity')

plt.savefig('Language Popularity Scatter Plot', bbox_inches='tight')
# display the plot
plt.show()

<h1> CONCLUSION AND FUTURE WORKS

<h4>Summary of Findings

<h5>Domestic Vs Foreign markets over the years.

 **~** The Domestic market have been stugnating in the range of between 200 million dollars  
 and 400 million dollars over the years from 2010.
 
 **~** The Foreign market on the other hand has been on an uptrend since 2010.
 Grossing from 500 million dollars to a high of 2 billion dollars for movies like avator. Most movies in 2017 and 2018
 had a worldwide gross of &1 billion dollars on average.

<h6> Budget and Profits

**~** Top grossing movies from 2010 have a production budget ranging between 35 million dollars and 
    500 million dollars.

**~**  Profit can be seen as worldwide gross less production budget but note that this is a simplified
    approach as does not account for other revenue streams and costs.

**~** The profits margin for these movies have been looking really good having a median of 600 million Dollars
    and having maximum profits of over 2 Billion dollars.

<h6> Runtime for Movies

**~** Average runtime is 104 minutes.

**~** Top movies are longer, around 120 minutes.

**~** No direct correlation between runtime and worldwide gross, nor runtime and production budget

<h6> Genre

**~** Overhalf of the movies produced from 2010 have been Drama

**~** Drama, Comedy, Action and Adventure Genres having been the genres for top 
    grossing movies.

**~**  Drama movies have been most profitable since the profit margin was relatively good
     Compared to the production cost involved.

<h6> Release Month

**~** Most successfull months have been May,July and December

<h6> Original Language

**~** Top grossing movies from 2010 have an original language of Production as
    English.

**~** Other languages used are Japanese and Spanish which also have a few top grossing movies    

<h3> Actionable Insights

**1** Have a budget of at least 100 million dollars and ideally between $200-500 million.

**2** Produce a long movie, with a runtime of at least 2 hours.

**3** Produce an action/adventure, Drama, Comedy type of movie.

**4** Aim for a release date in May/June or November

**5** Decide whether to pursue Box Office or online distribution as characteristics differ


<h3> Future Work

As next steps, we would suggest the following:

**~** A reconciliation of the financial data, reviewing other reported sources to strengthen the reliability of the data

**~** Adjust for inflation using Consumer Price Index

**~** Further financial analysis e.g. how to allocate the production budget

**~** Investigate top creative talent (director, producer, actors)

**~** Analysis of additional revenue streamse e.g. merchandise

**~** Sentiment analysis of reviews

**~** Analysis of screenplay source e.g. based on book or original

**~** Analysis of certification ratings

