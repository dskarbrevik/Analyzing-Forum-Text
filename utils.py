import pandas as pd
import nltk
import re
import sys
import string
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from collections import Counter, defaultdict

# remove any admin users from the database
def remove_admins(df, col, admins):
    all_rows_before = df.shape[0]
    not_found_count = 0
    
    for user in admins:
        rows_before = df.shape[0]
        df = df.drop(df[df[col] == user].index)
        rows_after = df.shape[0]
        if rows_before - rows_after == 0:
            print("The user '{}' was not found in the dataframe.".format(user))
            not_found_count += 1
            
    all_rows_after = df.shape[0]
    
    if not_found_count == len(admins):
        print("No one on your admin list is in this dataframe.")
    else:
        print("Found {0} admins in the dataframe. Removing those users dropped {1} rows from the dataframe."
              .format((len(admins) - not_found_count),(all_rows_before - all_rows_after)))
    return df

# check to see if admins in database and return those found
def check_for_admins(df, col, admins):
    count = 0
    admins_found = []
    for user in admins:
        if not df[df[col] == user].empty:
            admins_found.append(user)
        else:
            count += 1
    if count == len(set(admins)):
        print("No admins found in this dataframe.")
    else:
        print("There were admins in the dataframe.")
    return(admins_found)

# tokenize a given post
def count_tokens(posts):
    fdist = nltk.FreqDist()
    stop_words = stopwords.words('english') + list(string.punctuation)
    for i in range(len(posts)):
        if type(posts[i]) ==  str:
            sent = sent_tokenize(posts[i])
            for sentences in sent:
                text = re.sub(r'^https?:\/\/.*[\r\n]*', '', sentences) #try to get rid of urls in text
                words = word_tokenize(text)
                filtered_sentence = [w for w in words if not w in stop_words]
                for words in filtered_sentence:
                    #counts[words] += 1
                    fdist[words] += 1
    return fdist #returns a Counter like object

# isolate one month of posts
def df_for_month(df, month, year):
    if type(month) != int or month < 1 or month > 12:
        return "month must be int between 1 and 12"
    if type(year) != int or year not in [2016, 2017]:
        return "year must be int and either 2016 or 2017"
    
    df = df.loc[(df['date'].dt.month == month) & (df['date'].dt.year == year)]
    
    return df

def make_all_months(df):
    list_months_2016 = [6,7,8,9,10,11,12]
    list_months_2017 = [1,2,3,4,5,6,7,8,9,10,11,12]
    all_months_df = []
    
    for month in list_months_2016:
        all_months_df.append(df_for_month(df, month, 2016))
    for month in list_months_2017:
        all_months_df.append(df_for_month(df, month, 2017))
        
    return all_months_df
        

            

# combine two text columns in a dataframe 
def combine_text_cols(df, col1="topic", col2="text", new_col="all_text"):
    
    if new_col in df.columns.tolist():
        print("Column name '{}' already exists in dataframe. Please delete first or choose new name.".format(new_col))
        
    df[new_col] = ""
    
    for i in range(df.shape[0]):
  
        tmp_col1 = df.at[i, col1]
        tmp_col2 = df.at[i, col2]

        if type(tmp_col1) != str:
            tmp_col1 = ""
        if type(tmp_col2) != str:
            tmp_col2 = ""
        
        if tmp_col1[-1:] in string.punctuation:
            new_text = tmp_col1 + " " + tmp_col2
        else:
            new_text = tmp_col1 + ". " + tmp_col2
        df.set_value(i, new_col, new_text)
    
    return df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    