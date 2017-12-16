import pandas as pd
import nltk
import re
import sys
import string
import random
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

# find users with at least n number of posts
def users_with_n_posts(df, n):
    counter = Counter(df['user'])
    user_list = []
    for values in counter.items():
        if values[1] >= n:
            user_list.append(values[0])
    return user_list


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
                text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentences) #try to get rid of urls in text
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

# iteratively call "df_for_month()" to get all month dfs from this dataset
def make_all_months(df):
    list_months_2016 = [6,7,8,9,10,11,12]
    list_months_2017 = [1,2,3,4,5,6,7,8,9,10]
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
    
# return cross entropy for a given sentence
def score_sentence(model_counts, sentence, smoothing=None, k=None, trigram=True, model_context_totals=None, verbose=False):
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence) #try to get rid of urls in text
    words = word_tokenize(text)
    stop_words = stopwords.words('english') + list(string.punctuation)  
    filter_words = [w for w in words if not w in stop_words]
    if len(filter_words) > 30:
        filter_words = filter_words[0:30]

    # catch errors in calling this function #
    if len(filter_words) < 3:
        return_val = "sentence too short"
        return return_val
    else:
        if smoothing == "AddK":
            if k == None:
                raise Exception("If selecting 'AddK' smoothing, you must choose a value 'k='")
            if model_context_totals == None:
                raise Exception("Must provide a context_totals defaultdict to make AddK calculations.")
        
        
        #########################
        # compute cross-entropy #
        #########################
        Vocab = get_vocab_size(model_context_totals)

        cross_entropy = 0

        if verbose == True:
            build_sent = [filter_words[0], filter_words[1]]

        for i in range(len(filter_words)-2):
            numerator = model_counts.get((filter_words[i],filter_words[i+1]),0)
            if numerator != 0:
                numerator = numerator.get(filter_words[i+2], 0)
            total = model_context_totals.get((filter_words[i],filter_words[i+1]), 0)
            final_val = (numerator+k)/(total+k*Vocab)

            cross_entropy += np.log2(final_val)
            if verbose == True:
                build_sent.append(filter_words[i+2])
                print(build_sent)
                print("cross entropy = {}".format(cross_entropy))

        if verbose == False:

            return -cross_entropy   
    
# create a trigram language model from the dataframe that is provided  
# there are a lot of catches for errors b/c it took a lot of troubleshooting for me to realize
# there was actually a problem with the data not the model algorithm...
def make_trigram_model(df, num_users=500, smooth=None, num_post_per_user=2, word_limit=30, all_users=False):

    ######################################################################################
    # The default parameters in this function follow the work of (2012 Jurafsky et al.)  #
    #                                                                                    #
    # num_users: number of users to be included in language model training               #
    # num_post_per_user: number of posts to include from each user in the training data  #
    # word_limit: limit max number of words that can be in a post in training data       #
    # all_users: if you want to train a model on all data                                #
    # returns: defaultdict containing language model as well as new df that dropped rows #
    #          used to train the language model.                                         #
    ######################################################################################
    
    
    #################################################
    # Step 1) get users to include in training data #
    #################################################
    
    if all_users == False:
        user_list = users_with_n_posts(df, num_post_per_user)

        if(len(user_list) >= num_users):
            snapshot_users = random.sample(user_list, num_users)
            print("There are {0} users with at least {1} posts from the df provided.".format(len(user_list),num_post_per_user))
        else:
            return "Less than {0} users with {1} or more posts in the provided df. Cannot build model.".format(num_users,num_post_per_user)
              
    ###############################
    # Step 2) build training data #
    ###############################
              
    training_text = []
    drop_list = []
    
    if all_users == False:
    
        for i in range(len(snapshot_users)):
            user_text = df[df['user'] == snapshot_users[i]]["all_text"].tolist()
            text_sample = random.sample(user_text, num_post_per_user)
            count_check = 0
            if len(text_sample) != num_post_per_user:
                raise Exception("Stopped building LM; didn't pull enough samples from user text.")
            for post in text_sample:
                count_check += 1
                if count_check > num_post_per_user:
                    raise Exception("More than {} posts sent to training_text.".format(num_post_per_user))

                # add post to training text
                training_text.append(post) 

                tmp_index = df.index[(df['user'] == snapshot_users[i])].tolist()
                if not tmp_index:
                    raise Exception("Could not find the post in the df; {0}: {1}".format(snapshot_user[i], post))
                #if len(tmp_index) > 1:
                 #   raise Exception("More than 1 index scheduled to drop: {}".format(tmp_index))
                for val in tmp_index:
                    drop_list.append(val)
                
    else: 
        count = 0
        snapshot_users = list(set(df['user']))
        for i in range(len(snapshot_users)):
            count += 1
            if count % 100 == 0:
                print("Currently at user number {}".format(count))
            user_text = df[df['user'] == snapshot_users[i]]["all_text"].tolist()
            for post in user_text:
                # add post to training text
                training_text.append(post) 
        
    
    print("LM training data created successfully with {} posts total.".format(len(training_text)))
    
    
    ###############################################
    # Step 3) create new df without training data #
    ###############################################
    if all_users == False:
        new_df = df.drop(drop_list)
        #rows_dropped = df.shape[0] - new_df.shape[0]
        #if rows_dropped != len(drop_list):
         #   raise Exception("Should have dropped {0} rows but instead dropped {1} rows.".format(len(drop_list), rows_dropped))

        
    ##################################
    # Step 4) tokenize training data #
    ##################################
    print("Tokenizing training data")      
    trigram_model = defaultdict(lambda: defaultdict(lambda: 0))
    stop_words = stopwords.words('english') + list(string.punctuation)  
    all_filtered_sents = []
    
    # iterate through each post in the training data
    for i in range(len(training_text)):

        word_count = 0
        sent = sent_tokenize(training_text[i])
        
        # iterate through each sentence in a post
        for sentence in sent:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence) #try to get rid of urls in text
            words = word_tokenize(text)
            filtered_sentence = [w for w in words if not w in stop_words]

            # limit posts to first 30 words/tokens
            word_space_remaining = word_limit - word_count - len(filtered_sentence)
            if word_space_remaining <= 0:
                filtered_sentence = filtered_sentence[0:(len(filtered_sentence)-abs(word_space_remaining))] # limit sentence length
                all_filtered_sents.append(filtered_sentence)
                break
            else:
                word_count += len(filtered_sentence)
                
            all_filtered_sents.append(filtered_sentence)
        
        
    ####################################
    # Step 5) build counts dict for LM #
    ####################################
    print("Building counts for LM")
    for filtered_sent in all_filtered_sents:
        for word_1, word_2, word_3 in nltk.trigrams(filtered_sent, pad_right=True, pad_left=True):
            trigram_model[(word_1, word_2)][word_3] += 1   
                

    ##############################
    # Step 6) Get Context Counts #
    ##############################
    if smooth == None:
        print("Calculating probabilities for LM")
        for word_1_2 in trigram_model:
            total = float(sum(trigram_model[word_1_2].values()))
            for word_3 in trigram_model[word_1_2]:
                trigram_model[word_1_2][word_3] /= total

        print("Language model successfully built")
        if all_users == False:
            return new_df, trigram_model
        else:
            return trigram_model
    
    
    elif smooth == "AddK":  
        print("Building context totals dict")
        trigram_context_totals = defaultdict(lambda: 0)
        for word_1_2 in trigram_model:
             trigram_context_totals[word_1_2] = float(sum(trigram_model[word_1_2].values())) 
        print("Models successfully built")
        
        return new_df, trigram_model, trigram_context_totals
    
        
    
# creats a bigram language model from given dataframe
# THIS CURRENTLY ONLY MAKES UNSMOOTHED BIGRAM MODELS... PLEASE USE TRIGRAM MODEL TO GET K-SMOOTHED MODEL
def make_bigram_model(df, num_users=500, num_post_per_user=2, word_limit=30 ):

    ######################################################################################
    # The default parameters in this function follow the work of (2012 Jurafsky et al.)  #
    #                                                                                    #
    # num_users: number of users to be included in language model training               #
    # num_post_per_user: number of posts to include from each user in the training data  #
    # word_limit: limit max number of words that can be in a post in training data       #
    #                                                                                    #
    # returns: defaultdict containing language model as well as new df that dropped rows #
    #          used to train the language model.                                         #
    ######################################################################################
    
    
    #################################################
    # Step 1) get users to include in training data #
    #################################################
    user_list = users_with_n_posts(df, num_post_per_user)
    
    if(len(user_list) >= num_users):
        snapshot_users = random.sample(user_list, num_users)
        print("There are {0} users with at least {1} posts from the df provided.".format(len(user_list),num_post_per_user))
    else:
        return "Less than {0} users with {1} or more posts in the provided df. Cannot build model.".format(num_users, num_post_per_user)
    
              
    ###############################
    # Step 2) build training data #
    ###############################
              
    training_text = []
    drop_list = []
    
    for i in range(len(snapshot_users)):
        user_text = df[df['user'] == snapshot_users[i]]["all_text"].tolist()
        text_sample = random.sample(user_text, num_post_per_user)
        count_check = 0
        if len(text_sample) != num_post_per_user:
            raise Exception("Stopped building LM; didn't pull enough samples from user text.")
        for post in text_sample:
            count_check += 1
            if count_check > num_post_per_user:
                raise Exception("More than {} posts sent to training_text.".format(num_post_per_user))
                
            # add post to training text
            training_text.append(post) 
            
        
            tmp_index = df.index[(df['user'] == snapshot_users[i]) & (df['all_text'] == post)].tolist()
            if not tmp_index:
                raise Exception("Could not find the post in the df; {0}: {1}".format(snapshot_user[i], post))
            if len(tmp_index) > 1:
                raise Exception("More than 1 index scheduled to drop: {}".format(tmp_index))
            
            drop_list.append(tmp_index[0])
                

    print("LM training data created successfully with {} posts total.".format(len(training_text)))
    print("\n")
    
    
    ###############################################
    # Step 3) create new df without training data #
    ###############################################
    
    new_df = df.drop(drop_list)
    rows_dropped = df.shape[0] - new_df.shape[0]
    if rows_dropped != len(drop_list):
        raise Exception("Should have dropped {0} rows but instead dropped {1} rows.".format(len(drop_list), rows_dropped))
        
        
    ##################################
    # Step 4) tokenize training data #
    ##################################
    print("Building language model...")      
    bigram_model = defaultdict(lambda: defaultdict(lambda: 0))
    stop_words = stopwords.words('english') + list(string.punctuation)  
    all_filtered_sents = []
    
    # iterate through each post in the training data
    for i in range(len(training_text)):

        word_count = 0
        sent = sent_tokenize(training_text[i])
        
        # iterate through each sentence in a post
        for sentence in sent:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence) #try to get rid of urls in text
            words = word_tokenize(text)
            filtered_sentence = [w for w in words if not w in stop_words]

            # limit posts to first 30 words/tokens
            word_space_remaining = word_limit - word_count - len(filtered_sentence)
            if word_space_remaining <= 0:
                filtered_sentence = filtered_sentence[0:(len(filtered_sentence)-abs(word_space_remaining))] # limit sentence length
                all_filtered_sents.append(filtered_sentence)
                break
            else:
                word_count += len(filtered_sentence)
                
            all_filtered_sents.append(filtered_sentence)
        
        
    ####################################
    # Step 5) build counts dict for LM #
    ####################################

    for filtered_sent in all_filtered_sents:
        for word_1, word_2 in nltk.bigrams(filtered_sent, pad_right=True, pad_left=True):
            bigram_model[word_1][word_2] += 1      
    
    
    #######################################################
    # Step 6) finalize LM by calculating LM probabilities #
    #######################################################
    
    for word_1 in bigram_model:
        total = float(sum(bigram_model[word_1].values()))
        for word_2 in bigram_model[word_1]:
            bigram_model[word_1][word_2] /= total
    
    print("Language model successfully built.")
    return new_df, bigram_model    


# get size of vocab in model
def get_vocab_size(model_context):
    vocab_size = set()
    for context in model_context.keys():
        vocab_size.add(context[0])
        vocab_size.add(context[1])
    return len(list(vocab_size))
    

# generate toy sentences from supplied language models
# only works on trigrams currently
def generate_sentences(model_count, model_context, ngram="trigram", n=1, k=1, save=True):
    
    if (ngram != "trigram"):
        raise Exception("model_type paramter must be 'trigram'")
    
    list_of_sents = []
    Vocab = get_vocab_size(model_context)

    if ngram == "trigram":   
        
            for i in range(n):
                text = [None, None]

                sentence_finished = False

                while not sentence_finished:
                    r = random.random()
                    accumulator = .0

                    for word in model_count[tuple(text[-2:])].keys():
                        numerator = model_count[tuple(text[-2:])].get(word,0)
                        total = model_context.get(tuple(text[-2:]), 0)
                        final_val = (numerator+k)/(total+k*Vocab)

                        accumulator += final_val

                        if accumulator >= r:
                            text.append(word)
                            break

                    if text[-2:] == [None, None]:
                        sentence_finished = True
                if save == False:
                    print(' '.join([t for t in text if t]))
                else:
                    list_of_sents.append(' '.join([t for t in text if t]))
    
    if save == True:
        return(list_of_sents)
 

 # pretty print fake sentences
def print_fake_sentences(model_counts, model_context, n=1, k=1):
    count = 0
    while count < n:
        fake_sentence = generate_sentences(model_counts, model_context, n=1, save=True)    
        cross_entropy = score_sentence(model_counts, fake_sentence[0], smoothing="AddK", k=1, 
                                             model_context_totals=model_context)

        if cross_entropy != "sentence too short":
            count +=1
            print("")
            print(fake_sentence[0])
            print("cross-entropy = {}".format(cross_entropy))
