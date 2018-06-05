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
import matplotlib.pyplot as plt


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


# return a dict with labels for easy variable naming later in analysis
def get_month_labels(df):
    try:
        test = df['date']
        del test
    except KeyError:
        raise Exception("df provided must have a 'date' with type datetime")
            
    year_month_list = defaultdict(lambda:0)
    year = df['date'].dt.year.tolist()
    month = df['date'].dt.month.tolist()
    year_set = set()
    month_set = set()
    tmp_list = []

    for i in range(df.shape[0]):
        if year[i] not in year_set:
            year_set.add(year[i])
            if tmp_list:
                year_month_list[year[i-1]] = sorted(list(set(tmp_list)))
            tmp_list = []
        if month[i] < 10:
            month[i] = "0{}".format(month[i])
        tmp_list.append("{0}_{1}".format(month[i], year[i]))
    year_month_list[year[df.shape[0]-1]] = sorted(list(set(tmp_list)))    

    return year_month_list


# isolate one month of posts
def df_for_month(df, month, year):
    if type(month) != int or month < 1 or month > 12:
        return "month must be int between 1 and 12"
    if type(year) != int or year < 1900:
        return "year must be an int from a valid year"
    
    df = df.loc[(df['date'].dt.month == month) & (df['date'].dt.year == year)]
    
    if df.empty:
        raise Exception("There are no posts in the df for {0} {1}.".format(month, year))
    return df


# iteratively call "df_for_month()" to get all month dfs from this dataset
def make_all_months(df, dict_of_months):
    
    all_months_df = defaultdict(lambda:0)
    
    for key in dict_of_months.keys():
        for date_label in dict_of_months[key]:
            month = date_label[:2]
            year = date_label[3:]
            all_months_df["df_{}".format(date_label)] = df_for_month(df, int(month), int(year))
    
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
        elif smoothing == "KN":
            pass
        
        #########################
        # compute cross-entropy #
        #########################
        Vocab = get_vocab_size(model_context_totals)

        cross_entropy = 0

        if verbose == True:
            build_sent = [filter_words[0], filter_words[1]]

        for i in range(len(filter_words)-2):
            if smoothing == "AddK":
                numerator = model_counts.get((filter_words[i],filter_words[i+1]),0)
                if numerator != 0:
                    numerator = numerator.get(filter_words[i+2], 0)
                total = model_context_totals.get((filter_words[i],filter_words[i+1]), 0)
                final_val = (numerator+k)/(total+k*Vocab)


            elif smoothing == "KN":
                pass
            else: # for unsmoothed language model
                numerator = model_counts.get((filter_words[i],filter_words[i+1]),0)
                if numerator != 0:
                    numerator = numerator.get(filter_words[i+2], 0)
                total = model_context_totals.get((filter_words[i],filter_words[i+1]), 0)
                final_val = (numerator)/(total)
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
def make_trigram_model(df, num_users=500, smooth=None, num_post_per_user=2, word_limit=30, all_users=False, verbose=True):

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
            if verbose:
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
            user_text = df[df['user'] == snapshot_users[i]]['text'].tolist()
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

                tmp_index = df.index[(df['user'] == snapshot_users[i]) & (df['text'] == post)].tolist()
                if not tmp_index:
                    raise Exception("Could not find the post in the df; {0}: {1}".format(snapshot_user[i], post))
                #if len(tmp_index) > 1:
                 #   raise Exception("More than 1 index scheduled to drop: {}".format(tmp_index))
                drop_list.append(tmp_index[0])
                
    else: 
        count = 0
        snapshot_users = list(set(df['user']))
        for i in range(len(snapshot_users)):
            count += 1
            if count % 100 == 0:
                print("Currently at user number {}".format(count))
            user_text = df[df['user'] == snapshot_users[i]]["text"].tolist()
            for post in user_text:
                # add post to training text
                training_text.append(post) 
        
    if verbose:
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
    if verbose:
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
    if verbose:
        print("Building counts for LM")
    for filtered_sent in all_filtered_sents:
        for word_1, word_2, word_3 in nltk.trigrams(filtered_sent, pad_right=True, pad_left=True):
            trigram_model[(word_1, word_2)][word_3] += 1   
                

    ##############################
    # Step 6) Get Context Counts #
    ##############################
    if smooth == None:
        if verbose:
            print("Calculating probabilities for LM")
        for word_1_2 in trigram_model:
            total = float(sum(trigram_model[word_1_2].values()))
            for word_3 in trigram_model[word_1_2]:
                trigram_model[word_1_2][word_3] /= total
        if verbose:
            print("Language model successfully built")
        if all_users == False:
            return new_df, trigram_model
        else:
            return trigram_model
    
    
    elif smooth == "AddK":
        if verbose:
            print("Building context totals dict")
        trigram_context_totals = defaultdict(lambda: 0)
        for word_1_2 in trigram_model:
             trigram_context_totals[word_1_2] = float(sum(trigram_model[word_1_2].values())) 
        if verbose:
            print("Models successfully built")
        
        return new_df, trigram_model, trigram_context_totals
    
#########
### WHEN FINAL STATE OF make_trigram_model IS REACHED, BIGRAMS CAN BE WORKED ON
### UNTIL THEN THIS BIGRAM MODEL GENERATION METHOD WILL BE COMMENTED OUT
#########
# creats a bigram language model from given dataframe
# THIS CURRENTLY ONLY MAKES UNSMOOTHED BIGRAM MODELS... PLEASE USE TRIGRAM MODEL TO GET K-SMOOTHED MODEL
# def make_bigram_model(df, num_users=500, num_post_per_user=2, word_limit=30 ):

#     ######################################################################################
#     # The default parameters in this function follow the work of (2012 Jurafsky et al.)  #
#     #                                                                                    #
#     # num_users: number of users to be included in language model training               #
#     # num_post_per_user: number of posts to include from each user in the training data  #
#     # word_limit: limit max number of words that can be in a post in training data       #
#     #                                                                                    #
#     # returns: defaultdict containing language model as well as new df that dropped rows #
#     #          used to train the language model.                                         #
#     ######################################################################################
    
    
#     #################################################
#     # Step 1) get users to include in training data #
#     #################################################
#     user_list = users_with_n_posts(df, num_post_per_user)
    
#     if(len(user_list) >= num_users):
#         snapshot_users = random.sample(user_list, num_users)
#         print("There are {0} users with at least {1} posts from the df provided.".format(len(user_list),num_post_per_user))
#     else:
#         return "Less than {0} users with {1} or more posts in the provided df. Cannot build model.".format(num_users, num_post_per_user)
    
              
#     ###############################
#     # Step 2) build training data #
#     ###############################
              
#     training_text = []
#     drop_list = []
    
#     for i in range(len(snapshot_users)):
#         user_text = df[df['user'] == snapshot_users[i]]["all_text"].tolist()
#         text_sample = random.sample(user_text, num_post_per_user)
#         count_check = 0
#         if len(text_sample) != num_post_per_user:
#             raise Exception("Stopped building LM; didn't pull enough samples from user text.")
#         for post in text_sample:
#             count_check += 1
#             if count_check > num_post_per_user:
#                 raise Exception("More than {} posts sent to training_text.".format(num_post_per_user))
                
#             # add post to training text
#             training_text.append(post) 
            
        
#             tmp_index = df.index[(df['user'] == snapshot_users[i]) & (df['all_text'] == post)].tolist()
#             if not tmp_index:
#                 raise Exception("Could not find the post in the df; {0}: {1}".format(snapshot_user[i], post))
#             if len(tmp_index) > 1:
#                 raise Exception("More than 1 index scheduled to drop: {}".format(tmp_index))
            
#             drop_list.append(tmp_index[0])
                

#     print("LM training data created successfully with {} posts total.".format(len(training_text)))
#     print("\n")
    
    
#     ###############################################
#     # Step 3) create new df without training data #
#     ###############################################
    
#     new_df = df.drop(drop_list)
#     rows_dropped = df.shape[0] - new_df.shape[0]
#     if rows_dropped != len(drop_list):
#         raise Exception("Should have dropped {0} rows but instead dropped {1} rows.".format(len(drop_list), rows_dropped))
        
        
#     ##################################
#     # Step 4) tokenize training data #
#     ##################################
#     print("Building language model...")      
#     bigram_model = defaultdict(lambda: defaultdict(lambda: 0))
#     stop_words = stopwords.words('english') + list(string.punctuation)  
#     all_filtered_sents = []
    
#     # iterate through each post in the training data
#     for i in range(len(training_text)):

#         word_count = 0
#         sent = sent_tokenize(training_text[i])
        
#         # iterate through each sentence in a post
#         for sentence in sent:
#             text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence) #try to get rid of urls in text
#             words = word_tokenize(text)
#             filtered_sentence = [w for w in words if not w in stop_words]

#             # limit posts to first 30 words/tokens
#             word_space_remaining = word_limit - word_count - len(filtered_sentence)
#             if word_space_remaining <= 0:
#                 filtered_sentence = filtered_sentence[0:(len(filtered_sentence)-abs(word_space_remaining))] # limit sentence length
#                 all_filtered_sents.append(filtered_sentence)
#                 break
#             else:
#                 word_count += len(filtered_sentence)
                
#             all_filtered_sents.append(filtered_sentence)
        
        
#     ####################################
#     # Step 5) build counts dict for LM #
#     ####################################

#     for filtered_sent in all_filtered_sents:
#         for word_1, word_2 in nltk.bigrams(filtered_sent, pad_right=True, pad_left=True):
#             bigram_model[word_1][word_2] += 1      
    
    
#     #######################################################
#     # Step 6) finalize LM by calculating LM probabilities #
#     #######################################################
    
#     for word_1 in bigram_model:
#         total = float(sum(bigram_model[word_1].values()))
#         for word_2 in bigram_model[word_1]:
#             bigram_model[word_1][word_2] /= total
    
#     print("Language model successfully built.")
#     return new_df, bigram_model    


# takes a dict of "year":[month1,month2,...] and creats trigram LMs
def trigram_models_by_month(df_by_month):
    
    df_no_train = defaultdict(lambda:None)
    trigram_counts = defaultdict(lambda:0)
    trigram_context_totals = defaultdict(lambda:0)
    
    for key in df_by_month.keys():
        df_no_train[key], trigram_counts[key], trigram_context_totals[key] = make_trigram_model(df_by_month[key], smooth="AddK", verbose=False)
        print("Model for {} built successfully".format(key))
    print("DONE BUILDING MODELS")
    return df_no_train, trigram_counts, trigram_context_totals


# get cross entropy for each month for a specific user
def score_user(user, df_by_month, lm_counts, lm_context_totals):
    
    all_avg_entropy = defaultdict(lambda:None)
    for key in df_by_month.keys():
        df = df_by_month[key]
        user_posts = df[df['user'] == user]['text'].tolist()
        
        cross_entropy_all = []
        for post in user_posts:
            entropy_single = score_sentence(lm_counts[key], post, k=1, smoothing="AddK", model_context_totals=lm_context_totals[key])
            if type(entropy_single) == str:
                pass
            else:
                cross_entropy_all.append(entropy_single)

        all_avg_entropy[key] = float(np.mean(cross_entropy_all))
        
    return all_avg_entropy


# get min,max and warn about months with really low post number
def eda_on_months(df_by_month):
    all_num_posts = defaultdict(lambda:0)
    for key in df_by_month.keys():
        num_posts = df_by_month[key].shape[0]
        all_num_posts[key] = num_posts

    average_posts = 0
    post_counts = []
    for posts in list(all_num_posts.items()):
        average_posts += posts[1]
        post_counts.append(posts[1])
    average_posts /= len(df_by_month)
    min_posts = np.min(post_counts)
    max_posts = np.max(post_counts)
    really_low = average_posts - 2*np.std(post_counts)

    print("Average number of posts each month: {:.2f}".format(average_posts))
    warnings = []
    for key in all_num_posts.keys():
        if all_num_posts[key] == min_posts:
            print("Fewest posts: {0} with {1} posts".format(key, min_posts))
        if all_num_posts[key] == max_posts:
            print("Most posts: {0} with {1} posts".format(key, max_posts))
        if all_num_posts[key] < really_low:
            warnings.append("\x1b[1;31m" + "WARNING: " + "\x1b[0m" + "{0} only has {1} posts!".format(key, all_num_posts[key]))
    print("\n")
    for warning in warnings:
        print(warning)
            

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
            print('\x1b[1;32m'+fake_sentence[0]+'\x1b[0m')
            print("cross-entropy = {:.2f}".format(cross_entropy))
            print("\n")


# plot histogram of number of posts per user
def plot_user_posts(df, save=False, high_res=False):
    all_users = list(Counter(df['user'].tolist()).items())
    user_counts = []
    for user, count in all_users:
        user_counts.append(count)

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,8))

    # plot the same data on both axes
    ax.hist(user_counts, bins="auto")
    ax2.hist(user_counts, bins="auto")

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(100., 6000.)  # outliers only
    ax2.set_ylim(0, 100.)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    f.subplots_adjust(hspace=.03) # changes space between the two y axes

    plt.xlabel("Number of posts")
    ax.set_title("Histogram of posts per user")
    plt.xlim(0,100)
    
    # play with this to add y axis label
    #f.text(0.06, 0.5, 'common ylabel', ha='center', va='center', rotation='vertical')
    
    if save == True:
        if high_res == True:
            plt.savefig("plot_user_posts", dpi=1000)
        else:
            plt.savefig("plot_user_posts")
    else:
        plt.show()
       
            
# plots user_entropy over time
def plot_entropy(user_name, user_entropy, month_labels, labels="black", 
                 title=True, font_size=12, save=False, transparency=False, high_res=True):
    
    ################
    # order months #
    ################
    keys = sorted(list(month_labels.keys()))
    ordered_chaos = []
    for year in keys:
        for month in month_labels[year]:
            if user_entropy.get("df_{}".format(month),None):
                ordered_chaos.append((month,user_entropy["df_{}".format(month)]))
    
    
    ###########################################
    # Make prettier output for months in plot #
    ###########################################
    months = [("01", "January"), ("02", "February"), ("03", "March"), ("04", "April"), ("05", "May"),
                 ("06", "June"), ("07", "July"), ("08", "August"), ("09", "September"), ("10", "October"),
                 ("11", "November"), ("12", "December")]

    month_dict = defaultdict(list)
    for key, value in months:
        month_dict[key] = value
    
    
    #####################
    # make x and y axis #
    #####################
    x_labels = []
    y = []
    for x_tmp, y_tmp in ordered_chaos:
        x_labels.append("{0} {1}".format(month_dict.get(x_tmp[:2],None), x_tmp[3:]))
        y.append(y_tmp)
    x_nums = range(len(x_labels))
    
    
    ############################
    # the actual plotting part #
    ############################
    fig, ax = plt.subplots(1,1) 
    ax.plot(x_nums,y)
    
    if labels == "white":
        ax.tick_params(colors='white')
        ax.set_xticks(x_nums)
        ax.set_ylabel("Cross-entropy", fontsize=12, color="white")
        ax.set_xticklabels(x_labels, rotation='vertical', fontsize=12, color="white")
        if title:
            ax.set_title("User '{}' cross entropy over time".format(user_name), fontsize=14, color="white") 
    elif labels == "black":
        ax.set_xticks(x_nums)
        ax.set_ylabel("Cross-entropy", fontsize=12)
        ax.set_xticklabels(x_labels, rotation='vertical', fontsize=12)
        if title:
            ax.set_title("User '{}' cross entropy over time".format(user_name), fontsize=14)     
                     
    ########################################
    # either save or just display the plot #
    ########################################
    if save == False:   
        plt.show()
    elif save == True:
        if high_res:
            plt.savefig("plot_entropy_{}".format(user_name), transparent=transparency, dpi=1000)
        else:
            plt.savefig("plot_entropy_{}".format(user_name), transparent=transparency)
       
    
    