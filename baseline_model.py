#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install rouge')
get_ipython().system('pip install fastapi nest-asyncio pyngrok uvicorn')


# In[2]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import sys
from collections import Counter
import random
import re
import numpy as np
from rouge import Rouge


# In[ ]:





# In[3]:


"""Initializes Parameters:
  n_gram (int): the n-gram order.
  is_laplace_smoothing (bool): whether or not to use Laplace smoothing
  threshold: words with frequency  below threshold will be converted to token
"""
# Initializing different object attributes
n_gram = 3
is_laplace_smoothing = True
vocab = [] 
n_gram_counts = {}
n_minus_1_gram_counts = None
threshold = 1

V = 0
alpha = 1
UNK = "<UNK>"
SENT_BEGIN = "<s>"
SENT_END = "</s>"
PAR_BEGIN = "<p>"
PAR_END = "</p>"


# In[4]:


hyperlinks_str = r'\bhttp[sS]?://[A-Za-z0-9\./-_$&@=\+%#\(\)!`~\?\]\[\}\{<>\^\*]+\b'
unicode_str = r'&[A-Za-z0-9]+;'
punct_str = r'[^A-Za-z0-9</> ]'
whitespace_str = r'\s+'

def string_cleanup(string_):
    #replacing hyperlinks to http, most of the spams can have a hyperlink
    str_ = re.sub(hyperlinks_str,'http',string_) 
    #remove unicode 
    str_ = re.sub(unicode_str,'',str_)
    #remove puctuations 
    str_ = re.sub(punct_str,'',str_)
    #remove extra whitespace  
    str_ = re.sub(whitespace_str,' ',str_)
    return str_.strip().lower()

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(set(['may','might','can','could','have','had','has']))
def clean_message(message, stopword_ = True):
    '''
    Input:
        message: a string containing a message.
    Output:
        messages_cleaned: a list of words containing the processed message. 

    '''
    #stemmer = porter_stemmer
    english_stopwords = stop_words
    if stopword_ == False:
        english_stopwords = []
    message = string_cleanup(message)
    message_tokens = message.split()

    messages_cleaned = []
    last_word = SENT_BEGIN
    for word in message_tokens:
        if word not in english_stopwords:  #removing stopwords
            #stem_word = stemmer.stem(word)  #using stemming
            if last_word==SENT_BEGIN and word == SENT_END:
                messages_cleaned = messages_cleaned[:-1]
            else:
                last_word = word
                messages_cleaned.append(word)       

    return messages_cleaned


# In[5]:


def append_marker(sent,n_gram):
    return sent.replace(PAR_BEGIN, ' '.join([PAR_BEGIN] * (max(n_gram - 1, 1)))).replace(PAR_END, ' '.join(
            [PAR_END] * (max(n_gram - 1, 1))))

def prob(gram, n_gram_counts, n_minus_1_gram_counts, n_gram, V):
    if n_gram>1:
        num = n_gram_counts.get(gram, 0) + alpha
        denom = n_minus_1_gram_counts.get(gram[:-1], 0) + V * alpha
    else:
        num = n_gram_counts.get(gram, 0) + alpha
        denom = total_tokens + V * alpha
    return num/denom

def make_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
    Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

    Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
    """
    n_grams = [tuple(tokens[k:k + n]) for k in range(len(tokens) - n + 1)]
    ## Your code here 
    return n_grams

def get_key_phrase(tokens):
    r = Rake()
    # Extraction given the text.
    r.extract_keywords_from_sentences(['_'.join(e) for e in ngrams])

    # To get keyword phrases ranked highest to lowest with scores.
    return r.get_ranked_phrases_with_scores()[0][1]

def get_dicts(n_gram_counts,n_gram_counts_b):
    n_1_gram_dict = {}
    for k, v in n_gram_counts.items():
        p = n_1_gram_dict.get(tuple(k[:-1]), [])
        p.append(k[-1])
        n_1_gram_dict[k[:-1]] = p

    n_1_gram_dict_b = {}    
    for k, v in n_gram_counts_b.items():
        p = n_1_gram_dict_b.get(tuple(k[:-1]), [])
        p.append(k[-1])
        n_1_gram_dict_b[k[:-1]] = p
    return n_1_gram_dict, n_1_gram_dict_b


# In[6]:


n_gram_counts = None
def train(content,n_gram):
    """Trains the language model on the given data. Input file that
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Parameters:
      training_file_path (str): the location of the training data to read

    Returns:
    N Gram Counts, Vocab, N Minus 1 Gram Counts
    """

    # Read and split data to get list of words

    # Get the count of each word
    counts = Counter(content)
    # Replace the words with <UNK> if count is < threshold(=1)
    #unks = [k for k, v in counts.items() if v <= 1]

    #content = [e if e not in unks else UNK for e in content]    
    n_grams = make_ngrams(content, n_gram)
    n_gram_counts = Counter(n_grams)
    
    content_b = list(reversed(content))
    n_grams_b = make_ngrams(content_b, n_gram)
    n_gram_counts_b = Counter(n_grams_b)
    total_tokens = len(content)
    vocab = set(content)
    phrases = list(n_gram_counts.keys())
    p = np.array(list(n_gram_counts.values()))
    j = np.random.choice(range(len(phrases)),p = p/p.sum())
    keyphrase = phrases[j]
    #keyphrase = random.sample(n_grams,k=1)[0]
    if n_gram > 1:
        n_minus_1_gram_counts = Counter(make_ngrams(content, n_gram-1))
        n_minus_1_gram_counts_b = Counter(make_ngrams(content_b, n_gram-1))
    # Get the count of each word
    
    # Replace the words with <UNK> if count is < threshold(=1)
    # make use of make_n_grams function
    # Get the training data vocabulary
    # For n>1 grams compute n-1 gram counts to compute probability
    return n_gram_counts,n_gram_counts_b, vocab, n_minus_1_gram_counts, n_minus_1_gram_counts_b, keyphrase


# In[7]:


V = len(vocab)
print(n_gram_counts)
print(vocab)


# In[8]:


def generate_sents(seed_phrase, n_1_gram_dict, n_gram_counts, n_gram,END_CHAR, V, max_len=25):
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      str: the generated sentence
    """
    # Start with <s> and randomly generate words until we encounter sentence end
    # Append sentence begin markers for n>2
    # Keep track of previous word for stop condition
    prev_word = seed_phrase    
    sentence = [seed_phrase]
    k = 0
    if n_gram > 1:
        word = seed_phrase

        while prev_word != END_CHAR and k <=max_len:

            prs = []
            for each in n_1_gram_dict[word]:
                prs.append(prob(tuple(list(word) + [each]),n_gram_counts, n_gram_counts, n_gram, V))
            prev_word = np.random.choice(n_1_gram_dict[word], 1, prs)[0]
            sentence.append(prev_word)
            word = tuple(list(word[1:]) + [prev_word])
            k+=1

    else:
        # In case of unigram model, n-1 gram is just the previous word and possible choice is whole vocabulary
        word = seed_phrase

        while prev_word != END_CHAR and k <= max_len:

            prs = [prob(e) for e in vocab ]
            prev_word = np.random.choice(list(vocab), 1, prs)[0]
            sentence.append(prev_word)
            word = tuple(list(word[1:]) + [prev_word])
            k += 1
        if k > max_len and prev_word!= END_CHAR:
            sentence.append(END_CHAR)

    # Append sentence end markers for n>2
    
    return sentence


# In[9]:


def generate_summary(text, n_gram = 3):
    
    #Append the Sent start and end characters
    content = f"{PAR_BEGIN} {' '.join([ f'{SENT_BEGIN} {e} {SENT_END}' for e in sent_tokenize(text)])} {PAR_END}"
    content = clean_message(content, stopword_=False)
    #clean the sentence and get the key phrase
    #cleaned = clean_message(content)
    #create n_gram and n_gram-1 dictionaries
    n_gram_counts,n_gram_counts_b, vocab, n_minus_1_gram_counts, n_minus_1_gram_counts_b, keyphrase = train(content,n_gram)
    
    n_1_gram_dict, n_1_gram_dict_b = get_dicts(n_gram_counts,n_gram_counts_b)
    V = len(vocab)
    forward_sents = []
    if '</p>' not in keyphrase:
        forward_sents = generate_sents(keyphrase[1:], n_1_gram_dict, n_gram_counts,n_gram,'</p>',V)
        forward_sents = forward_sents[1:]
    backward_sents = []    
    if '<p>' not in keyphrase:        
        backward_sents = generate_sents(tuple(reversed(keyphrase[:-1])), n_1_gram_dict_b,n_gram_counts_b ,n_gram,'<p>',V)
        backward_sents = list(reversed(backward_sents[1:]))
    
    #create same for backward
    #create sentence 
    return ' '.join(backward_sents + list(keyphrase) + forward_sents).replace(SENT_END,'').replace(SENT_BEGIN,'')                                    .replace(PAR_END,'').replace(PAR_BEGIN,'').strip()
    


# In[ ]:





# In[396]:


train_ds = pd.read_csv('train_sample.csv')
train_ds['generated'] = train_ds['article'].map(generate_summary)
rouge = Rouge()
scores = rouge.get_scores(train_ds['generated'], train_ds['highlights'],avg=True)


# In[398]:


scores


# In[399]:


val_ds = pd.read_csv('val_sample.csv')
val_ds['generated'] = val_ds['article'].map(generate_summary)
rouge = Rouge()
scores = rouge.get_scores(val_ds['generated'], val_ds['highlights'],avg=True)


# In[400]:


scores


# In[401]:


test_ds = pd.read_csv('test_sample.csv')
test_ds['generated'] = test_ds['article'].map(generate_summary)
rouge = Rouge()
scores = rouge.get_scores(test_ds['generated'], test_ds['highlights'],avg=True)
scores


# In[14]:


train_ds = pd.read_csv('train_sample.csv')
train_ds['generated'] = train_ds['article'].map(generate_summary)


# In[28]:


train_ds['generated'][50]


# In[29]:


train_ds['highlights'][50]


# In[30]:


train_ds['article'][50]


# In[12]:



from fastapi import FastAPI
import nest_asyncio
from pyngrok import ngrok
import uvicorn
app = FastAPI()
@app.get('/test/{raw_text}')
async def home(raw_text):
        summary_result = generate_summary(raw_text)
        return summary_result
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)


# In[11]:


generate_summary("this is a summary")


# In[ ]:




