
# coding: utf-8

# In[ ]:


import pandas as pd
from numpy import nan
# twitter archive file (downloaded manually)
twi_arch = pd.read_csv("twitter-archive-enhanced.csv")


# # Gathering Data

# In[ ]:


# download the image_prediction.tsv file programatically
import requests

url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
response = requests.get(url)
with open("image_prediction.tsv", "wb") as file:
    file.write(response.content)


# In[ ]:


# load api for twitter
import tweepy

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[ ]:


# creat the tweet_json.txt file using tweepy
import json
import time

count = 0
not_found = []
with open('tweet_json.txt', 'w') as outfile:
    for t_id in twi_arch.tweet_id:
        start = time.time()
        try:
            tweet = api.get_status(t_id, tweet_mode='extended')
        except:
            print("No status found with that ID # {}\n".format(t_id))
            not_found.append(t_id)
            continue
        json.dump(tweet._json, outfile)
        outfile.write('\n')
        end = time.time()
        lapse = end - start
        count += 1
        print("{} done! in {:.2f} seconds".format(t_id, lapse))
        print("{} tweets collected.\n".format(count))

print("The following twitter id's were not found:", not_found, sep="\n")


# In[ ]:


# in case the above cell was not run
not_found = [888202515573088257, 873697596434513921, 869988702071779329, 
             866816280283807744, 861769973181624320, 845459076796616705, 
             842892208864923648, 837012587749474308, 827228250799742977, 
             802247111496568832, 775096608509886464]


# # Assessing Data

# In[ ]:


# load downloaded files into pandas dataframes
predict = pd.read_csv("image_prediction.tsv", sep='\t')
tweet_data = pd.read_json('tweet_json.txt', lines=True)
# get only id, retweet_count, favorite_count
tweet_data = tweet_data[['id', 'retweet_count', 'favorite_count']]


# #### Let's take a look at our data

# In[ ]:


twi_arch.info()


# In[ ]:


twi_arch.describe()


# In[ ]:


twi_arch.timestamp.value_counts()[0:10] # type of data time


# In[ ]:


twi_arch.source.value_counts() # source has only four values; categorical variables


# In[ ]:


twi_arch[twi_arch['in_reply_to_status_id'].notnull()] # this means these are replies
# they might still be good


# In[ ]:


twi_arch[twi_arch['retweeted_status_id'].notnull()] # these are retweets;
# we don't want retweets, so we will not include them in our master table


# In[ ]:


print(sum(twi_arch.expanded_urls.isnull()) )
twi_arch[twi_arch['expanded_urls'].isnull()] # missing expanded_urls
# could these be tweets with no pictures?...


# In[ ]:


twi_arch.text[64] # these are mostly replies


# In[ ]:


# check if all missing expanded_urls are in replies
miss_url_id = list(twi_arch[twi_arch['expanded_urls'].isnull()].tweet_id)
replies_id = list(twi_arch[twi_arch['in_reply_to_status_id'].notnull()].tweet_id)

for iden_num in miss_url_id:
    if iden_num not in replies_id:
        print(iden_num)
    else:
        pass


# In[ ]:


print(replies_id[0:10])
print(miss_url_id[0:10])
twi_arch[twi_arch.name == "Pipsy"]


# In[ ]:


# the four id's above in expanded_urls do not have a picture.
# all rows with no expanded_urls do not have picture


# In[ ]:


twi_arch.puppo.value_counts() # need dog stage variable not four columns


# In[ ]:


json_id = list(tweet_data.id)
twi_id = list(twi_arch.tweet_id)
pre_id = list(predict.tweet_id)

for num in pre_id:
    assert num in twi_id
# all twitter id's in prediction table are in twi_arch table
c = list(set(twi_id)^set(pre_id)) # list of id not in predicted table

print(len(c))


# In[ ]:


twi_arch[twi_arch.tweet_id.duplicated()] #no duplicated tweets


# In[ ]:


twi_arch.name.str.extractall("(^[a-z]+$)")  # names that are not capitalized are not actualy names


# In[ ]:


twi_arch.loc[649] # actual name is 'Forrest' 


# In[ ]:


twi_arch.expanded_urls[801] # same url multiple times (number of pictures)
twi_arch.expanded_urls[721]


# In[ ]:


twi_arch.rating_denominator.value_counts()


# In[ ]:


twi_arch.name.value_counts()


# In[ ]:


# after having submited my project once, I was informed of a very important issue
twi_arch.text[1689] # source text has a denominator of 9.5
# and for rating_numerator it has a 5, I need to fix this


# In[ ]:


predict.info()


# In[ ]:


predict.sample(5)


# In[ ]:


predict.p1_dog.value_counts()


# In[ ]:


tweet_data.info()


# #### Quality (8 issues or more)
# ##### `twi_arch` table
# - retweeted_status_id, retweeted_status_user_id, and retweeted_status_timestamp is information of original tweets.
# - expanded_urls contains urls for number of photos, videos and urls in text of tweet
# - there are 59 missing urls in expanded_urls
# - in_reply_status_id is information of tweets that are replies, but they still score dogs and some have pictures.
# - 'source' is in HTML form, only need source name, no urls.
# - Names are sometimes not real name but other lower-case words
#     - (index: 649) dog with tweet_id: 792913359805018113 is named 'Forrest' not 'a'
# - erronous data types (e.g. tweet_id, reply info, retweet info, timestamps, rating_numerator and denominator, source)
# - text rating and rating_numerator and rating_denominator might not match (after getting rid of retweets and tweets with no pictures)
# - tweet with rating_denominator of 7 is not a tweet of an actual rating
# - two outliers with rating numerator of 420 and 1776
# 
# ##### `predict` table
# - for p1, p2, and p3 the breed of the dog has _ instead of white space between words; all should be lower case for consistency
# - tweet_id not integer
# 
# ##### `tweet_data` table
# - 'id' variable should be labaled the same as for `twi_arch` table, 'tweet_id'
# - tweet_id not integer
# - 11 tweets not found

# #### Tidiness (2 issues or more)
# - drop uneccessary columns in `twi_arch` (retweets and replies)
# - one variable (dog_stage) in four columns in `twi_arch` table (doggo, floofer, pupper, and puppo)
# - there is no score_ratio value
# - number of images should be in the `predict` table
# - number of likes and retweet count should be in the `twi_arch` table 
# - merge the tables of `twi_clean` and `tweet_clean` to get retweet count and favorite count

# # Cleaning Data

# In[ ]:


# make copies of the tables
twi_clean = twi_arch.copy()
pre_clean = predict.copy()
tweet_clean = tweet_data.copy()


# #### `twi_arch`: 78 retweets present in this table

# ##### Define
# Delete these tweets. They are not needed since we want original tweets

# ##### Code

# In[ ]:


twi_clean = twi_clean[twi_clean.retweeted_status_id.isnull()]


# ##### Test

# In[ ]:


# should be empty
twi_clean[twi_clean.retweeted_status_id.notnull()]


# In[ ]:


twi_clean.info()


# #### `twi_arch`: expanded_urls contains urls for number of photos, videos and urls in text of tweet

# ##### Define
# Create number of photos in each tweet off expanded_urls

# ##### Code

# In[ ]:


# ensures they come from dog_rates

def photo_count(table):
    i = str(table['tweet_id'])
    url = str(table['expanded_urls'])
    s = "https://twitter.com/dog_rates/status/" + i + "/photo/1"
    return url.count(s)

# in case I wanted number of images, but I don't
# def video_count(table):
#     i = str(table['tweet_id'])
#     url = str(table['expanded_urls'])
#     s = "https://twitter.com/dog_rates/status/" + i + "/video/1"
#     return url.count(s)
# twi_clean["num_of_vid"] = twi_clean.apply(video_count, axis=1)

twi_clean["num_of_img"] = twi_clean.apply(photo_count, axis=1)


# ##### Test

# In[ ]:


twi_clean.num_of_img.value_counts()


# #### `twi_arch`: various tweets have zero images

# ##### Define
# Delete these tweets since we need tweets with images

# ##### Code

# In[ ]:


twi_clean = twi_clean[twi_clean.num_of_img != 0]


# ##### Test

# In[ ]:


twi_clean.num_of_img.value_counts()


# #### `twi_arch`: there are 59 missing urls in expanded_urls; all rows with no expanded_urls do not have picture

# ##### Define
# We only want tweets with pictures, so delete these tweets (rows)

# ##### Code

# In[ ]:


twi_clean[twi_clean.expanded_urls.isnull()] # there are no missing expanded_urls anymore


# ##### Test

# In[ ]:


twi_clean.info()


# #### `twi_arch`: in_reply_status_id is information of tweets that are replies, but they still score dogs and some have pictures

# ##### Define 
# Check that all these have a score and a picture.

# ##### Code

# In[ ]:


twi_clean[twi_clean.in_reply_to_status_id.notnull()]

# all replies have pictures and a score
# they are good. I simply looked at the 
# data below


# ##### Test

# In[ ]:


# replies are acceptable


# ##### `twi_arch`: 'source' is in HTML form, only need source name, no urls.

# ##### Define
# Use regular expression to extract the source name, not the urls

# ##### Code

# In[ ]:


twi_clean['source'] = twi_clean.source.str.extract(">(.*)<", expand=True)


# ##### Test

# In[ ]:


twi_clean.source.value_counts()


# #### `twi_arch`: Names are sometimes not real name but other lower-case words
# - (index: 649) dog with tweet_id: 792913359805018113 is named 'Forrest' not 'a'

# ##### Define
# Replace each occurance of a wrong name for Nan, since these tweets most likely did not include a name. There is an occurance in index 649 where the dog's name is 'Forrest'

# ##### Code

# In[ ]:


x = twi_clean.name.str.extractall("(^[a-z]+$)")
x.xs(0, level='match')
all_wrong_names = x[0].tolist()
wrong_names = list(set(all_wrong_names)) # get only the unique values
wrong_names


# In[ ]:


def a_to_none(s):
    if s in wrong_names:
        return nan
    else:
        return s
    
twi_clean['name'] = twi_clean.name.map(a_to_none)
twi_clean.loc[649, 'name'] = 'Forrest'  # 'Forrest' with double 'r'


# ##### Test

# In[ ]:


twi_clean.name.str.extractall("(^[a-z]+$)")


# In[ ]:


twi_clean[twi_clean.name == 'Forrest']


# #### `twi_arch`: erronous data types (e.g. tweet_id, reply info, retweet info, timestamps, rating_numerator and denominator, source)

# ##### Define
# Change variables to appropitate data types

# ##### Code

# In[ ]:


# To category
twi_clean.source = twi_clean.source.astype('category')

# To datetime
twi_clean.timestamp = pd.to_datetime(twi_clean.timestamp)
twi_clean.retweeted_status_timestamp = pd.to_datetime(twi_clean.retweeted_status_timestamp)  # even though I am droping this column

# To string
twi_clean.tweet_id = twi_clean.tweet_id.astype(str)
twi_clean.in_reply_to_status_id = twi_clean.in_reply_to_status_id.astype(str) # even though I am droping this column
twi_clean.in_reply_to_user_id = twi_clean.in_reply_to_user_id.astype(str) # even though I am droping this column
twi_clean.retweeted_status_id = twi_clean.retweeted_status_id.astype(str) # even though I am droping this column
twi_clean.retweeted_status_user_id = twi_clean.retweeted_status_user_id.astype(str)  # even though I am droping this column

# To float since I am going to get correct numerators which can be floats
twi_clean.rating_numerator = twi_clean.rating_numerator.astype(float)
twi_clean.rating_denominator = twi_clean.rating_denominator.astype(float)


# ##### Test

# In[ ]:


twi_clean.info()


# #### `twi_arch`: source text rating and rating_numerator and rating_denominator might not match

# ##### Define
# Use regex to get the correct numerator and denominators from text

# ##### Code

# In[ ]:


c = twi_clean.text.str.extractall("([0-9]+[.][0-9]+\/\d+)")
d = c.xs(0, level='match')
d.loc[:,'rating_numerator'], d.loc[:,'rating_denominator'] = d[0].str.split('/', 1).str
d.rating_numerator = d.rating_numerator.astype(float)
d.rating_denominator = d.rating_denominator.astype(float)
d = d.drop(0, axis=1)
d


# Need to change the above ratings

# In[ ]:


twi_clean.update(d)
index_rating_decimal = [45, 695, 763, 1712]  # updated ratings with decimal places


# These tweets have at least two ratings or similar to a rating.
# 
# index: first,second or average (XXX indicates that I won't need to change them)
# - 766 second
# - 1007 average
# - 1068 second
# - 1165 second
# - 1202 second
# - 1222 average
# - 1359 first XXX
# - 1465 first XXX
# - 1508 average
# - 1525 first XXX
# - 1538 average
# - 1662 second
# - 1795 average
# - 1832 first XXX
# - 1897 first XXX
# - 1901 second
# - 1970 first XXX
# - 2010 first XXX
# - 2064 first XXX
# - 2113 average
# - 2177 average
# - 2216 first XXX
# - 2263 average
# - 2272 first XXX
# - 2306 average but it's both 10/10 XXX
# - 2335 second

# In[ ]:


a = twi_clean.text.str.extractall("(\d+\/\d+)")
f = a.xs(1, level='match')
f.loc[:,'rating_numerator'], f.loc[:,'rating_denominator'] = f[0].str.split('/', 1).str
f.rating_numerator = f.rating_numerator.astype(float)
f.rating_denominator = f.rating_denominator.astype(float)
f = f.drop(0, axis=1)

g = a.xs(0, level='match')
g.loc[:,'rating_numerator'], g.loc[:,'rating_denominator'] = g[0].str.split('/', 1).str
g.rating_numerator = g.rating_numerator.astype(float)
g.rating_denominator = g.rating_denominator.astype(float)
g = g.drop(0, axis=1)


# In[ ]:


to_drop = [1359, 1465, 1525, 1832, 1897, 1970, 2010, 2064, 2216, 2272, 2306] # these will stay as they are in twi_clean
f = f.drop(to_drop) 


# In[ ]:


average_list = [1007, 1222, 1508, 1538, 1795, 2113, 2177, 2263]
for index in average_list:  # average values since there were two dogs and two scores
    n1 = f.loc[index, 'rating_numerator']
    n2 = g.loc[index, 'rating_numerator']
    f.loc[index, 'rating_numerator'] = (n1 + n2) / 2


# In[ ]:


twi_clean.update(f)
# updated ratings that had two ratings (average or just the second one was correct)


# ##### Test

# In[ ]:


twi_clean.loc[average_list]


# In[ ]:


twi_clean.loc[index_rating_decimal]


# #### `twi_arch`: tweet with rating_denominator of 7 is not a tweet of an actual rating

# ##### Define
# Delete this tweet from the data

# ##### Code

# In[ ]:


not_rating = twi_clean.index[twi_clean.rating_denominator == 7.0].tolist() # not a rating, need to delete tweet
twi_clean = twi_clean.drop(not_rating)  # it's a dog that's sick and needs help raising money (I think...)


# ##### Test

# In[ ]:


twi_clean[twi_clean.rating_denominator == 7.0]


# #### `twi_arch`: two outliers with rating numerator of 420 and 1776

# ##### Define
# Remove these two outliers

# ##### Code

# In[ ]:


not_rating = twi_clean.index[twi_clean.rating_numerator == 420.0].tolist()
twi_clean = twi_clean.drop(not_rating)  # Snoop Dog...

not_rating = twi_clean.index[twi_clean.rating_numerator == 1776.0].tolist()  
twi_clean = twi_clean.drop(not_rating)  # Dog celebrating July 4th


# ##### Test

# In[ ]:


twi_clean[twi_clean.rating_numerator == 420.0]


# In[ ]:


twi_clean[twi_clean.rating_numerator == 1776.0]


# #### `predict`: for p1, p2, and p3 the breed of the dog has _ instead of white space between words; all should be lower case for consistency

# ##### Define
# raplce _ for a white space and lower case all leters

# ##### Code

# In[ ]:


pre_clean['p1'] = pre_clean.p1.str.replace('_', ' ').str.lower()
pre_clean['p2'] = pre_clean.p2.str.replace('_', ' ').str.lower()
pre_clean['p3'] = pre_clean.p3.str.replace('_', ' ').str.lower()


# ##### Test

# In[ ]:


pre_clean.sample(10)


# #### `predict`: tweet_id is not integer type

# ##### Define
# Change tweet_id's data type to str 

# ##### Code

# In[ ]:


pre_clean.tweet_id = pre_clean.tweet_id.astype(str)


# ##### Test

# In[ ]:


pre_clean.info()


# #### `tweet_data`: 'id' variable should be labaled the same as for `twi_arch` table, 'tweet_id'

# ##### Define
# Change 'id' to 'tweet_id'

# ##### Code

# In[ ]:


tweet_clean.rename(columns={'id':'tweet_id'}, inplace=True)


# ##### Test

# In[ ]:


tweet_clean.info()


# #### `tweet_data`: tweet_id is not integer type

# ##### Define
# Change tweet_id to str type

# ##### Code

# In[ ]:


tweet_clean.tweet_id = tweet_clean.tweet_id.astype(str)


# ##### Test

# In[ ]:


tweet_clean.info()


# # Tidiness

# #### Drop the values for retweets and for replies. 

# ##### Define
# Drop theses columns by simply using the drop function in pandas

# ##### Code

# In[ ]:


not_needed = ['in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp']
twi_clean = twi_clean.drop(not_needed, axis=1)


# ##### Test

# In[ ]:


list(twi_clean)


# #### One variable (dog_stage) in four columns in `twi_arch` table (doggo, floofer, pupper, and puppo)

# ##### Define 
# Combine these four columns to make one under the value "dog_stage"

# ##### Code

# In[ ]:


twi_clean['dog_stage'] = twi_clean[twi_clean.columns[8:12]].apply(lambda x: ''.join(x), axis=1).str.replace('None','')
twi_clean['dog_stage'] = twi_clean.dog_stage.map(lambda x: nan if x == '' else x)

twi_clean = twi_clean.drop(['doggo', 'floofer', 'pupper', 'puppo'], axis=1)


# In[ ]:


## There is a couple that is two stages of dog
twi_clean['dog_stage'] = twi_clean.dog_stage.str.replace('doggopupper', 'doggo-pupper')
twi_clean['dog_stage'] = twi_clean.dog_stage.str.replace('doggopuppo', 'doggo-puppo')
twi_clean['dog_stage'] = twi_clean.dog_stage.str.replace('doggofloofer', 'doggo-floofer')


# ##### Test

# In[ ]:


twi_clean.dog_stage.value_counts()


# #### There is no score_ratio value

# ##### Define
# Obtain the ratio from the rating numerator and denominator

# ##### Code

# In[ ]:


twi_clean['rating_ratio'] = twi_clean['rating_numerator'] / twi_clean['rating_denominator']


# ##### Test

# In[ ]:


twi_clean.sample(5)


# #### number of images should be in the `predict` table, also delete tweets in `predict` not present in `twi_arch`

# ##### Define
# Place the number of pictures in the `predict` table and drop it from the `twi_arch` table. Delete tweets not in `twi_arch`

# ##### Code

# In[ ]:


pre_clean = pd.merge(pre_clean, twi_clean[['tweet_id','num_of_img']], on='tweet_id', how='inner')
twi_clean = twi_clean.drop('num_of_img', axis=1)


# In[ ]:


# conver num_of_img back to int
pre_clean['num_of_img'] = pre_clean.num_of_img.astype(int)


# In[ ]:


pre_clean.info()


# #### number of likes and retweet count should be in the `twi_arch` table 

# ##### Define
# merge the tables of `twi_clean` and `tweet_clean` to get retweet count and favorite count

# ##### Code

# In[ ]:


twi_clean = pd.merge(twi_clean, tweet_clean, on='tweet_id', how='left')


# ##### Test

# In[ ]:


twi_clean.info()


# #### Combine `predictions` tale and `twi_arch` table

# ##### Define
# merge the tables of `twi_clean` and `tweet_clean` to get retweet count and favorite count

# ##### Code

# In[ ]:


twi_clean = pd.merge(twi_clean, pre_clean, on='tweet_id', how='left')


# ##### Test

# In[ ]:


list(twi_clean)


# # Save our Master Tables

# In[ ]:


# re-arange columns to look better
twi_master = twi_clean[['tweet_id', 'timestamp', 'source', 'expanded_urls', 'text', 'name', 'dog_stage',
                        'rating_numerator', 'rating_denominator', 'rating_ratio', 'retweet_count',
                        'favorite_count', 'num_of_img', 'img_num', 'jpg_url', 'p1', 'p1_conf', 'p1_dog',
                        'p2', 'p2_conf', 'p2_dog', 'p3', 'p3_conf', 'p3_dog']]


# In[ ]:


twi_master.to_csv('twitter_archive_master.csv', index=False)


# # Analyze

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# In[ ]:


twi_master.info()


# In[ ]:


twi_master.describe()


# In[ ]:


twi_master.rating_ratio.value_counts()


# An insight comes from the rating for each tweet. Since a couple of tweets do not have a rating denominator of 10, either because there is more than one dog in the picture and 'WeRateDogs' added all the ratings together, I created a new variable to obtain the ratio between the rating numerator and the rating denominator. The rating ratio has a mean value of 1.05 and a median on 1.1, which means that on average more dogs obtain a higher rating numerator than denominator. This is common for the Twitter account 'WeRateDogs', since they usually give scores such as 11/10 or 13/10. 

# In[ ]:


twi_master.p1_dog.value_counts()


# In[ ]:


1428 / (1428+489)


# The confidence percent for the number one prediction from a picture has a flat distribution, with a mean of 60% and a median of 60%. The histogram below shows the confidence percent for all three predictions. As we can see, the histogram for the number one prediction is pretty flat until we reach confidence levels of about 95% and more. From this histogram it is clear that the number one prediction is considerably more accurate than the other two predictions. Noting that from the number one prediction only 74% is actually a dog breed, I can think of two possibilities. Either the dog in the picture was wearing a costume or the dog simply looked like another thing.

# In[ ]:


n, bins, patches = plt.hist(twi_master.p1_conf, 30, alpha=0.75, label='# 1')
n, bins, patches = plt.hist(twi_master.p2_conf, 20, alpha=0.75, label='# 2')
n, bins, patches = plt.hist(twi_master.p3_conf, 14, alpha=0.75, label='# 3')
plt.legend(title='Prediction')
plt.xlabel("Percent (%)")
plt.ylabel("Count")

# plt.savefig("p_conf_hist.pdf")


# In[ ]:


count = list(twi_master.p1_dog.value_counts())
x_pos = np.arange(2)
x_labels = ['Dog Breed', 'Other']

plt.bar(x_pos, count, align='center', alpha=0.5)
plt.xticks(x_pos, x_labels)
plt.title("Number 1 Image Prediction")
plt.ylabel("Count")

# plt.savefig("p1_dog_bar.pdf")


# In[ ]:


plt.plot(twi_master.rating_ratio, twi_master.favorite_count, 'k.', alpha=1/20)
plt.ylabel('Favorite Count')
plt.xlabel('Rating Ratio')


# In[ ]:


plt.plot(twi_master.favorite_count, twi_master.retweet_count, 'b.', alpha=1/20)
plt.xlim([-2000, 60000])
plt.ylim([-1000, 17500])
plt.xlabel('Favorite Count')
plt.ylabel('Retweet Count')

slope, intercept, r_value, p_value, std_err = stats.linregress(twi_master.favorite_count, twi_master.retweet_count)
print("The correlation coefficient between favorite and retweet count is {:.2f}".format(r_value))

# plt.savefig("fav_ret.pdf")


# Another insight I found while analyzing the data, was that the favorite count was very much correlated to the retweet count. Looking at the figure above, we can see that as the favorite count increases, the retweet count increases as well. It is also good to note that tweets, on average get more favorites than are being retweeted. These two variables have a positive correlation coefficient of 0.94.
