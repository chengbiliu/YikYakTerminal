# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 09:09:26 2016

@author: Golfbrother
"""

import tweepy

auth = tweepy.OAuthHandler(mxD2a4gjS6e32bb5vkZXUGMPi, EkxfHUTKmdmLJlIsTnmgNkRWrQ5U51H8J5powZHsqyQlaKPPKu)
auth.set_access_token(3028938291-zcwuImuaCSfQSlrQ7FXVWPqVysVvyS4CPdUtpgS, 6unlb5UNRUNcn5pyyO0KegVyhfTSLJryjHKsFixJMOGPJ)

api = tweepy.API(auth)
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text
#    
#places = api.geo_search(query="USA", granularity="country")
#place_id = places[0].id
#
#tweets = api.search(q="place:%s" % place_id)
#for tweet in tweets:
#    print (tweet.text + " | " + tweet.place.name if tweet.place else "Undefined place")