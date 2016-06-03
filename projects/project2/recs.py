#!/usr/bin python
"""
    @author Matthew Koken <mkoken@scu.edu>
    @file recs.py
    This file takes tab delimited txt files for users and their movie ratings.
    Based on the users and ratings, recommendations are given.
"""
#pin 3224592822747566
import csv
import math
import sys as Sys
import time
import numpy

# Globals here
# Could go in main + pass along, but this is easier/cleaner for now
REC_ROW_USERS = 200
REC_COL_MOVIES = 1000
RATING_MAX = 5
TRAIN_RECS = [[0 for x in range(REC_COL_MOVIES)] for y in range(REC_ROW_USERS)]
USER_RATINGS = [] #U M R, U = [0][x], M = [1][x], R = [2][x]
PREDICTED_RATINGS = []
DEBUG = 0 #disable progress bar

###################
#Class Definitions#
###################

class Algorithms:
    """
        @class Algorithms
        Holds the possible algorithm selections to be used.
    """
    pearson, pearson_iuf, pearson_case, cosine_sim, item_cos, custom = range(6)

class User:
    """
        @class Relevant_User
        Holds movie ratings for co-rated movies and similarity weights.
    """
    def calc_avg(self, ratings):
        if len(ratings) > 0:
            if self.needed_rating != 0:
                return float(sum(ratings) + self.needed_rating) / float(len(ratings) + 1)
            else:
                return float(sum(ratings)) / float(len(ratings))
        else:
            if self.needed_rating != 0:
                return self.needed_rating
            else:
                return 0

    def calc_std_dev(self, ratings):
        variance = map(lambda x: (x - self.average)**2, ratings)
        avg_variance = (float(sum(variance)) / len(variance))
        return math.sqrt(avg_variance)

    def set_similarity(self, similarity):
        self.similarity = similarity

    def __init__(self, similarity, corated_movie_ratings, needed_rating):
        self.similarity = similarity
        self.corated_movie_ratings = corated_movie_ratings
        self.needed_rating = needed_rating
        self.average = self.calc_avg(corated_movie_ratings)
        self.std_dev = self.calc_std_dev(corated_movie_ratings)

class Movie:
    """
        @class Movie
        Holds movie ratings for user ratings and similarity weights.
    """
    def calc_avg(self, ratings):
        if len(ratings) > 0:
            if self.needed_rating != 0:
                return float(sum(ratings) + self.needed_rating) / float(len(ratings) + 1)
            else:
                return float(sum(ratings)) / float(len(ratings))
        else:
            if self.needed_rating != 0:
                return self.needed_rating
            else:
                return 0

    def __init__(self, m_id, similarity, user_ratings, needed_rating):
        self.m_id = m_id
        self.similarity = similarity
        self.ratings = user_ratings
        self.needed_rating = needed_rating
        self.average = self.calc_avg(self.ratings)

    def set_similarity(self, similarity):
        self.similarity = similarity

    def append_rating(self, new_rating):
        self.ratings.append(new_rating)

    def recalc(self):
        self.average = self.calc_avg(self.ratings)


######################
#Function Definitions#
######################

def clear_console():
    print "\n" * 180


# result5.txt consists a list of predictions in the form:
# (U, M, R),
# where U is the userid, M is the movieid, and R is your predicted rating.
# ID's OFFSET BY 1: [1:200]

def write_file(file_out, data, delim):
    """
        @function write_recs
        @param file_out: string for the file to write
        @param data: array of data to be written
        Writes values of given array to the specified results file
    """
    with open(file_out, "wb") as out_file:
        writer = csv.writer(out_file, delimiter=delim)
        for item in data:
            writer.writerow(item)

def int_wrapper(reader):
    """
        @function int_wrapper
        @param reader: a csv reader
        Maps the string content read by the reader to an int value
    """
    for v in reader:
        yield map(int, v)

def read_in(file_to_read, delim):
    """
        @function read_in
        @param file: the file that will be imported
        Returns an array of integers read in from the tab delimited file.
    """
    data = []
    with open(file_to_read, "rU") as in_file:
        reader = csv.reader(in_file, delimiter=delim)
        reader = int_wrapper(reader)
        data = list(reader)
    return data

def pearson_sim(cur_user=None, test_user=None):
    """
        @Function pearson_sim
        @param cur_user: the current user we want to compare to
        @param test_user: the user we want to compare to
        Calculates the similarity between users using the pearson method
    """
    numerator = 0.0
    len1 = 0.0
    len2 = 0.0
    for index in range(len(cur_user.corated_movie_ratings)):
        if cur_user.corated_movie_ratings[index] != 0 and test_user.corated_movie_ratings[index] != 0:
            diff_1 = (cur_user.corated_movie_ratings[index] - cur_user.average)
            diff_2 = (test_user.corated_movie_ratings[index] - test_user.average)
            numerator += (float(diff_1) * float(diff_2))
            len1 += float(diff_1 * diff_1)
            len2 += float(diff_2 * diff_2)

    denominator = (math.sqrt(len1) * math.sqrt(len2))

    # Don't break, just no similarity if denominator = 0
    if denominator == 0:
        return 0

    #final calc for similarity
    return float(numerator) / float(denominator)


def pearson(user_id=None, movie_id=None):
    """
        @function pearson
        @param user_id: id of the user that we will be predicting for
        @param movie_id: id of the movie that will be given a predicted rating
        Uses pearson correlation to calculate a predicted rating for the movie.
    """
    #calculate the user's standard deviation
    #cur_user_std_dev = 0
    #for rating in cur_user:
    #    cur_user_std_dev += (rating - cur_user_average) * (rating - cur_user_average)
    #cur_user_std_dev = math.sqrt(cur_user_std_dev)
    #global TRAIN_RECS
    #global USER_RATINGS
    #global PREDICTED_RATINGS
    #global RATING_MAX
    num_similar = 30
    rating = 0
    relevant_users = []
    cur_user_ratings = []
    cur_user_rated_movies = []

    #calculate average for the current user
    num_rated_movies = 0
    cur_user_average = 0.0
    for recs in USER_RATINGS:
        # [U, M, R] -> [0, 1, 2]
        if recs[0] == user_id and recs[2] != 0 and num_rated_movies < RATING_MAX:
            cur_user_average += recs[2]
            cur_user_ratings.append(recs[2])
            num_rated_movies += 1
            cur_user_rated_movies.append(recs[1])
    cur_user_average = float(cur_user_average)/float(num_rated_movies)
    cur_user = User(0, cur_user_ratings, 0)

    #find the most similar users
    for index, user in enumerate(TRAIN_RECS):
        sim_user_ratings = []

        #get ratings that other user has also rated
        for movie in cur_user_rated_movies:
            sim_user_ratings.append(TRAIN_RECS[index][movie - 1])

        #sim user has a similarity weight and a rating for movie_id
        needed_rating = TRAIN_RECS[index][movie_id - 1]
        sim_user = User(0, sim_user_ratings, needed_rating)

        #caclulate similarity
        w_a_u = pearson_sim(cur_user, sim_user)
        sim_user.set_similarity(w_a_u)
        #keep only the k most relevant users
        if len(relevant_users) < num_similar:
            relevant_users.append(sim_user)
        else:
            saved_index = -1
            for idx in range(len(relevant_users)):
                if sim_user.similarity > relevant_users[idx].similarity:
                    saved_index = idx

            if saved_index !=-1:
                relevant_users[saved_index] = sim_user

    #have the most similar users, now calculate
    numerator = 0.0
    denominator = 0.0
    for user in relevant_users:
        #user: [w_a_u, r_a_i]
        w_a_u = user.similarity
        r_a_i = user.needed_rating
        numerator += (float(w_a_u) * float(r_a_i - user.average))
        denominator += abs(float(w_a_u))

    if denominator != 0:
        #rounding too early here?
        rating = cur_user.average + (float(numerator)/float(denominator))
        rating = round(rating)

    #default to the user's average rating
    if rating == 0:
        rating = round(cur_user.average)

    if rating > 5: # don't exceed the max rating
        rating = 5

    #cleanup
    del relevant_users[:]
    del cur_user_ratings[:]
    del cur_user_rated_movies[:]

    return int(rating)

def get_num_ratings(movie_id, cur_user_rated):
    """
        @function get_num_ratings
        @param movie_id: the id for a movie. [1,1000]
        Returns the number of users that have rated the movie.
    """
    num_ratings = cur_user_rated
    for user in TRAIN_RECS:
        if user[movie_id - 1] != 0:
            num_ratings += 1
    return num_ratings

def get_iuf(movie_id, cur_user_rated):
    """
        @function get_iuf
        @param movie_id: the id for a movie
        Returns the iuf of a movie
    """
    #IUF(j) = log(m/m_j)
    #m = number of users
    #m_j = number of users that rated movie j
    m = 201 #the number of users
    m_j = get_num_ratings(movie_id, cur_user_rated)
    if m_j != 0:
        iuf = math.log((float(m)/float(m_j)), 2)
    else:
        iuf = 0
    return iuf

def pearson_iuf_sim(cur_user=None, test_user=None):
    """
        @Function pearson_sim
        @param cur_user: the current user we want to compare to
        @param test_user: the user we want to compare to
        Calculates the similarity between users using the pearson method
    """
    numerator = 0.0
    len1 = 0.0
    len2 = 0.0
    for index in range(len(cur_user.corated_movie_ratings)):
        if cur_user.corated_movie_ratings[index] != 0 and test_user.corated_movie_ratings[index] != 0:
            iuf1 = get_iuf(cur_user.corated_movie_ratings[index], 1)
            iuf2 = get_iuf(test_user.corated_movie_ratings[index], 1)
            diff_1 = (iuf1 * cur_user.corated_movie_ratings[index] - cur_user.average)
            diff_2 = (iuf2 * test_user.corated_movie_ratings[index] - test_user.average)
            numerator += (float(diff_1) * float(diff_2))
            len1 += float(diff_1 * diff_1)
            len2 += float(diff_2 * diff_2)

    denominator = (math.sqrt(len1) * math.sqrt(len2))

    # Don't break, just no similarity if denominator = 0
    if denominator == 0:
        return 0

    #final calc for similarity
    return float(numerator) / float(denominator)


def pearson_iuf(user_id=None, movie_id=None):
    """
        @function pearson_iuf
        @param user_id: the id of the user that needs a movie rating prediction
        @param movie_id: the movie id that the user needs a rating for
        Uses the pearson method to predict user ratings, with the addition
        of IUF modification.
    """
    #calculate the user's standard deviation
    #cur_user_std_dev = 0
    #for rating in cur_user:
    #    cur_user_std_dev += (rating - cur_user_average) * (rating - cur_user_average)
    #cur_user_std_dev = math.sqrt(cur_user_std_dev)
    #global TRAIN_RECS
    #global USER_RATINGS
    #global PREDICTED_RATINGS
    #global RATING_MAX
    num_similar = 50
    rating = 0
    relevant_users = []
    cur_user_ratings = []
    cur_user_rated_movies = []

    #calculate average for the current user
    num_rated_movies = 0
    #cur_user_average = 0.0
    for recs in USER_RATINGS:
        # [U, M, R] -> [0, 1, 2]
        if recs[0] == user_id and recs[2] != 0 and num_rated_movies < RATING_MAX:
            #cur_user_average += recs[2]
            #iuf = get_iuf(recs[1], 1) #the user has rated the movie
            cur_user_ratings.append(recs[2])
            num_rated_movies += 1
            cur_user_rated_movies.append(recs[1])
    #cur_user_average = float(cur_user_average)/float(num_rated_movies)
    cur_user = User(0, cur_user_ratings, 0)

    #find the most similar users
    for index, user in enumerate(TRAIN_RECS):
        sim_user_ratings = []

        #get ratings that other user has also rated
        for movie in cur_user_rated_movies:
            #IUF(j) = log(m/m_j)
            #m = number of users
            #m_j = number of users that rated movie j
            #iuf = get_iuf(movie - 1, 1) #the user has rated the movie
            user_rating = TRAIN_RECS[index][movie - 1]
            sim_user_ratings.append(user_rating)

        #sim user has a similarity weight and a rating for movie_id
        #iuf = get_iuf(movie_id - 1, 0) #the user has not rated - needs a prediction
        needed_rating = TRAIN_RECS[index][movie_id - 1]


        sim_user = User(0, sim_user_ratings, needed_rating)

        #caclulate similarity
        w_a_u = pearson_iuf_sim(cur_user, sim_user)
        sim_user.set_similarity(w_a_u)
        #keep only the k most relevant users
        if len(relevant_users) < num_similar:
            relevant_users.append(sim_user)
        else:
            saved_index = -1
            for idx in range(len(relevant_users)):
                if sim_user.similarity > relevant_users[idx].similarity:
                    saved_index = idx

            if saved_index !=-1:
                relevant_users[saved_index] = sim_user

    #have the most similar users, now calculate
    numerator = 0.0
    denominator = 0.0
    for user in relevant_users:
        #user: [w_a_u, r_a_i]
        w_a_u = user.similarity
        r_a_i = user.needed_rating
        numerator += (float(w_a_u) * float((r_a_i - user.average)))
        denominator += abs(float(w_a_u))

    if denominator != 0:
        #rounding too early here?
        rating = cur_user.average + (float(numerator)/float(denominator))
        rating = round(rating)

    #default to the user's average rating
    if rating == 0:
        rating = round(cur_user.average)

    if rating > 5: # don't exceed the max rating
        rating = 5

    #cleanup
    del relevant_users[:]
    del cur_user_ratings[:]
    del cur_user_rated_movies[:]

    return int(rating)

def pearson_case(user_id=None, movie_id=None):
    """
        @function pearson_case
        @param user_id: the id of the active user
        @param movie_id: the id of the movie for which the active user needs a prediction
        Pearson using case amplification.
    """
    #global TRAIN_RECS
    #global USER_RATINGS
    #global PREDICTED_RATINGS
    #global RATING_MAX
    rho = 3.5
    num_similar = 30
    rating = 0
    relevant_users = []
    cur_user_ratings = []
    cur_user_rated_movies = []

    #calculate average for the current user
    num_rated_movies = 0
    #cur_user_average = 0.0
    for recs in USER_RATINGS:
        # [U, M, R] -> [0, 1, 2]
        if recs[0] == user_id and recs[2] != 0 and num_rated_movies < RATING_MAX:
            #cur_user_average += recs[2]
            cur_user_ratings.append(recs[2])
            num_rated_movies += 1
            cur_user_rated_movies.append(recs[1])
    #cur_user_average = float(cur_user_average)/float(num_rated_movies)
    cur_user = User(0, cur_user_ratings, 0)

    #find the most similar users
    for index, user in enumerate(TRAIN_RECS):
        sim_user_ratings = []

        #get ratings that other user has also rated
        for movie in cur_user_rated_movies:
            sim_user_ratings.append(TRAIN_RECS[index][movie - 1])

        #sim user has a similarity weight and a rating for movie_id
        needed_rating = TRAIN_RECS[index][movie_id - 1]
        sim_user = User(0, sim_user_ratings, needed_rating)

        #caclulate similarity
        w_a_u = pearson_sim(cur_user, sim_user)
        w_amplified = w_a_u * math.pow(abs(w_a_u), rho - 1)
        sim_user.set_similarity(w_amplified)
        #keep only the k most relevant users
        if len(relevant_users) < num_similar:
            relevant_users.append(sim_user)
        else:
            saved_index = -1
            for idx in range(len(relevant_users)):
                if sim_user.similarity > relevant_users[idx].similarity:
                    saved_index = idx

            if saved_index !=-1:
                relevant_users[saved_index] = sim_user

    #have the most similar users, now calculate
    numerator = 0.0
    denominator = 0.0
    for user in relevant_users:
        #user: [w_a_u, r_a_i]
        w_a_u = user.similarity
        r_a_i = user.needed_rating
        numerator += (float(w_a_u) * float(r_a_i - user.average))
        denominator += abs(float(w_a_u))

    if denominator != 0:
        #rounding too early here?
        rating = cur_user.average + (float(numerator)/float(denominator))
        rating = round(rating)

    #default to the user's average rating
    if rating == 0:
        rating = round(cur_user.average)

    if rating > 5: # don't exceed the max rating
        rating = 5

    #cleanup
    del relevant_users[:]
    del cur_user_ratings[:]
    del cur_user_rated_movies[:]

    return int(rating)


def cosine_calc(user1=None, user2=None):
    """
        @function cosine_calc
        @param user1: a list of user movie ratings
        @param user2: a list of user movie ratings
        Calculates the cosine similarity between lists of user ratings as long
        as both users have rated the same movie.
    """
    # cosine sim = AdotB/(len(A) * len(B))
    # dot product = sum of multiplications, but only for shared ratings:
    # A[0] * B[0] + A[1] * B[1] + ... + A[n-1] * B[n-1]
    #dot_product = sum([user1[i]*user2[i] for i in range(len(user2))])
    dot_product = 0.0
    len_1 = 0.0
    len_2 = 0.0
    #using adjusted cosine
    for idx in range(len(user2.corated_movie_ratings)):
        #if both have provided ratings for the same, then this is a valid point
        if user1.corated_movie_ratings[idx] != 0 and user2.corated_movie_ratings[idx] != 0:
            diff1 = user1.corated_movie_ratings[idx] - user1.average
            diff2 = user2.corated_movie_ratings[idx] - user2.average
            dot_product += (diff1 * diff2)
            len_1 += (diff1 * diff1)
            len_2 += (diff2 * diff2)
    # length of vector = sqrt(A[0]*A[0] + A[1]*A[1] + ... + A[n]*A[n])
    len_1 = math.sqrt(float(len_1))
    len_2 = math.sqrt(float(len_2))

    #vectors of length 0 break, aren't relevant
    if len_1 == 0 or len_2 == 0:
        return 0

    return float(dot_product) / float((len_1 * len_2))

#cosine similarity
def cosine_sim(user_id=None, movie_id=None):
    """
        @function cosine_sim
        @param user_id: the id of the user in USER_RATINGS that we are predicting for
        @param movie_id the id of the movie we are predicting for
        Uses cosine similarity to calculate the weight for predicting a movie rating.
    """
    #global TRAIN_RECS
    #global USER_RATINGS
    #global PREDICTED_RATINGS
    #global RATING_MAX
    num_similar = 30
    rating = 0
    relevant_users = []
    cur_user_ratings = []
    cur_user_rated_movies = []

    #calculate average for the current user
    num_rated_movies = 0
    cur_user_average = 0.0
    for recs in USER_RATINGS:
        # [U, M, R] -> [0, 1, 2]
        if recs[0] == user_id and recs[2] != 0 and num_rated_movies < RATING_MAX:
            cur_user_average += recs[2]
            cur_user_ratings.append(recs[2])
            num_rated_movies += 1
            cur_user_rated_movies.append(recs[1])
    cur_user_average = float(cur_user_average)/float(num_rated_movies)
    cur_user = User(0, cur_user_ratings, 0)

    #find the most similar users
    for index, user in enumerate(TRAIN_RECS):
        sim_user_ratings = []

        #get ratings that other user has also rated
        for movie in cur_user_rated_movies:
            sim_user_ratings.append(TRAIN_RECS[index][movie - 1])

        #sim user has a similarity weight and a rating for movie_id
        needed_rating = TRAIN_RECS[index][movie_id - 1]
        sim_user = User(0, sim_user_ratings, needed_rating)

        #caclulate similarity
        w_a_u = cosine_calc(cur_user, sim_user)
        sim_user.set_similarity(w_a_u)
        #keep only the k most relevant users
        if len(relevant_users) < num_similar:
            relevant_users.append(sim_user)
        else:
            saved_index = -1
            for idx in range(len(relevant_users)):
                if sim_user.similarity > relevant_users[idx].similarity:
                    saved_index = idx

            if saved_index !=-1:
                relevant_users[saved_index] = sim_user

    #have the most similar users, now calculate
    numerator = 0.0
    denominator = 0.0
    for user in relevant_users:
        w_a_u = user.similarity
        r_a_i = user.needed_rating
        numerator += (float(w_a_u) * float(r_a_i))
        denominator += abs(float(w_a_u))

    if denominator != 0:
        #rounding too early here?
        rating = cur_user.average + (float(numerator)/float(denominator))
        rating = round(rating)

    #default to the user's average rating
    if rating == 0:
        rating = round(cur_user.average)

    if rating > 5: # don't exceed the max rating
        rating = 5

    #cleanup
    del relevant_users[:]
    del cur_user_ratings[:]
    del cur_user_rated_movies[:]

    return int(rating)

def item_adjs_cos(movie1=None, movie2=None, r_u_avgs=None, r_a_avg=None):
    """
        @function item_adj_cos
        @param movie1: the movie we want to compare against
        @param movie2: the movie we are comparing too
        @param r_u_avgs: the averages of user ratings in TRAIN_RECS
        Uses adjusted cosine similarity
    """
    numerator = 0.0
    len1 = 0.0
    len2 = 0.0
    #sum((r_u_i - r_u_avg) * (r_u_j - r_u_avg)
    for index in range(len(movie1.ratings)):
        if(movie1.ratings[index]!=0 and movie2.ratings[index]!=0):
            diff1 = movie1.ratings[index] - r_a_avg
            diff2 = movie2.ratings[index] - r_u_avgs[index]
            numerator += diff1 * diff2
            len1 += diff1 * diff1
            len2 += diff2 * diff2
    len1 = math.sqrt(len1)
    len2 = math.sqrt(len2)

    if len1 == 0 or len2 == 0:
        return 0
    return float(numerator) / float(len1 * len2)

def item_cos(user_id=None, movie_id=None):
    """
        @function item_cos
        @param user_id: The id of the user to be predicted for
        @param movie_id: The id of the movie that the user will predict for
        Uses item based comparison with adjusted cosine similarity to predict
        a rating for the user. Compares the users previously rated movies against
        the movie that is to be predicted for.
    """
    #global TRAIN_RECS
    #global USER_RATINGS
    #global PREDICTED_RATINGS
    #global RATING_MAX
    rating = 0
    cur_user_ratings = []
    cur_user_rated_movies = []
    rated_movies = []

    rel_user_ratings_averages = [0 for x in range(200)]

    num_rated_movies = 0
    for recs in USER_RATINGS:
        # [U, M, R] -> [0, 1, 2]
        if recs[0] == user_id and recs[2] != 0 and num_rated_movies < RATING_MAX:
            cur_user_ratings.append(recs[2])
            num_rated_movies += 1
            cur_user_rated_movies.append(recs[1])
            rated_movies.append(Movie(recs[1], 0, [], recs[2]))
    cur_user = User(0, cur_user_ratings, 0)

    needed_movie = Movie(movie_id, 0, [], 0)

    # build ratings lists for the movies
    for user in TRAIN_RECS:
        for idx, movie in enumerate(rated_movies):
            rated_movies[idx].append_rating(user[movie.m_id - 1])
        needed_movie.append_rating(user[movie_id - 1])

    # recalc averages, etc
    needed_movie.recalc()
    for index in range(len(rated_movies)):
        rated_movies[index].recalc()

    #calc user averages
    for index, user in enumerate(TRAIN_RECS):
        avg = numpy.average(user)
        rel_user_ratings_averages[index] = avg


    #find the most similar items
    for index, check_movie in enumerate(rated_movies):

        #caclulate similarity
        w_a_u = item_adjs_cos(needed_movie, check_movie, rel_user_ratings_averages, cur_user.average)
        rated_movies[index].set_similarity(w_a_u)

    #have the most similar users, now calculate
    numerator = 0.0
    denominator = 0.0
    for movie in rated_movies:
        w_a_u = movie.similarity
        r_a_i = movie.needed_rating
        numerator += (float(w_a_u) * float(r_a_i))
        denominator += abs(float(w_a_u))

    if denominator != 0:
        #rounding too early here?
        rating = cur_user.average + (float(numerator)/float(denominator))
        rating = round(rating)

    #default to the user's average rating
    if rating <= 0:
        rating = round(cur_user.average)

    if rating > 5: # don't exceed the max rating
        rating = 5

    return int(rating)

def euclidean_distance(user1, user2):
    """
        @function euclidean_distance
        @param user1: The current user
        @param user2: The user to be compared against
        Calculates the euclidean distance between two vectors of user ratings.
        Similarity = 1/(distance + 1)
    """
    cur_sum = 0
    for idx in range(len(user2.corated_movie_ratings)):
        #if both have provided ratings for the same, then this is a valid point
        if user1.corated_movie_ratings[idx] != 0 and user2.corated_movie_ratings[idx] != 0:
            diff = abs(user1.corated_movie_ratings[idx] - user2.corated_movie_ratings[idx])
            cur_sum += (diff * diff)
    distance = math.sqrt(cur_sum)
    return distance

def euclidean_custom(user_id=None, movie_id=None):
    """
        @function euclidean_custom
        @param user_id: the id of the user to be predicted for.
        @param movie_id: the id of the movie that is ot be receiving a prediction
        Uses euclidean distance to calculate similarity between movies in order
        for predicting ratings.
    """
    #similarity = 1/(d+1)

    #global TRAIN_RECS
    #global USER_RATINGS
    #global PREDICTED_RATINGS
    #global RATING_MAX
    num_similar = 30
    rating = 0
    relevant_users = []
    cur_user_ratings = []
    cur_user_rated_movies = []

    #calculate average for the current user
    num_rated_movies = 0
    cur_user_average = 0.0
    for recs in USER_RATINGS:
        # [U, M, R] -> [0, 1, 2]
        if recs[0] == user_id and recs[2] != 0 and num_rated_movies < RATING_MAX:
            cur_user_average += recs[2]
            cur_user_ratings.append(recs[2])
            num_rated_movies += 1
            cur_user_rated_movies.append(recs[1])
    cur_user_average = float(cur_user_average)/float(num_rated_movies)
    cur_user = User(0, cur_user_ratings, 0)

    #find the most similar users
    for index, user in enumerate(TRAIN_RECS):
        sim_user_ratings = []

        #get ratings that other user has also rated
        for movie in cur_user_rated_movies:
            sim_user_ratings.append(TRAIN_RECS[index][movie - 1])

        #sim user has a similarity weight and a rating for movie_id
        needed_rating = TRAIN_RECS[index][movie_id - 1]
        sim_user = User(0, sim_user_ratings, needed_rating)

        #caclulate similarity
        w_a_u = 1/float(euclidean_distance(cur_user, sim_user) + 1)
        sim_user.set_similarity(w_a_u)
        #keep only the k most relevant users
        if len(relevant_users) < num_similar:
            relevant_users.append(sim_user)
        else:
            saved_index = -1
            for idx in range(len(relevant_users)):
                if sim_user.similarity > relevant_users[idx].similarity:
                    saved_index = idx

            if saved_index !=-1:
                relevant_users[saved_index] = sim_user

    #have the most similar users, now calculate
    numerator = 0.0
    denominator = 0.0
    for user in relevant_users:
        w_a_u = user.similarity
        r_a_i = user.needed_rating
        numerator += (float(w_a_u) * float(r_a_i))
        denominator += abs(float(w_a_u))

    if denominator != 0:
        #rounding too early here?
        rating = cur_user.average + (float(numerator)/float(denominator))
        rating = round(rating)

    #default to the user's average rating
    if rating == 0:
        rating = round(cur_user.average)

    if rating > 5: # don't exceed the max rating
        rating = 5

    #cleanup
    del relevant_users[:]
    del cur_user_ratings[:]
    del cur_user_rated_movies[:]

    return int(rating)

def custom(user_id=None, movie_id=None):
    """
        @function custom
        @param user_id: the id of the user to be predicted for.
        @param movie_id: the id of the movie that is ot be receiving a prediction
        Uses a hybrid of cosine similarity ratings and euclidean_distance_ratings
        to predict the new movie's rating.
    """
    cosine_sim_rating = cosine_sim(user_id, movie_id)
    euclidean_distance_rating = euclidean_custom(user_id, movie_id)
    rating = round((euclidean_distance_rating + cosine_sim_rating) / 2.0)
    #rating = euclidean_distance_rating
    return int(rating)

def algo_driver(algo=None):
    """
        @function algo_driver
        @param algo: Algorithm class to specify which computation to perform
        Main loop for calculations. Loops through all users to find users w/o ratings
        (R = 0) and predicts their ratings using the specified algorithm
    """
    global PREDICTED_RATINGS
    global USER_RATINGS
    start = time.time()
    for index, rec in enumerate(USER_RATINGS):
        #have the user id, movie id, and rating {USERID, MOVIEID, RATING}
        rating = 0

        #If there is no rating (rating = 0), predict using algo
        if rec[2] == 0:
            #calculate the predicted rating
            if algo == Algorithms.cosine_sim:
                rating = cosine_sim(rec[0], rec[1]) # userid, movieid
            elif algo == Algorithms.pearson:
                rating = pearson(rec[0], rec[1]) # userid, movieid
            elif algo == Algorithms.pearson_iuf:
                rating = pearson_iuf(rec[0], rec[1]) # userid, movieid
            elif algo == Algorithms.pearson_case:
                rating = pearson_case(rec[0], rec[1]) # userid, movieid
            elif algo == Algorithms.item_cos:
                rating = item_cos(rec[0], rec[1]) # userid, movieid
            elif algo == Algorithms.custom:
                rating = custom(rec[0], rec[1]) # userid, movieid
            #update with the predicted rating
            #USER_RATINGS[index][2] = rating
            PREDICTED_RATINGS.append(([rec[0]] + [rec[1]] + [rating]))
            #print "Movie being rated: " + str(rec[1])
            #print rating

            if DEBUG != 1:
                end = time.time()
                hours, rem = divmod(end-start, 3600)
                minutes, seconds = divmod(rem, 60)
                elapsed = ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

                #Show progress, but not too often
                filledLength = int(round(30 * index / float(len(USER_RATINGS))))
                percents = round(100.00 * (index / float(len(USER_RATINGS))), 1)
                bar = '#' * filledLength + '-' * (30 - filledLength)
                Sys.stdout.write('%s [%s] %s%s %s\r' % ("Progress", bar, percents, '%', "done. Time elapsed: " + elapsed))
                Sys.stdout.flush()
                if index == len(USER_RATINGS):
                    print "\n\n"

def main():
    """
        @function main
        The main loop, reads in the base train.txt file and gives options for
        next files to import and perform analysis for recommendations.
    """
    global TRAIN_RECS
    global USER_RATINGS
    global RATING_MAX

    TRAIN_RECS = read_in("train.txt", "\t")

    #Driver for importing files
    response = 1
    while response != 0:
        option_text = """Which file would you like to test?
            (1)test5.txt
            (2)test10.txt
            (3)test20.txt
            (0) quit\n> """
        response = input(option_text)

        read_file = ""
        out_file = ""
        #no case switch so use if
        if response == 1:
            read_file = "test5.txt"
            RATING_MAX = 5
            out_file = "result5.txt"
        elif response == 2:
            read_file = "test10.txt"
            RATING_MAX = 10
            out_file = "result10.txt"
        elif response == 3:
            read_file = "test20.txt"
            RATING_MAX = 20
            out_file = "result20.txt"
        elif response == 0: #exit condition
            break
        else:
            print "Invalid option"
            continue #didn't get a valid option, do not proceed, try again

        # got a valid file, now proceed with import and recommendations
        del USER_RATINGS[:]
        del PREDICTED_RATINGS[:]
        USER_RATINGS = read_in(read_file, " ")

        #Driver for selecting math to perform
        math_selection = 1
        while math_selection != 0:
            print "Current file: " + read_file
            algorithm_text = """Which algorithm would you like to use?
            (1) Pearson Correlation
            (2) Pearson Correlation - Inverse User Frequency
            (3) Pearson Correlation - Case Amplification
            (4) Cosine Similarity
            (5) Item based Similarity with Cosine
            (6) Custom Algorithm
            (0) Quit\n>"""
            math_selection = input(algorithm_text)
            print "Calculating..."
            algo = Algorithms.cosine_sim
            if math_selection == 1:
                # Pearson Correlation
                algo = Algorithms.pearson
            elif math_selection == 2:
                # Pearson Correlation - Inverse User Frequency
                algo = Algorithms.pearson_iuf
            elif math_selection == 3:
                # Pearson Correlation - Case Modification
                algo = Algorithms.pearson_case
            elif math_selection == 4:
                # Cosine Similarity
                algo = Algorithms.cosine_sim
            elif math_selection == 5:
                # Item based similarity with cosine
                algo = Algorithms.item_cos
            elif math_selection == 6:
                # Custom algorithm
                algo = Algorithms.custom
            elif math_selection == 0: #exit condition
                break
            else:
                print "Invalid option"
                continue #didn't get a valid option, do not proceed, try again

            algo_driver(algo)
            print "\nDone! saving to: " + out_file
            write_file(out_file, PREDICTED_RATINGS, " ")

################################################################################
"""
    Main Function Call
"""
if __name__ == '__main__':
    main()
