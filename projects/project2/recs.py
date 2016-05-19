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

# Globals here
# Could go in main + pass along, but this is easier/cleaner for now
REC_ROW_USERS = 200
REC_COL_MOVIES = 1000
RATING_MAX = 5
TRAIN_RECS = [[0 for x in range(REC_COL_MOVIES)] for y in range(REC_ROW_USERS)]
USER_RATINGS = [] #U M R, U = [0][x], M = [1][x], R = [2][x]

class Algorithms:
    """
        @class Algorithms
        Holds the possible algorithm selections to be used.
    """
    pearson, pearson_iuf, pearson_case, cosine_sim, item_cos, custom = range(6)

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

def pearson(user_id, movie_id):
    """
        @
    """
    rating = 999999
    print "Pearson"
    #TODO ALL THE MATH
    return rating

def pearson_iuf(user_id, movie_id):
    rating = 0
    print "Pearson IUF"
    #TODO ALL THE MATH
    return rating

def pearson_case(user_id, movie_id):
    print "Pearson Case"
    rating = 0
    #TODO ALL THE MATH
    return rating

def cosine_calc(user1, user2):
    # cosine sim = AdotB/(len(A) * len(B))
    # dot product = sum of multiplications:
    # A[0] * B[0] + A[1] * B[1] + ... + A[n-1] * B[n-1]
    dot_product = sum([user1[i]*user2[i] for i in range(len(user2))])
    # length of vector = sqrt(A[0]*A[0] + A[1]*A[1] + ... + A[n]*A[n])
    len_1 = math.sqrt(float(sum(user1[i] * user1[i] for i in range(len(user1)))))
    len_2 = math.sqrt(float(sum(user2[j] * user2[j] for j in range(len(user2)))))

    #vectors of length 0 break, aren't relevant
    if len_1 == 0 or len_2 == 0:
        return 0

    return float(dot_product / (len_1 * len_2))

# Cosine similarity = dot(d1,d2)/(|d1|*|d2|)
def cosine_sim(user_id, movie_id):
    """
        @fucntion cosine_sime
        @param user_id: the int id for the current user
        @param movie_id: int id of the movie that needs a predicted rating
        Calculates a rating prediction using the weighted averages of the
        k(num_similar) users to the given user
    """
    global TRAIN_RECS
    global USER_RATINGS
    num_similar = 100 # use this many similar users
    rating = 0
    # rel_user[user][0] = cosine_similarity value
    relevant_users = [[0 for rating in range(REC_COL_MOVIES+1)] for user in range(num_similar)]
    #initialize movie ratings for calculating similarity later
    cur_user = []
    test_movies = []

    #calculate the average for current user -> user_id
    #also build array for the user recs for cosine sim later
    rating_count = 0
    cur_user_average = 0.0
    for rec in USER_RATINGS:
        if rec[0] == user_id and rec[2] != 0: #for our user
            cur_user_average += rec[2]
            #update the movie rating for the current user
            #cur_user[rec[1]] = rec[2]
            cur_user.append(rec[2]) # save movie ratings for user
            rating_count += 1
            test_movies.append(rec[1]) # save the relevant movie id
    cur_user_average = round(cur_user_average/rating_count)

    #calculate the user's standard deviation
    cur_user_std_dev = 0
    for rating in cur_user:
        cur_user_std_dev += (rating - cur_user_average) * (rating - cur_user_average)
    cur_user_std_dev = math.sqrt(cur_user_std_dev)

    #calculate the k most similar users
    for index, user in enumerate(TRAIN_RECS):
        #user has string of ratings
        #compute cosine similarity
        #only look at co-rated movies
        rel_user = []
        for item in test_movies: #get the relevant movie ratings
            rel_user.append(TRAIN_RECS[index][item-1])
            #movie id's 1-1000 -> -1 for index
        w_a_u = cosine_calc(cur_user, rel_user)

        #compare against other similar users
        #keep only the k(num_similar) most relevant
        saved_index = -1
        for idx in range(len(relevant_users)):
            if w_a_u > relevant_users[idx][0]: #compare to other cosine sim
                saved_index = idx

        #save the more similar user
        if saved_index != -1:
            relevant_users[saved_index] = [w_a_u] + rel_user


    #now we have the k(num_similar) most similar users
    #calculate the new predicted rating
    #P_a_i = sum(w_a_u * r_u_i)/sum(w_a_u)
    #calculate the numerator
    numerator = 0
    denominator = 0
    for index, rec in enumerate(relevant_users):
        # w_a_u = rec[0]
        # r_u_i = rec[movie_id]
        #weighted average of all
        for rating in rec:
            numerator += rec[0] * rating
            denominator += rec[0]

    #round the final result for a nice number
    if denominator != 0:
        rating = numerator / denominator
    if rating < 1 and rating > 0:
        rating = math.ceil(rating)
    elif rating == 0 or rating == 0.0:
        #not enough hits to calc a rating
        #use the current user's average instead
        rating = cur_user_average
    else:
        rating = round(rating)

    rating = int(rating) # don't want the float
    return rating

def item_cos(user_id, movie_id):
    print "Item Cosine"
    rating = 0
    #TODO ALL THE MATH
    return rating

def custom(user_id, movie_id):
    print "CUSTOM"
    rating = 33
    #TODO ALL THE MATH
    return rating

def algo_driver(algo):
    """
        @function algo_driver
        @param algo: Algorithm class to specify which computation to perform
        Main loop for calculations. Loops through all users to find users w/o ratings
        (R = 0) and predicts their ratings using the specified algorithm
    """
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
            USER_RATINGS[index][2] = rating
            #print rating

def main():
    """
        @function main
        The main loop, reads in the base train.txt file and gives options for
        next files to import and perform analysis for recommendations.
    """
    global TRAIN_RECS
    global USER_RATINGS
    global RATING_MAX
    #with open("train.txt", "rU") as in_file:
    #    reader = csv.reader(in_file, delimiter="\t")
    #    reader = int_wrapper(reader)
    #    TRAIN_RECS = list(reader)

    TRAIN_RECS = read_in("train.txt", "\t")

    #for line in TRAIN_RECS:
    #    pLine = ""
    #    for item in line:
    #        pLine += str(item) + '\t'
    #    print pLine + '\n'

    #Driver for importing files
    response = 1
    while response != 0:
        print "Which file would you like to test?"
        option_text = "(1)test5.txt (2)test10.txt (3)test20.txt or (0) quit\n"
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
        USER_RATINGS = read_in(read_file, " ")

        #Driver for selecting math to perform
        math_selection = 1
        while math_selection != 0:
            print "Current file: " + read_file
            algorithm_text = """Which algorithm would you like to use?
            (1) Pearson Correlation
            (2) Pearson Correlation - Inverse User Frequency
            (3) Pearson Correlation - Case Modfification
            (4) Cosine Similarity
            (5) Item based Similarity with Cosine
            (6) Custom Algorithm
            (0) Quit
            """
            math_selection = input(algorithm_text)
            algo = Algorithms.cosine_sim
            if math_selection == 1:
                # Pearson Correlation
                algo = Algorithms.pearson
            elif math_selection == 2:
                # Pearson Correlation - Inverse User Frequency
                algo = Algorithms.pearson_iuf
            elif math_selection == 3:
                # Pearson Correlation - Case Modification
                algo = Algorithms.pearson_iuf
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
            print "Done! saving to: " + out_file
            write_file(out_file, USER_RATINGS, " ")


#	Algorithms
#	: Pearson Correlation
#	: Pearson Correlation - Inverse User Frequency
#	: Pearson Correlation - Case Modification
#	: Cosine Similarity
#	: Item Based Similarity w/ Cosine
#	: Custom Algorithm

main()
