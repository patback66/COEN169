li = [[1,1,1],[0,0,0]]
for idx, item in enumerate(li):
    print item
    li[idx] = 'foo'

print li



# Cosine similarity = dot(d1,d2)/(|d1|*|d2|)
def cosine_sim_old(user_id, movie_id):
    """
        @fucntion cosine_sime
        @param user_id: the int id for the current user
        @param movie_id: int id of the movie that needs a predicted rating
        Calculates a rating prediction using the weighted averages of the
        k(num_similar) users to the given user
    """
    global TRAIN_RECS
    global USER_RATINGS
    num_similar = 200 # use this many similar users
    rating = 0
    # rel_user[user][0] = cosine_similarity value
    relevant_users = []#[[0 for rating in range(REC_COL_MOVIES+1)] for user in range(num_similar)]
    #initialize movie ratings for calculating similarity later
    cur_user = []
    test_movies = []

    #calculate the average for current user -> user_id
    #also build array for the user recs for cosine sim later
    rating_count = 0
    cur_user_average = 0.0
    for rec in USER_RATINGS:
        if rec[0] == user_id and rec[2] != 0 and rating_count < RATING_MAX: #for our user
            cur_user_average += rec[2]
            #update the movie rating for the current user
            #cur_user[rec[1]] = rec[2]
            cur_user.append(rec[2]) # save movie ratings for user
            rating_count += 1
            test_movies.append(rec[1]) # save the relevant movie id
    cur_user_average = float(cur_user_average)/float(rating_count)

    #calculate the k most similar users
    for index, user in enumerate(TRAIN_RECS):
        #user has string of ratings
        #compute cosine similarity
        #only look at co-rated movies
        rel_user = []
        cur_movie_rating = TRAIN_RECS[index][movie_id - 1]
        for item in test_movies: #get the relevant movie ratings for each other user
            rel_user.append(TRAIN_RECS[index][item - 1])
            #movie id's 1-1000 -> -1 for index

        w_a_u = cosine_calc(cur_user, rel_user)

        #compare against other similar users
        #keep only the k(num_similar) most relevant
        if len(relevant_users) < num_similar:
            # [weight, movie ratings, movie_rating_needed]
            relevant_users.append(([w_a_u] + rel_user + [cur_movie_rating]))
        else:
            saved_index = -1
            for idx in range(len(relevant_users)):
                if w_a_u > relevant_users[idx][0]: #compare to other cosine sim
                    saved_index = idx

            #save the more similar user
            if saved_index != -1:
                relevant_users[saved_index] = ([w_a_u] + rel_user + [cur_movie_rating])

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
        #print "Weight: " + str(rec[0]) + " user id: " + str(index+1)

        #for idx in range(1, len(rec)):
        numerator += (rec[0] * rec[-1])
        denominator += rec[0]

    #round the final result for a nice number
    if denominator != 0:
        rating = round(float(numerator) / float(denominator))


    if rating < 1 and rating > 0:
        rating = 1
    elif rating == 0 or rating == 0.0:
        #not enough hits to calc a rating
        #use the current user's average instead
        rating = round(cur_user_average)
    else:
        rating = round(rating)

    rating = int(rating) # don't want the float
    return rating
