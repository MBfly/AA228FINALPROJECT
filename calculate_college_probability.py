from scipy.stats import norm
#Qs: what's the data structure of the school?

#School: name, acceptance rate, 

#SAT score and GPA in percentile, essay score as raw number (gets mapped)

def get_essay_percentile(essay_score):
    essay_score_mean = 1032.45
    essay_score_std_dev = 66.52

    p = norm.cdf(essay_score, loc=1032.45, scale=66.52)

    return p

#Testing get essay percentile:

def get_probability(school, sat_score, gpa_score, essay_score):

    STD_DEV = 10

    essay_score_adjusted = get

    student_total_percentile = 