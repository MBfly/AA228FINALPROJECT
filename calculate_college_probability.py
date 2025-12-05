from scipy.stats import norm
import json
from statistics import NormalDist
import math
import csv


with open("schools.json", "r") as f:
    colleges = json.load(f)

sat_lookup = {}
with open("sat_percentiles.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        score = int(row["score"])
        percentile = int(row["percentile"])
        sat_lookup[score] = percentile

normal = NormalDist()  # standard normal


# School: name, acceptance rate,

# SAT score and GPA in percentile, essay score as raw number (gets mapped)


def get_essay_percentile(essay_score):
    essay_score_mean = 1032.45
    essay_score_std_dev = 66.52

    p = norm.cdf(essay_score, loc=1032.45, scale=66.52)

    return p


def get_sat_percentile(sat_score):  # Assumes SAT is divisible by 10

    return sat_lookup[sat_score] / 100


def get_probability(school, sat_score, gpa_percentile, essay_score):

    sat_percentile = get_sat_percentile(sat_score)

    # weightings:
    w_sat = 0.25
    w_gpa = 0.25
    w_essay = 0.5

    STD_DEV = 0.02  # standard deviation of 2% of acceptance

    essay_percentile = get_essay_percentile(essay_score)

    z_sat = normal.inv_cdf(sat_percentile)
    z_gpa = normal.inv_cdf(gpa_percentile)
    z_essay = normal.inv_cdf(essay_percentile)

    z_student = z_sat * w_sat + z_gpa * w_gpa + z_essay * w_essay

    student_total_percentile = normal.cdf(z_student)

    # print(f"Student percentile: {student_total_percentile}")

    # student_total_percentile = sat_score * 0.25 + gpa_score * 0.25 + essay_percentile * 0.5 #TODO distributions don't work like this
    lookup = {c["name"]: c["acceptance_rate"] for c in colleges}

    school_acceptance_rate = lookup[school] / 100

    z_school = normal.inv_cdf(1 - school_acceptance_rate)

    # 5% acceptance rate: 0.02 standard dev
    # 50% acceptance rate: 0.1 standard dev
    # x = school acceptance rate
    # m = -0.19/0.45 = -0.422
    # 0.01 = 0.05 * -0.422 + b
    # b = 0.0311

    school_standard_dev = -0.422 * school_acceptance_rate + 0.0311

    admissions_probability = 1 - normal.cdf((z_school - z_student) / 0.5)

    return admissions_probability


# Testing probability

# highly competitive student, highly competitive school
test1 = get_probability("Harvard University", 1550, 0.99, 1150)
print(f"Competitive student at Harvard: {test1}")

# highly competitive student, uncompetitive school
test2 = get_probability("Uncompetitive College", 1550, 0.99, 1150)
print(f"Competitive student at Uncompetitive school: {test2}")

# uncompetitive student, highly competitive school
test3 = get_probability("Harvard University", 1200, 0.2, 900)
print(f"Uncompetitive student at Harvard: {test3}")

# uncompetitive student, uncompetitive school
test4 = get_probability("Uncompetitive College", 1200, 0.2, 900)
print(f"Uncompetitive student at Uncompetitive school: {test4}")
