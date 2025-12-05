from scipy.stats import norm
import json
from statistics import NormalDist
import csv


_colleges = None
_sat_lookup = None
normal = NormalDist()  # standard normal


# School: name, acceptance rate,

# SAT score and GPA in percentile, essay score as raw number (gets mapped)


def get_essay_percentile(essay_score):
    essay_score_mean = 1032.45
    essay_score_std_dev = 66.52

    p = norm.cdf(essay_score, loc=essay_score_mean, scale=essay_score_std_dev)

    return p


def get_sat_percentile(sat_score):  # Assumes SAT is divisible by 10

    global _sat_lookup, _colleges
    if _sat_lookup is None:
        _load_data()
    return _sat_lookup[sat_score] / 100


def get_probability(school, sat_score, gpa_percentile, essay_score):

    global _colleges, _sat_lookup
    if _colleges is None or _sat_lookup is None:
        _load_data()
    sat_percentile = get_sat_percentile(sat_score)

    # weightings:
    w_sat = 0.25
    w_gpa = 0.25
    w_essay = 0.5

    essay_percentile = get_essay_percentile(essay_score)
    # Clamp percentiles to (0,1) to avoid inv_cdf domain errors
    eps = 1e-6
    sat_percentile = min(1.0 - eps, max(eps, sat_percentile))
    gpa_percentile = min(1.0 - eps, max(eps, gpa_percentile))
    essay_percentile = min(1.0 - eps, max(eps, essay_percentile))

    z_sat = normal.inv_cdf(sat_percentile)
    z_gpa = normal.inv_cdf(gpa_percentile)
    z_essay = normal.inv_cdf(essay_percentile)

    z_student = z_sat * w_sat + z_gpa * w_gpa + z_essay * w_essay

    student_total_percentile = normal.cdf(z_student)

    # print(f"Student percentile: {student_total_percentile}")

    # student_total_percentile = sat_score * 0.25 + gpa_score * 0.25 + essay_percentile * 0.5 #TODO distributions don't work like this
    lookup = {c["name"]: c["acceptance_rate"] for c in _colleges}

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


def _load_data():
    """Lazy-load data to avoid import-time side effects."""
    global _colleges, _sat_lookup
    if _colleges is None:
        with open("schools.json", "r") as f:
            _colleges = json.load(f)
    if _sat_lookup is None:
        _sat_lookup = {}
        with open("sat_percentiles.csv", "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = int(row["score"])
                percentile = int(row["percentile"])
                _sat_lookup[score] = percentile


if __name__ == "__main__":
    # Testing probability
    _load_data()
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
