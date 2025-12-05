from typing import List, Dict, TypedDict
import numpy as np
from scipy.optimize import curve_fit


class School(TypedDict):
    name: str
    acceptance_rate: float
    SAT_quartiles: List[int]
    applying: bool
    desireability: int


class Student(TypedDict):
    sat_score: int
    gpa: float
    gpa_percentile: float
    application_scores: Dict[str, float]
    application_score_history: Dict[str, List[Dict[str, float]]]


def school_reward(
    admitted_schools: List[str], schools_data: List[School], l: float = 0.1
) -> float:
    """
    returns a numerical reward based on the list of admitted schools
    l controls the diminishing returns factor for multiple schools
    """
    admitted_schools_data: List[School] = []
    for school_name in admitted_schools:
        for school in schools_data:
            if school["name"] == school_name:
                admitted_schools_data.append(school)
                break

    admitted_schools_data.sort(key=lambda x: x["desireability"], reverse=True)

    r = (
        admitted_schools_data[0]["desireability"]
        if len(admitted_schools_data) > 0
        else 0
    )
    for school in admitted_schools_data[1:]:
        r += school["desireability"] * l

    return r


def expected_essay_improvement(essay_score_history: List[Dict[str, float]]) -> float:
    """
    returns the expected improvement in essay score based on historical data
    essay_score_history is a list of dictionaries with 'score' and 'hours' keys
    `hours` is total hours spent on the essay at that point
    Fits a log curve to the data and returns projected score after 2 more hours
    """
    # Handle edge case: one data point or less
    if len(essay_score_history) <= 1:
        if len(essay_score_history) == 0:
            return 800  # Default starting score
        # If just one datapoint, assume linear improvement with slope of 7
        current_score: float = essay_score_history[0]["score"]
        return current_score + 7 * 2

    hours: np.ndarray = np.array([point["hours"] for point in essay_score_history])
    scores: np.ndarray = np.array([point["score"] for point in essay_score_history])

    def log_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.log(x + b) + c

    initial_guess: List[float] = [10.0, 1.0, float(np.mean(scores))]
    params, _ = curve_fit(log_curve, hours, scores, p0=initial_guess)

    current_hours: float = float(np.max(hours))
    future_hours: float = current_hours + 2.0

    projected_score: float = float(log_curve(future_hours, *params))
    return projected_score
