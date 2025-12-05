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
    # Handle edge cases with limited data
    if len(essay_score_history) <= 1:
        if len(essay_score_history) == 0:
            return 800  # Default starting score
        # If just one datapoint, assume linear improvement with slope of 7
        current_score: float = essay_score_history[0]["score"]
        return current_score + 7 * 2
    if len(essay_score_history) == 2:
        h0, h1 = essay_score_history[0]["hours"], essay_score_history[1]["hours"]
        s0, s1 = essay_score_history[0]["score"], essay_score_history[1]["score"]
        slope = (s1 - s0) / (max(1e-6, h1 - h0))
        return s1 + slope * 2.0

    hours: np.ndarray = np.array(
        [point["hours"] for point in essay_score_history], dtype=float
    )
    scores: np.ndarray = np.array(
        [point["score"] for point in essay_score_history], dtype=float
    )

    # Shift hours so log arguments stay positive
    min_shift: float = max(0.0, 1e-3 - float(np.min(hours)))
    hours_shifted: np.ndarray = hours + min_shift

    def log_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.log(x + b) + c

    # Constrain b to keep x + b > 0; allow more iterations to converge
    lower_bounds: List[float] = [-np.inf, 1e-3, -np.inf]
    upper_bounds: List[float] = [np.inf, np.inf, np.inf]
    initial_guess: List[float] = [10.0, 1.0, float(np.mean(scores))]

    try:
        params, _ = curve_fit(
            log_curve,
            hours_shifted,
            scores,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,
        )
        current_hours: float = float(np.max(hours_shifted))
        future_hours: float = current_hours + 2.0
        projected_score: float = float(log_curve(future_hours, *params))
    except Exception:
        # Fallback: simple linear extrapolation
        total_hours_spent: float = float(hours[-1] - hours[0])
        slope: float = float(scores[-1] - scores[0]) / (total_hours_spent + 1e-6)
        projected_score = float(scores[-1] + slope * 2.0)

    return projected_score
