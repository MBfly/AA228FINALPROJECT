import json
from typing import List
from mcts import mcts_search
from models import School, Student


schools_data: List[School] = []


def load_school_data(file_path: str) -> None:
    global schools_data
    with open(file_path, "r") as file:
        schools_data = json.load(file)


def print_schools_list(schools_data: List[School]) -> None:
    for i, school in enumerate(schools_data):
        print(f"{i+1}. {school['name']}")


def __main__():
    load_school_data("schools.json")
    print_schools_list(schools_data)
    student = {
        "sat_score": 1350,
        "gpa": 4.0,
        "gpa_percentile": 0.75,
        "application_scores": {
            "Harvard University": 1130,
            "Columbia University": 1082,
            "Stanford University": 1210,
        },
        "application_score_history": {
            "Harvard University": [
                {"score": 1021, "hours": 2},
                {"score": 1130, "hours": 4},
            ],
            "Columbia University": [
                {"score": 1082, "hours": 2},
            ],
            "Stanford University": [
                {"score": 950, "hours": 2},
                {"score": 1050, "hours": 4},
                {"score": 1210, "hours": 6},
            ],
        },
    }
    best_action = mcts_search(
        student,
        schools_data,
        time_limit=10.0,
        exploration_weight=1.41,
        exploitation_weight=1.0,
    )
    print(f"\nRecommended action: {best_action}")


if __name__ == "__main__":
    __main__()
