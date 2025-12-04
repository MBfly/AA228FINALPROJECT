import json

schools_data = []


def load_school_data(file_path):
    global schools_data
    with open(file_path, "r") as file:
        schools_data = json.load(file)


def print_schools_list(schools_data):
    for i, school in enumerate(schools_data):
        print(f"{i+1}. {school["name"]}")


def __main__():
    load_school_data("schools.json")
    print_schools_list(schools_data)


def school_reward(admitted_schools, schools_data, l=0.1):
    """
    returns a numerical reward based on the list of admitted schools
    l controls the diminishing returns factor for multiple schools
    """
    admitted_schools_data = []
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


if __name__ == "__main__":
    __main__()
