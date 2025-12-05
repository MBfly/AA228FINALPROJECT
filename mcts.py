from typing import List, Dict, Optional, Tuple
import math
import random
import copy
import time
from models import School, Student, expected_essay_improvement, school_reward
from calculate_college_probability import get_probability


MAX_HOURS_PER_SCHOOL = 2.0
COST_PER_HOUR = -0.5  # Desirability points per hour spent
STOP_ACTION = "STOP"
DEFAULT_TIME_LIMIT = 30.0  # seconds
DEFAULT_EXPLORATION_WEIGHT = 1.41
DEFAULT_EXPLOITATION_WEIGHT = 1.0


def get_total_hours(student: Student, school_name: str) -> float:
    """Get total hours spent on a school's essay"""
    history = student["application_score_history"].get(school_name, [])
    if not history:
        return 0.0
    return max(point["hours"] for point in history)


def available_actions(student: Student, schools_data: List[School]) -> List[str]:
    """
    Returns a list of actions: school names that can be worked on (< 10 hours) + STOP
    """
    actions = []
    for school in schools_data:
        if school["applying"]:
            total_hours = get_total_hours(student, school["name"])
            if total_hours < MAX_HOURS_PER_SCHOOL:
                actions.append(school["name"])
    actions.append(STOP_ACTION)
    return actions


def copy_student(student: Student) -> Student:
    """Deep copy a student"""
    return {
        "sat_score": student["sat_score"],
        "gpa": student["gpa"],
        "gpa_percentile": student["gpa_percentile"],
        "application_scores": copy.deepcopy(student["application_scores"]),
        "application_score_history": copy.deepcopy(
            student["application_score_history"]
        ),
    }


def copy_schools(schools_data: List[School]) -> List[School]:
    """Deep copy schools data"""
    return copy.deepcopy(schools_data)


def apply_action(
    student: Student, schools_data: List[School], action: str
) -> Tuple[Student, List[School], float]:
    """
    Apply an action and return new state + immediate cost
    Returns: (new_student, new_schools_data, hours_spent)
    """
    new_student = copy_student(student)
    new_schools = copy_schools(schools_data)

    if action == STOP_ACTION:
        return new_student, new_schools, 0.0

    hours_spent = 2.0
    history = new_student["application_score_history"].get(action, [])

    new_score = expected_essay_improvement(history)
    current_hours = get_total_hours(new_student, action)
    new_hours = current_hours + hours_spent

    if action not in new_student["application_score_history"]:
        new_student["application_score_history"][action] = []
    new_student["application_score_history"][action].append(
        {"hours": new_hours, "score": new_score}
    )

    new_student["application_scores"][action] = new_score

    return new_student, new_schools, hours_spent


def calculate_expected_reward(
    student: Student, schools_data: List[School], total_hours_spent: float
) -> float:
    """
    Calculate expected reward considering admission probabilities
    Uses Monte Carlo sampling to handle non-linear reward structure
    """
    num_samples = 1000
    total_reward = 0.0

    for _ in range(num_samples):
        admitted_schools = []
        for school in schools_data:
            if school["applying"]:
                school_name = school["name"]
                essay_score = student["application_scores"].get(school_name, 700)

                prob = get_probability(
                    school_name,
                    student["sat_score"],
                    student["gpa_percentile"],
                    essay_score,
                )

                if random.random() < prob:
                    admitted_schools.append(school_name)

        sample_reward = school_reward(admitted_schools, schools_data)
        total_reward += sample_reward

    expected_reward = total_reward / num_samples

    # Subtract time cost
    time_cost = COST_PER_HOUR * total_hours_spent

    return expected_reward + time_cost


class MCTSNode:
    def __init__(
        self,
        student: Student,
        schools_data: List[School],
        total_hours_spent: float = 0.0,
        parent: Optional["MCTSNode"] = None,
        action: Optional[str] = None,
    ):
        self.student: Student = student
        self.schools_data: List[School] = schools_data
        self.total_hours_spent: float = total_hours_spent
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[str] = action
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.untried_actions: List[str] = available_actions(student, schools_data)

    def is_terminal(self) -> bool:
        """Check if node is terminal (no more actions or STOP was taken)"""
        if self.action == STOP_ACTION:
            return True
        return (
            len(available_actions(self.student, self.schools_data)) == 1
        )  # Only STOP left

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0

    def best_child(
        self,
        exploration_weight: float = DEFAULT_EXPLORATION_WEIGHT,
        exploitation_weight: float = DEFAULT_EXPLOITATION_WEIGHT,
    ) -> "MCTSNode":
        """Select best child using UCB1 formula"""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                ucb_value = float("inf")
            else:
                exploitation = exploitation_weight * (child.total_reward / child.visits)
                exploration = exploration_weight * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                ucb_value = exploitation + exploration
            choices_weights.append((child, ucb_value))

        return max(choices_weights, key=lambda x: x[1])[0]

    def expand(self) -> "MCTSNode":
        """Expand node by trying an untried action"""
        action = self.untried_actions.pop()
        new_student, new_schools, hours_spent = apply_action(
            self.student, self.schools_data, action
        )
        child_node = MCTSNode(
            new_student,
            new_schools,
            self.total_hours_spent + hours_spent,
            parent=self,
            action=action,
        )
        self.children.append(child_node)
        print(f"Expanded node with action: {action}")
        return child_node

    def rollout(self) -> float:
        """Simulate random playout from this node"""
        current_student = copy_student(self.student)
        current_schools = copy_schools(self.schools_data)
        current_hours = self.total_hours_spent

        # Random policy until terminal
        while True:
            actions = available_actions(current_student, current_schools)
            if len(actions) == 1 and actions[0] == STOP_ACTION:
                break

            if STOP_ACTION in actions and random.random() < 0.3:
                break

            action = random.choice(
                [a for a in actions if a != STOP_ACTION] or [STOP_ACTION]
            )
            if action == STOP_ACTION:
                break

            current_student, current_schools, hours = apply_action(
                current_student, current_schools, action
            )
            current_hours += hours

        return calculate_expected_reward(
            current_student, current_schools, current_hours
        )

    def backpropagate(self, reward: float) -> None:
        """Backpropagate reward up the tree"""
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


def mcts_search(
    student: Student,
    schools_data: List[School],
    time_limit: float = DEFAULT_TIME_LIMIT,
    exploration_weight: float = DEFAULT_EXPLORATION_WEIGHT,
    exploitation_weight: float = DEFAULT_EXPLOITATION_WEIGHT,
) -> str:
    """
    Run MCTS to find the best action

    Args:
        student: Current student state
        schools_data: List of schools
        time_limit: Maximum time in seconds to run MCTS
        exploration_weight: UCB1 exploration parameter (weight for exploration term)
        exploitation_weight: UCB1 exploitation parameter (weight for exploitation term)

    Returns:
        Best action (school name or STOP)
    """
    root = MCTSNode(student, schools_data)
    start_time = time.time()
    iteration = 0

    # Fully explore first layer (all direct children of root)
    print("Fully exploring first layer...")
    while not root.is_fully_expanded():
        node = root.expand()
        reward = node.rollout()
        node.backpropagate(reward)
        iteration += 1

        if time.time() - start_time >= time_limit:
            print(f"Time limit reached after {iteration} iterations")
            break

    print(
        f"First layer fully explored with {len(root.children)} children after {iteration} iterations"
    )

    print("Continuing MCTS search...")
    while time.time() - start_time < time_limit:
        node = root

        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(exploration_weight, exploitation_weight)

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # Simulation
        reward = node.rollout()

        # Backpropagation
        node.backpropagate(reward)
        iteration += 1

    elapsed_time = time.time() - start_time
    print(f"MCTS completed: {iteration} total iterations in {elapsed_time:.2f}s")

    if not root.children:
        return STOP_ACTION

    best_child = max(
        root.children, key=lambda c: c.total_reward / c.visits if c.visits > 0 else 0
    )

    print("\nFirst layer statistics:")
    for child in sorted(
        root.children,
        key=lambda c: c.total_reward / c.visits if c.visits > 0 else 0,
        reverse=True,
    ):
        avg_reward = child.total_reward / child.visits if child.visits > 0 else 0
        print(f"  {child.action}: visits={child.visits}, avg_reward={avg_reward:.2f}")

    return best_child.action or STOP_ACTION
