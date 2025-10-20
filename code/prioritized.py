import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost

class PrioritizedPlanningSolver(object):
    """Planner that plans agents sequentially using prioritized planning."""

    def __init__(self, my_map, starts, goals):
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.CPU_time = 0

        # compute heuristics for each goal
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        start_time = timer.time()
        result = []
        constraints = []  # accumulated constraints for future agents

        # --------------------------
        # Plan agents one by one
        for i in range(self.num_of_agents):
            # Task 2.4: Calculate upper bound for time horizon
            max_path_len = sum([len(p) for p in result]) + len(self.my_map) * len(self.my_map[0])

            # Find path with current constraints
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i, constraints)
            if path is None or len(path) > max_path_len:
                raise BaseException(f"No solutions for agent {i} within time horizon")

            result.append(path)

            ##############################
            # Task 2.1: Vertex constraints for all future agents
            for t, loc in enumerate(path):
                for j in range(i+1, self.num_of_agents):
                    constraints.append({'agent': j, 'loc': [loc], 'timestep': t})

            # Task 2.2: Edge constraints for all future agents
            for t in range(1, len(path)):
                for j in range(i+1, self.num_of_agents):
                    constraints.append({'agent': j,
                                        'loc': [path[t-1], path[t]],
                                        'timestep': t})

            # Task 2.3: Additional constraints after goal is reached
            goal_time = len(path) - 1
            for t in range(goal_time, max_path_len):
                for j in range(i+1, self.num_of_agents):
                    constraints.append({'agent': j, 'loc': [path[-1]], 'timestep': t})
            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
