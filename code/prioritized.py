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

        # Pre-compute heuristics for each goal
        self.heuristics = [compute_heuristics(my_map, g) for g in self.goals]

    def find_solution(self):
        """Find paths for all agents using prioritized planning."""
        start_time = timer.time()
        result = []
        constraints = []

        for i in range(self.num_of_agents):
            # Plan for agent i
            path = a_star(self.my_map, self.starts[i], self.goals[i],
                          self.heuristics[i], i, constraints)
            if path is None:
                raise BaseException(f"No solution found for agent {i}")

            result.append(path)

            # ---------------------------------------------------------------
            # Task 2.1 — Add Vertex Constraints
            # ---------------------------------------------------------------
            for t, loc in enumerate(path):
                for j in range(i + 1, self.num_of_agents):
                    constraints.append({
                        'agent': j,
                        'loc': [loc],
                        'timestep': t
                    })

            # Keep goal reserved indefinitely (until time horizon)
            goal_time = len(path) - 1
            for t in range(goal_time + 1, goal_time + 50):
                for j in range(i + 1, self.num_of_agents):
                    constraints.append({
                        'agent': j,
                        'loc': [path[-1]],
                        'timestep': t
                    })

            # ---------------------------------------------------------------
            # Task 2.2 — Add Edge Constraints
            # ---------------------------------------------------------------
            # Prevent lower-priority agents from traversing the same edge or swapping
            for t in range(len(path) - 1):
                curr = path[t]
                nxt = path[t + 1]
                for j in range(i + 1, self.num_of_agents):
                    # Block same-direction edge
                    constraints.append({
                        'agent': j,
                        'loc': [curr, nxt],
                        'timestep': t + 1
                    })
                    # Block opposite-direction swap (nxt → curr)
                    constraints.append({
                        'agent': j,
                        'loc': [nxt, curr],
                        'timestep': t + 1
                    })
            # ---------------------------------------------------------------

        self.CPU_time = timer.time() - start_time

        print("\n***Run Prioritized***")
        print("\nFound a solution!\n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        print("***Test paths on a simulation***")
        return result
