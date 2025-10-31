import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """Planner that plans agents sequentially using prioritized planning (Tasks 2.1–2.4)."""

    def __init__(self, my_map, starts, goals):
        # store the map, start, and goal info
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        # number of agents in the problem
        self.num_of_agents = len(goals)
        # track total computation time
        self.CPU_time = 0
        # precompute heuristics (distance from any point to goal)
        self.heuristics = [compute_heuristics(my_map, g) for g in self.goals]


    # 2.1–2.4 Prioritized Planning Main Function
    def find_solution(self):
        """Plans paths for all agents one at a time while applying vertex, edge, and goal constraints."""

        start_time = timer.time()     # start timer
        result = []                   # stores all paths found so far
        constraints = []              # global list of constraints (shared by all agents)

        # plan each agent in order
        for i in range(self.num_of_agents):

    
            # 2.4 Time Horizon Limitation
            # put a cap on path length to avoid infinite loops
            # upper bound = total path lengths of all previous agents + grid size
            max_path_len = sum(len(p) for p in result) + len(self.my_map) * len(self.my_map[0])

            # run A* for this agent with all accumulated constraints
            path = a_star(self.my_map, self.starts[i], self.goals[i],
                          self.heuristics[i], i, constraints)

            # if A* fails or path is too long, stop and report failure
            if path is None or len(path) > max_path_len:
                raise BaseException(f"No solutions for agent {i} within time horizon")

            # store the path we found
            result.append(path)


            # 2.1 Vertex Constraints
            # for every step in this agent’s path, block that cell at that time
            # so later agents can’t occupy it at the same timestep
            for t, loc in enumerate(path):
                for j in range(i + 1, self.num_of_agents):
                    constraints.append({'agent': j, 'loc': [loc], 'timestep': t})


            # 2.2 Edge Constraints
            # prevent lower-priority agents from:
            #   1. moving through the same edge (u→v)
            #   2. swapping places with this agent (v→u)
            for t in range(len(path) - 1):
                curr, nxt = path[t], path[t + 1]
                for j in range(i + 1, self.num_of_agents):
                    # block same-direction movement
                    constraints.append({'agent': j, 'loc': [curr, nxt], 'timestep': t + 1})
                    # block swaps
                    constraints.append({'agent': j, 'loc': [nxt, curr], 'timestep': t + 1})


            # 2.3 Goal Constraints
            # once this agent reaches its goal, that cell stays occupied forever
            # prevents later agents from entering or waiting there
            goal_time = len(path) - 1
            for t in range(goal_time, max_path_len):
                for j in range(i + 1, self.num_of_agents):
                    constraints.append({'agent': j, 'loc': [path[-1]], 'timestep': t})


        # record total planning time
        self.CPU_time = timer.time() - start_time

   
        print("\n*** Run Prioritized Planning (Tasks 2.1–2.4) ***")
        print("\nFound a solution!\n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        print("*** Test paths on a simulation ***")

        # return all agent paths
        return result
