import time as timer
import heapq
import random
import copy
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost

# --- Task 3.1: Collision Detection ---
def detect_collision(path1, path2, a1=None, a2=None):
    max_t = max(len(path1), len(path2))
    for t in range(max_t):
        loc1 = get_location(path1, t)
        loc2 = get_location(path2, t)
        # Vertex collision
        if loc1 == loc2:
            return {'a1': a1, 'a2': a2, 'loc': [loc1], 'timestep': t}
        # Edge collision
        prev1 = get_location(path1, t-1) if t > 0 else path1[0]
        prev2 = get_location(path2, t-1) if t > 0 else path2[0]
        if loc1 == prev2 and loc2 == prev1:
            return {'a1': a1, 'a2': a2, 'loc': [prev1, loc1], 'timestep': t}
    return None

def detect_collisions(paths):
    collisions = []
    for i in range(len(paths)-1):
        for j in range(i+1, len(paths)):
            c = detect_collision(paths[i], paths[j], i, j)
            if c:
                collisions.append(c)
    return collisions

# --- Task 3.2: Standard Splitting ---
def standard_splitting(collision):
    constraints = []
    t = collision['timestep']
    if len(collision['loc']) == 1:
        loc = collision['loc']
        constraints.append({'agent': collision['a1'], 'loc': loc, 'timestep': t})
        constraints.append({'agent': collision['a2'], 'loc': loc, 'timestep': t})
    else:
        loc1, loc2 = collision['loc']
        constraints.append({'agent': collision['a1'], 'loc': [loc1, loc2], 'timestep': t})
        constraints.append({'agent': collision['a2'], 'loc': [loc1, loc2], 'timestep': t})
    return constraints

# --- Task 4.2: Disjoint Splitting ---
def disjoint_splitting(collision):
    constraints = []
    agent = random.choice([collision['a1'], collision['a2']])
    t = collision['timestep']
    if len(collision['loc']) == 1:
        loc = collision['loc']
        constraints.append({'agent': agent, 'loc': loc, 'timestep': t, 'positive': True})
        other = collision['a1'] if agent == collision['a2'] else collision['a2']
        constraints.append({'agent': other, 'loc': loc, 'timestep': t, 'positive': False})
    else:
        loc1, loc2 = collision['loc']
        constraints.append({'agent': agent, 'loc': [loc1, loc2], 'timestep': t, 'positive': True})
        other = collision['a1'] if agent == collision['a2'] else collision['a2']
        constraints.append({'agent': other, 'loc': [loc1, loc2], 'timestep': t, 'positive': False})
    return constraints

def paths_violate_constraint(paths, constraint):
    violating_agents = []
    t = constraint['timestep']
    for i, path in enumerate(paths):
        if len(constraint['loc']) == 1:
            if get_location(path, t) != constraint['loc'][0]:
                violating_agents.append(i)
        else:
            loc1, loc2 = constraint['loc']
            prev = get_location(path, t-1) if t > 0 else path[0]
            curr = get_location(path, t)
            if [prev, curr] != [loc1, loc2]:
                violating_agents.append(i)
    return violating_agents

# --- CBS Solver ---
class CBSSolver(object):
    def __init__(self, my_map, starts, goals):
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.open_list = []
        self.heuristics = [compute_heuristics(my_map, g) for g in goals]

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, _, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        start_time = timer.time()
        root = {'cost': 0, 'paths': [], 'constraints': [], 'collisions': []}

        # Compute initial paths
        for i in range(self.num_of_agents):
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['collisions'] = detect_collisions(root['paths'])
        root['cost'] = get_sum_of_cost(root['paths'])
        self.push_node(root)

        while self.open_list:
            node = self.pop_node()
            if not node['collisions']:
                self.print_results(node, start_time)
                return node['paths']

            collision = node['collisions'][0]
            if disjoint:
                constraints = disjoint_splitting(collision)
            else:
                constraints = standard_splitting(collision)

            for constraint in constraints:
                child = {'paths': copy.deepcopy(node['paths']),
                         'constraints': node['constraints'] + [constraint],
                         'collisions': [],
                         'cost': 0}

                # Determine which agents need new paths
                agent_ids = [constraint['agent']]
                if constraint.get('positive', False):
                    agent_ids = paths_violate_constraint(child['paths'], constraint)

                valid = True
                for agent in agent_ids:
                    path = a_star(self.my_map, self.starts[agent], self.goals[agent],
                                  self.heuristics[agent], agent, child['constraints'])
                    if path is None:
                        valid = False
                        break
                    child['paths'][agent] = path

                if not valid:
                    continue

                child['collisions'] = detect_collisions(child['paths'])
                child['cost'] = get_sum_of_cost(child['paths'])
                self.push_node(child)

        raise BaseException("No solutions found")

    def print_results(self, node, start_time):
        CPU_time = timer.time() - start_time
        print("\nFound a solution!")
        print("CPU time (s): {:.2f}".format(CPU_time))
        print("Sum of costs: {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:", self.num_of_expanded)
        print("Generated nodes:", self.num_of_generated)
