# cbs.py
import time as timer
import heapq
from single_agent_planner import a_star, get_location, get_sum_of_cost, compute_heuristics


# 3.1 Detect Collisions
# checks if two agents bump into each other — either on the same spot
# or by swapping positions between timesteps
def detect_collision(path1, path2):
    max_t = max(len(path1), len(path2))  # go up to the longer path

    for t in range(max_t):
        loc1 = get_location(path1, t)  # where agent 1 is at time t
        loc2 = get_location(path2, t)  # where agent 2 is at time t

        # same cell at the same time → vertex collision
        if loc1 == loc2:
            return {'loc': [loc1], 'timestep': t}

        # if they swap spots → edge collision
        if t > 0:
            prev1 = get_location(path1, t - 1)
            prev2 = get_location(path2, t - 1)
            if prev1 == loc2 and prev2 == loc1:
                return {'loc': [prev1, loc1], 'timestep': t}

    # no collision found
    return None


# 3.2 Detect All Collisions
# loops through every pair of agent paths and records all collisions
def detect_collisions(paths):
    collisions = []  # where we’ll store all collision info

    for i in range(len(paths) - 1):
        for j in range(i + 1, len(paths)):
            c = detect_collision(paths[i], paths[j])  # check between agent i and j
            if c is not None:
                c['a1'] = i  # store agent indices
                c['a2'] = j
                collisions.append(c)  # keep it
    return collisions


# 4.1 Standard Splitting
# when we find a collision, we create constraints for both agents
# each agent gets a rule saying “don’t be here at this time”
def standard_splitting(collision):
    constraints = []
    t = collision['timestep']
    loc = collision['loc']

    if len(loc) == 1:  # vertex collision
        constraints.append({'agent': collision['a1'], 'loc': loc, 'timestep': t})
        constraints.append({'agent': collision['a2'], 'loc': loc, 'timestep': t})
    else:  # edge collision
        a1, a2 = collision['a1'], collision['a2']
        constraints.append({'agent': a1, 'loc': [loc[0], loc[1]], 'timestep': t})
        constraints.append({'agent': a2, 'loc': [loc[1], loc[0]], 'timestep': t})
    return constraints


# 4.2 Disjoint Splitting
# this version makes one agent get a “positive” constraint
# (basically, “you must be here”), while the other gets the opposite.
# I alternate which agent gets the positive one to keep it fair.
def disjoint_splitting(collision):
    t, loc = collision['timestep'], collision['loc']

    # deterministic choice for optimal CBS (lower-index agent gets chosen)
    chosen = min(collision['a1'], collision['a2'])

    # Disjoint splitting requires both children to constrain the SAME agent:
    # Child 1: negative for chosen; Child 2: positive for chosen.
    if len(loc) == 1:  # vertex collision
        return [
            {'agent': chosen, 'loc': loc, 'timestep': t, 'positive': False},
            {'agent': chosen, 'loc': loc, 'timestep': t, 'positive': True}
        ]
    else:  # edge collision
        x, y = loc
        # get the directed edge attempted by the chosen agent in this collision
        if chosen == collision['a1']:
            directed = [x, y]
        else:
            directed = [y, x]
        return [
            {'agent': chosen, 'loc': directed, 'timestep': t, 'positive': False},
            {'agent': chosen, 'loc': directed, 'timestep': t, 'positive': True}
        ]


# 4.3 Detect Agents Violating Constraints
# after adding a constraint, we check which agents break it
# (those are the ones that need to replan their path)
def paths_violate_constraint(paths, constraint):
    bad_agents = []
    t = constraint['timestep']

    for i, path in enumerate(paths):
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint.get('positive', False):
                # positive means the agent *must* be at this loc
                if get_location(path, t) != constraint['loc'][0]:
                    bad_agents.append(i)
            else:
                # negative means the agent *cannot* be at this loc
                if get_location(path, t) == constraint['loc'][0]:
                    bad_agents.append(i)
        else:  # edge constraint
            loc1, loc2 = constraint['loc']
            prev = get_location(path, t - 1) if t > 0 else path[0]
            curr = get_location(path, t)
            if constraint.get('positive', False):
                # must take this exact edge
                if [prev, curr] != [loc1, loc2]:
                    bad_agents.append(i)
            else:
                # cannot take this edge
                if [prev, curr] == [loc1, loc2]:
                    bad_agents.append(i)
    return bad_agents


# this is where everything ties together — the main CBS algorithm
class CBSSolver(object):
    def __init__(self, my_map, starts, goals):
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.open_list = []  # priority queue
        # precompute heuristics for all agents
        self.heuristics = [compute_heuristics(my_map, g) for g in goals]

    # pushes a node into the open list (with priority)
    def push_node(self, node):
        # cost-only priority, deterministic (for optimal CBS)
        heapq.heappush(self.open_list, (node['cost'], self.num_of_generated, node))
        self.num_of_generated += 1

    # pops the node with the lowest cost/priority
    def pop_node(self):
        _, _, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    # main CBS algorithm
    def find_solution(self, disjoint=True):
        start_time = timer.time()

        # create the root node — no constraints yet
        root = {'cost': 0, 'constraints': [], 'paths': [], 'collisions': []}

        # generate initial paths for all agents
        for i in range(self.num_of_agents):
            path = a_star(self.my_map, self.starts[i], self.goals[i],
                          self.heuristics[i], i, root['constraints'])
            if path is None:
                raise BaseException('No solutions found')
            root['paths'].append(path)

        # find any initial collisions and cost
        root['collisions'] = detect_collisions(root['paths'])
        root['cost'] = get_sum_of_cost(root['paths'])
        self.push_node(root)

        closed_set = set()

        # main CBS loop
        while self.open_list:
            node = self.pop_node()
            print(f"Expanding node with cost {node['cost']} and {len(node['collisions'])} collisions")

            # if no collisions, we’re done
            if not node['collisions']:
                self.print_results(node, start_time)
                return node['paths']

            # pick the earliest collision (no sort)
            # collision = node['collisions'][0]
            collision = min(node['collisions'], key=lambda c: c['timestep'])
            print("Chosen collision:", collision)

            # split it using disjoint or standard method
            constraints = disjoint_splitting(collision) if disjoint else standard_splitting(collision)

            # for each constraint, make a new child node
            for c in constraints:
                # skip if constraint already exists
                if c in node['constraints']:
                    continue

                # shallow copy of paths list
                child = {'constraints': node['constraints'] + [c],
                         'paths': [list(p) for p in node['paths']],
                         'collisions': [], 'cost': 0}

                # figure out which agents to replan
                if c.get('positive', False):
                    agents_to_replan = paths_violate_constraint(child['paths'], c)
                else:
                    agents_to_replan = [c['agent']]

                valid = True
                for agent in agents_to_replan:
                    path = a_star(self.my_map, self.starts[agent], self.goals[agent],
                                  self.heuristics[agent], agent, child['constraints'])
                    if path is None:
                        valid = False
                        break
                    child['paths'][agent] = path

                if not valid:
                    continue

                # check for new collisions
                child['collisions'] = detect_collisions(child['paths'])
                child['cost'] = get_sum_of_cost(child['paths'])

                # unique signature
                sig = (tuple(sorted((cc['agent'], tuple(cc['loc']),
                                     cc['timestep'], cc.get('positive', False))
                                    for cc in child['constraints'])),
                       tuple(tuple(p) for p in child['paths']))

                if sig in closed_set:
                    continue
                closed_set.add(sig)
                self.push_node(child)

        raise BaseException("No solutions found")

    def print_results(self, node, start_time):
        CPU_time = timer.time() - start_time
        print("\nFound a solution!")
        print(f"CPU time (s): {CPU_time:.2f}")
        print(f"Sum of costs: {get_sum_of_cost(node['paths'])}")
        print(f"Expanded nodes: {self.num_of_expanded}")
        print(f"Generated nodes: {self.num_of_generated}")
