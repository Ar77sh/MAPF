import heapq

# ---------- Map movements ----------
def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

# ---------- Cost helper ----------
def get_sum_of_cost(paths):
    return sum(len(path)-1 for path in paths)

# ---------- Heuristics (Dijkstra to goal) ----------
def compute_heuristics(my_map, goal):
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (0, goal, root))
    closed_list[goal] = root

    while open_list:
        cost, loc, curr = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            if child_loc[0]<0 or child_loc[0]>=len(my_map) or child_loc[1]<0 or child_loc[1]>=len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child_cost = cost + 1
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                if closed_list[child_loc]['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))
    return {loc: node['cost'] for loc, node in closed_list.items()}

# ---------- Constraint table ----------
def build_constraint_table(constraints, agent):
    table = dict()
    for c in constraints:
        if c['agent'] == agent or c['agent'] is None:
            t = c['timestep']
            if t not in table:
                table[t] = []
            table[t].append(c)
    return table

# ---------- Get location at time t ----------
def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]

# ---------- Reconstruct path ----------
def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr.get('parent', None)
    path.reverse()
    return path

# ---------- Check constraints ----------
def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    if next_time in constraint_table:
        for c in constraint_table[next_time]:
            positive = c.get('positive', False)
            # Negative constraints
            if not positive:
                if len(c['loc'])==1 and c['loc'][0]==next_loc:
                    return True
                elif len(c['loc'])==2 and c['loc']==[curr_loc, next_loc]:
                    return True
            # Positive constraints
            else:
                if len(c['loc'])==1 and c['loc'][0]!=next_loc:
                    return True
                elif len(c['loc'])==2 and c['loc']!=[curr_loc, next_loc]:
                    return True
    return False

# ---------- Priority Queue Helpers ----------
def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val']+node['h_val'], node['h_val'], node['loc'], node))

def pop_node(open_list):
    _, _, _, node = heapq.heappop(open_list)
    return node

def compare_nodes(n1, n2):
    return n1['g_val']+n1['h_val'] < n2['g_val']+n2['h_val']

# ---------- A* Search with Space-Time ----------
def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    open_list = []
    closed_list = dict()
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_values[start_loc], 'parent': None, 'timestep':0}
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root

    constraint_table = build_constraint_table(constraints, agent)
    max_timestep = 1000  # safety limit for CBS positive constraints

    while open_list:
        curr = pop_node(open_list)
        if curr['loc']==goal_loc:
            # Check goal positive constraint (wait until allowed)
            blocked = False
            t = curr['timestep']
            while t <= max_timestep:
                if is_constrained(curr['loc'], curr['loc'], t, constraint_table):
                    blocked = True
                    break
                t += 1
            if not blocked:
                return get_path(curr)

        # Expand neighbors
        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            if child_loc[0]<0 or child_loc[0]>=len(my_map) or child_loc[1]<0 or child_loc[1]>=len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc,
                     'g_val': curr['g_val']+1,
                     'h_val': h_values[child_loc],
                     'parent': curr,
                     'timestep': curr['timestep']+1}
            if is_constrained(curr['loc'], child_loc, child['timestep'], constraint_table):
                continue
            key = (child['loc'], child['timestep'])
            if key in closed_list:
                existing = closed_list[key]
                if compare_nodes(child, existing):
                    closed_list[key] = child
                    push_node(open_list, child)
            else:
                closed_list[key] = child
                push_node(open_list, child)

        # Wait in place
        child = {'loc': curr['loc'],
                 'g_val': curr['g_val']+1,
                 'h_val': h_values[curr['loc']],
                 'parent': curr,
                 'timestep': curr['timestep']+1}
        if not is_constrained(curr['loc'], curr['loc'], child['timestep'], constraint_table):
            key = (child['loc'], child['timestep'])
            if key not in closed_list or compare_nodes(child, closed_list[key]):
                closed_list[key] = child
                push_node(open_list, child)

    return None
