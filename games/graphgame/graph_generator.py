import numpy as np
import networkx as nx
import random
from os.path import exists
import matplotlib.pyplot as plt
import time

class GraphGenerator:
    
    def __init__(self, n, m, n_rooms, cycle):
        self.n = n
        self.m = m
        self.n_rooms = n_rooms
        self.cycle = cycle
        self.map_array = np.zeros((n, m))
        self.G = nx.Graph()
        self.current_pos = np.random.randint(0, n), np.random.randint(0, m)
        self.map_array[self.current_pos] = 1
        self.G.add_node(self.current_pos)


    def generate_instance(self):

        dir2delta = {'north': np.array((0, 1)),
                     'south': np.array((0, -1)),
                     'east': np.array((1, 0)),
                     'west': np.array((-1, 0))}

        def find_cycle(source=None, orientation=None):

            if not self.G.is_directed() or orientation in (None, "original"):

                def tailhead(edge):
                    return edge[:2]

            elif orientation == "reverse":

                def tailhead(edge):
                    return edge[1], edge[0]

            elif orientation == "ignore":

                def tailhead(edge):
                    if edge[-1] == "reverse":
                        return edge[1], edge[0]
                    return edge[:2]

            explored = set()
            cycle = []
            final_node = None
            for start_node in self.G.nbunch_iter(source):
                if start_node in explored:
                    # No loop is possible.
                    continue

                edges = []
                # All nodes seen in this iteration of edge_dfs
                seen = {start_node}
                # Nodes in active path.
                active_nodes = {start_node}
                previous_head = None

                for edge in nx.edge_dfs(self.G, start_node, orientation):
                    # Determine if this edge is a continuation of the active path.
                    tail, head = tailhead(edge)
                    if head in explored:
                        # Then we've already explored it. No loop is possible.
                        continue
                    if previous_head is not None and tail != previous_head:
                        # This edge results from backtracking.
                        # Pop until we get a node whose head equals the current tail.
                        # So for example, we might have:
                        #  (0, 1), (1, 2), (2, 3), (1, 4)
                        # which must become:
                        #  (0, 1), (1, 4)
                        while True:
                            try:
                                popped_edge = edges.pop()
                            except IndexError:
                                edges = []
                                active_nodes = {tail}
                                break
                            else:
                                popped_head = tailhead(popped_edge)[1]
                                active_nodes.remove(popped_head)

                            if edges:
                                last_head = tailhead(edges[-1])[1]
                                if tail == last_head:
                                    break
                    edges.append(edge)

                    if head in active_nodes:
                        # We have a loop!
                        cycle.extend(edges)
                        final_node = head
                        break
                    else:
                        seen.add(head)
                        active_nodes.add(head)
                        previous_head = head

                if cycle:
                    break
                else:
                    explored.update(seen)

            else:
                assert len(cycle) == 0
                answer= "No cycle found"

            for i, edge in enumerate(cycle):
                tail, head = tailhead(edge)
                if tail == final_node:
                    break
            if len(cycle) != 0:
                answer=cycle[i:]
                
            return answer
        
        flag=False
        
        first_node=self.current_pos
        paths=[]
        while self.G.number_of_nodes() < self.n_rooms :
            random_dir = np.random.choice(list(dir2delta.keys()))
            new_pos = tuple(np.array(self.current_pos) + dir2delta[random_dir])
            if min(new_pos) < 0 or new_pos[0] >= self.n or new_pos[1] >= self.m:
                # illegal move
                continue

            # initialize a copy of the graph to test whether it has a cycle  
            copy_graph = self.G.copy()
            copy_graph.add_node(new_pos)
            copy_graph.add_edge(self.current_pos, new_pos)
            start_time = time.time()
            answer_cycle = find_cycle(copy_graph, orientation="ignore")
            # Check for cycle before adding an edge
            if not self.cycle and answer_cycle != "No cycle found":
                # Skip adding the edge that creates a cycle if Flag is True
                if start_time > 8:
                    flag=True
                    break
                continue
            if flag==True:
                return "No graph generated"

            self.map_array[new_pos] = 1
            self.G.add_node(new_pos)
            self.G.add_edge(self.current_pos, new_pos)
            paths.append((self.current_pos,random_dir,new_pos))
            self.current_pos = new_pos

        # control time complexity
        if self.cycle and find_cycle(source=self.current_pos, orientation="ignore") == "No cycle found":
            random_node = random.choice(list(self.G.nodes()))
            while random_node == self.current_pos or random_node in self.G.neighbors(self.current_pos):
                random_node = random.choice(list(self.G.nodes()))
            self.G.add_edge(self.current_pos, random_node)
            self.current_pos = random_node

        nodes_graph=list(self.G.nodes())
        edges_graph=list(self.G.edges())
        if len(nodes_graph)<self.n_rooms:
            return "No graph generated"
        if self.cycle==False and find_cycle(source=self.current_pos, orientation="ignore") != "No cycle found":
            return "No graph generated"
        picture_number=random.randint(0,10000)
        picture_name="graph_"+str(picture_number)+".png"
        file_exists = exists("/Users/yerkesoul/Downloads/clembench/games/graphgame/resources/pictures/"+picture_name)
        if file_exists:
            picture_name="graph_"+str(picture_number+1)+".png"
        nx.draw_networkx(self.G, pos={n: n for n in self.G.nodes()})
        plt.savefig("/Users/yerkesoul/Downloads/clembench/games/graphgame/resources/pictures/"+picture_name)
        plt.clf()
        
        def direction_list_maker(node, directions_list):

            from_node=[]
            to_node=[]
            opposite_direction_dict={'north':'south', 'south':'north', 'east':'west', 'west':'east'}
            for d in directions_list:
                if d[0]==node:
                    from_node.append(d[1])
                elif d[2]==node:
                    opposite_direction=opposite_direction_dict[d[1]]
                    to_node.append(opposite_direction)

            combined=list(set(from_node) | set(to_node))
            return combined
        
        graph_directions=[]
        for n in nodes_graph:
            node_path=direction_list_maker(n, paths)
            graph_directions.append((n,node_path))

        #-----------------------------------------
        #get the next node fore the move

        def get_directions(node, direction_list):
            node_directions = None  
            for i in direction_list:
                if i[0]==node:
                    node_directions=i[1]
                    break
            return node_directions

        def next_node_label(node, direction_list,nodes_list):
            dir2delta_inverse = {'north': np.array((0, 1)),
                        'south': np.array((0, -1)),
                        'east': np.array((1, 0)),
                        'west': np.array((-1, 0))}
            
            node_directions=get_directions(node, direction_list)
            next_nodes_list=[]
            for move in node_directions:
                next_node=tuple(np.array(node)+dir2delta_inverse[move])
                if next_node not in nodes_list:
                    raise ValueError("The next chosen path is not possible")
                else:
                    next_nodes_list.append((move,next_node))
            return next_nodes_list

        moves_nodes_list=[]
        for node in self.G.nodes():
            node_dict={}
            node_dict["node"]=node
            node_moves_each=next_node_label(node,graph_directions,self.G.nodes())
            node_dict["node_moves"]=node_moves_each
            moves_nodes_list.append(node_dict)

        graph_dict={"Picture_Name":picture_name, "Grid_Dimension": str(self.n), "Graph_Nodes":nodes_graph, "Graph_Edges":edges_graph,"N_edges": len(list(self.G.edges())) , "Initial_Position": first_node, "Paths": paths,"Directions": graph_directions, "Moves": moves_nodes_list ,"Cycle":self.cycle}
        self.G.clear()
        return  graph_dict
    
