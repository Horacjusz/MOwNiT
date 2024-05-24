import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from collections import deque, OrderedDict

def read_graph(filename):
    G = nx.DiGraph()  # Używamy Directed Graph, aby uwzględnić kierunek krawędzi
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[0] == 's':
                s = int(line[1])
            elif line[0] == 't':
                t = int(line[1])
            elif line[0] == 'E':
                E = float(line[1])
            elif line[0] == 'V':
                V = int(line[1])
            elif line[0] == 'e':
                u, v, w = int(line[1]), int(line[2]), float(line[3])
                # Dodajemy krawędź od u do v z wagą oraz krawędź od v do u z przeciwną wagą
                edges.append((u,v,-w))
                edges.append((v,u,w))
                G.add_edge(u, v, weight=-w, amperage = None)
                G.add_edge(v, u, weight=w, amperage = None)
    return G, s, t, E,edges


def find_paths(graph, start, end, n, to_solve, vector, E, edges, e, G, eps=1e-8):
    paths = []  # Lista przechowująca znalezione ścieżki
    V = len(to_solve)

    counter = 0

    # BFS
    queue = deque([(start, OrderedDict({start: True}))])  # Kolejka przechowująca wierzchołki i ich ścieżki
    while queue:  # Dodatkowe sprawdzenie czy nie znaleziono już wystarczającej liczby ścieżek
        current, path = queue.popleft()
        # print(counter)
        counter += 1
        
        if current == end:
            for v in path : print(v,end = " ")
            print()
            paths.append(list(path.keys()))  # Znaleziono ścieżkę od start do end
            row = [0] * (E + 1)
            u = next(iter(path))
            for v in path:
                if u == v:
                    continue
                row[edges[(u, v)]] = G[u][v]['weight']
                u = v
            vector = np.concatenate((vector, np.array([e])), axis=None)
            to_solve = np.concatenate((to_solve, np.array([row])), axis=0)
            print(to_solve,vector)
            print()
            
            # Sprawdzanie, czy wszystkie wiersze macierzy to_solve oraz wektora vector są niezerowe
            if len(to_solve) == n:
                
                print(to_solve,vector,len(to_solve),n)
                print()
                
                output = None
                try :
                    output = np.linalg.solve(to_solve, vector)
                except :
                    print("Weren\'t ready yet, index:",len(paths))
                    to_solve = np.delete(to_solve,V,axis=0)
                    vector = np.delete(vector,V,axis=0)
                    output = None
                if output is not None :
                    return output
                    
        else:
            for neighbor in graph.neighbors(current):
                if neighbor not in path:  # Zapobieganie cyklom
                    updated_path = OrderedDict(path)
                    updated_path[neighbor] = True
                    queue.append((neighbor, updated_path))

    for path in paths :
        print(path)


def check_graphs(text, number, eps=10**-8):
    graph_name = f"{text}_{number}"
    filename = f"graphs/{text}/{graph_name}.txt"

    G, s, t, e, edges_order = read_graph(filename)
    V = G.number_of_nodes()
    E = G.number_of_edges() // 2

    to_solve = np.array([[0 for _ in range(E + 1)] for _ in range(V)], dtype=float)
    vector = np.array([0] * V, dtype=float)

    edges = OrderedDict()
    indices = [(s, t)]
    edge_ind = 1
    for u, v, w in edges_order:
        if G[u][v]['weight'] > 0:
            continue
        print(u,v)
        if u == s:
            to_solve[u][0] = 1
        if u == t:
            to_solve[u][0] = -1
        if v == s:
            to_solve[v][0] = 1
        if v == t:
            to_solve[v][0] = -1

        edges[(u, v)] = edge_ind
        edges[(v, u)] = edge_ind
        indices.append((u, v))

        to_solve[u][edge_ind] = -1
        to_solve[v][edge_ind] = 1
        edge_ind += 1

    paths_to_find = E + 1

    print(to_solve)
    print()

    amps = find_paths(G, s, t, paths_to_find, to_solve, vector, E, edges, e, G)
    if amps is None : 
        print("NONE")
        return 
    D = nx.DiGraph()

    minimum = float('inf')
    maximum = float('-inf')

    for i in range(1, len(amps)):
        u, v = indices[i]
        G[u][v]['amperage'] = -amps[i]
        G[v][u]['amperage'] = amps[i]

        maximum = max(maximum, abs(amps[i]))
        minimum = min(minimum, abs(amps[i]))

        if amps[i] > 0:
            D.add_edge(u, v, amperage=amps[i])
        else:
            D.add_edge(v, u, amperage=-amps[i])

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(D)

    node_colors = ['pink' if node == s else 'purple' if node == t else 'skyblue' for node in D.nodes()]
    nx.draw(D, pos, with_labels=True, node_color=node_colors, node_size=500, arrows=True)

    edge_colors = []
    for u, v, data in D.edges(data=True):
        amperage = data['amperage']
        if abs(amperage - minimum) < eps:
            edge_colors.append('green')
        elif abs(amperage - maximum) < eps:
            edge_colors.append('red')
        else:
            if (maximum - minimum) <= eps:
                green_value, red_value = 0, 255
            else:
                a_g = (-255) / (maximum - minimum)
                b_g = 255 - a_g * minimum
                green_value = abs(a_g * amperage + b_g)
                a_r = (255) / (maximum - minimum)
                b_r = -a_r * minimum
                red_value = abs(a_r * amperage + b_r)
            edge_colors.append(mcolors.to_hex((red_value / 255, green_value / 255, 0)))

    nx.draw_networkx_edges(D, pos, edge_color=edge_colors, width=2.0, arrows=True)

    input_patch = mpatches.Patch(color='purple', label='Input')
    output_patch = mpatches.Patch(color='pink', label='Output')
    min_patch = mpatches.Patch(color='green', label=f'Min ({minimum})')
    max_patch = mpatches.Patch(color='red', label=f'Max ({maximum})')

    plt.legend(handles=[input_patch, output_patch, min_patch, max_patch], loc='upper right')
    plt.title('Graph D with Amperage Values')
    plt.savefig(f"graphs/{text}/{graph_name}_solve.png")
    print(text,number,"done")





texts = ["test","erdos_renyi","3_regular","2_random","2D_grid","small_world"]

# for number in range(1,6) :
#     for text in texts :
#         print("Test",text,number)
#         check_graphs(text, number)
#         print("Done\n")

text = texts[0]


for number in range(2,3) :
    check_graphs(text,number)
