import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

def create_graphs(number):
    
    def save_graph(G,text,i,number,directory,text2 = "") :
        plt.figure(figsize=(8, 6))
        pos_G = nx.spring_layout(G)
        nx.draw(G, pos=pos_G, with_labels=True)
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if 'weight' in d}
        draw_edges(G, pos_G)
        nx.draw_networkx_edge_labels(G, pos=pos_G, edge_labels=edge_labels)
        plt.title(f"{text}_{(i % number) + 1}")
        plt.savefig(os.path.join(directory, f"{text}_{(i % number) + 1}{text2}_prev.png"))
        plt.close()
    
    def rand_weight():
        return abs(int(np.random.normal(loc=0, scale=np.sqrt(variance)))) + 1
    
    def generate_connected_erdos_renyi_graph(n, p):
        while True:
            G = nx.erdos_renyi_graph(n, p)
            if nx.is_connected(G):
                return G
            print("NOT CONNECTED")
    
    def generate_connected_3_regular_graph(n):
        while True:
            G = nx.random_regular_graph(3, n)
            if nx.is_connected(G):
                return G
            print("NOT CONNECTED")

    def generate_2_random_connected(n, p, i, text,directory):
        n1 = n // 2
        n2 = n - n1

        G1 = generate_connected_erdos_renyi_graph(n1, p)
        G2 = generate_connected_erdos_renyi_graph(n2, p)

        G2 = nx.relabel_nodes(G2, {node: node + n1 for node in G2.nodes()})
        
        # Dodanie losowych wag do krawędzi w G1 i G2
        for u, v in G1.edges():
            G1[u][v]['weight'] = rand_weight()
            G1[u][v]['dir'] = random.choice(['forward','backward'])
        for u, v in G2.edges():
            G2[u][v]['weight'] = rand_weight()
            G2[u][v]['dir'] = random.choice(['forward','backward'])
        
        save_graph(G1,text,i,number,directory,"_G1")
        save_graph(G2,text,i,number,directory,"_G2")
        
        # Dodanie wagi dla krawędzi łączącej G1 i G2
        random_node_G1 = random.choice(list(G1.nodes()))
        random_node_G2 = random.choice(list(G2.nodes()))
        weight = random.randint(1, 10)
        G1.add_node(random_node_G2)
        G1.add_edge(random_node_G1, random_node_G2, weight=weight)

        G = nx.compose(G1, G2)
        
        
        return G

    def draw_edges(G, pos):
        for edge, dir in nx.get_edge_attributes(G, 'dir').items():
            if dir == 'forward':
                plt.annotate("", xy=pos[edge[1]], xytext=pos[edge[0]], arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            else:
                plt.annotate("", xy=pos[edge[0]], xytext=pos[edge[1]], arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))



    
    def generate_2D_grid_graph(n):
        G = nx.grid_2d_graph(int(np.sqrt(n)), int(np.sqrt(n)))
        num_nodes_per_row = int(np.sqrt(n))
        
        G = nx.relabel_nodes(G, {(i, j): i * num_nodes_per_row + j for i, j in G.nodes()})
        
        return G
    
    def generate_small_world_graph(n, k, p):
        G = nx.watts_strogatz_graph(n, k, p)
        return G


    text = "test"
    variance = np.sqrt(10)

    for i in range(6 * number):
        if i == number:
            text = "erdos_renyi"
            print()
        if i == 2 * number:
            text = "3_regular"
            print()
        if i == 3 * number:
            text = "2_random"
            print()
        if i == 4 * number:
            text = "2D_grid"
            print()
        if i == 5 * number :
            text = "small_world"
        print(i,text)
        
        directory = os.path.join('graphs', text)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        n = np.random.randint(15, 20)
        
        if text == "test" : n = 6
        p = 0.5
        

        if text == "erdos_renyi" or text == "test" and i % number == 0:
            G = generate_connected_erdos_renyi_graph(n, p)
        elif text == "3_regular" or text == "test" and i % number == 1:
            while n % 2 != 0:
                n = np.random.randint(15, 200)
            G = generate_connected_3_regular_graph(n)
        elif text == "2_random" or text == "test" and i % number == 2:
            G = generate_2_random_connected(n, p, i, text,directory)
        elif text == "2D_grid" or text == "test" and i % number == 3:
            G = generate_2D_grid_graph(n)
        elif text == "small_world" or text == "test" and i % number == 4:
            G = generate_small_world_graph(n, 4, 0.3)
        else:
            break


        for u, v in G.edges():
            if 'weight' not in G[u][v]:
                G[u][v]['weight'] = rand_weight()
            if 'dir' not in G[u][v] :
                G[u][v]['dir'] = random.choice(['forward','backward'])
        
        rang = G.number_of_nodes()
        
        
        # file_path = os.path.join(directory, f"{text}_{(i % number) + 1}.txt")
        file_path = os.path.join(directory, f"{text}_{(i % number) + 1}.txt")
        with open(file_path, 'w') as file:
            s = np.random.randint(0, rang)
            while s >= rang : s = np.random.randint(0, rang)
            t = np.random.randint(0, rang)
            while t == s or t >= rang:
                np.random.randint(0, rang)
            file.write(f"s {s}\nt {t}\nE {abs(int(np.random.normal(loc=0, scale=np.sqrt(variance)))) + 1}\nV {n}\n")
            for edge in G.edges(data=True):
                if edge[2]['dir'] == 'forward' :
                    file.write(f"e {edge[0]} {edge[1]} {edge[2]['weight']}\n")
                else :
                    file.write(f"e {edge[1]} {edge[0]} {edge[2]['weight']}\n")
        
        if text == "2D_grid" :
            rang *= rang
        
        
        save_graph(G,text,i,number,directory)
        
        print("done\n")


create_graphs(5)
