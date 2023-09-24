input_file = open("/Users/chinmaymittal/Downloads/167.txt_graph", 'r')
output_file = open("yeast", "w")


def split_dataset(input_dataset):
    graphs = []
    current_graph = []

    for item in input_dataset:
        if not item.strip():
            if current_graph:
                graphs.append(current_graph)
                current_graph = []
        else:
            current_graph.append(item)

    if current_graph:
        graphs.append(current_graph)

    return graphs

atoms = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'S': 4, 'F': 5, 'Cl': 6, 'Si': 7,  'P': 8, 'I': 9, 'Br': 10}

def convert_graph(graph):
    graph_id = graph[0].strip()[1:]
    output_file.write(f"t # {graph_id}\n")
    num_vertices = int(graph[1].strip())
    for i in range(num_vertices):
        node_label = graph[i+2].strip()
        output_file.write(f"v {i} {atoms[node_label]}\n")
    num_edges = int(graph[2+num_vertices].strip())
    for i in range(num_edges):
        edge = graph[2+num_vertices+i+1].strip().split()
        output_file.write(f"e {edge[0]} {edge[1]} {edge[2]}\n")
    

lines = input_file.readlines()
graphs = split_dataset(lines)
for graph in graphs:
    convert_graph(graph)

input_file.close()
output_file.close()
print(len(graphs))
