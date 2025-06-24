################################ LIBRERIAS UTILIZADAS ############################
import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import netwulf as nw
import matplotlib.pyplot as plt
import igraph as ig
import leidenalg as la
from collections import Counter
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import time
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import random
PYTHONHASHSEED=0
############################### LECTURA DE DATOS #################################################
#Leer los datos del excel limpios
# Ruta del archivo Excel
archivo = "C:/Users/Famil/OneDrive - Universidad Complutense de Madrid (UCM)/4º Ingeniería matemática/TFG/Datosvf/Base datos/Nodos_CTS.xlsx"

# Nombre de la hoja a leer
hoja = "Nodos_CTS"

# Leer la hoja específica en un DataFrame
datos = pd.read_excel(archivo, sheet_name=hoja)

# Crear variables de todas las columnas
nodosi = datos["ASIN"]
Nf1 = datos["Nf1"]
Nf2 = datos["Nf2"]
Nf3 = datos["Nf3"]
Nf4 = datos["Nf4"]
Nf5 = datos["Nf5"]
Grupo = list(datos["Group"])


###################### ANÁLISIS DE NODOS CLASIFICADOS Y NO CLASIFICADOS ########################
# Necesito guardar a qué grupo pertenece cada nodo de la columna nodosi, después veremos que hay nodos
# que no se muestran en los nodos iniciales, si no sólo en los nodos finales (en lo que serían las columnas Nf1...Nf5)
# (esos los meteremos en otro grupo pues no sabemos a qué género pertenecen)
Grupos = []
for i in range(len(nodosi)):
    Grupos.append((nodosi[i], Grupo[i]))

# Dado que parece que hay más nodos de los estipulados quiero hacer una lista de la unión de las columnas que contienen los nodos
nodos = list(set(nodosi) | set(Nf1) | set(Nf2) | set(Nf3) | set(Nf4) | set(Nf5))
nodos.remove(0)

nodosclasificadosvf = set(nodosi)
nodosnoclasificados = set(Nf1) | set(Nf2) | set(Nf3) | set(Nf4) | set(Nf5)

#Nodos que estaban como no clasificados pero que en verdad si que estan clasificados
nodosnoclasenclas = nodosnoclasificados & nodosclasificadosvf

#Ahora quiero quitar del conjunto de los que no estaban clasificados a los anteriores
nodosnoclasificadosvf = nodosnoclasificados - nodosnoclasenclas
nodosnoclasificadosvflist = list(nodosnoclasificadosvf)
nodosnoclasificadosvflist.remove(0)

#Creamos el grupo 4, que son los nodos no clasificados
for i in range(len(nodosnoclasificadosvflist)):
    Grupos.append((nodosnoclasificadosvflist[i], 4))

################################### CREACIÓN DE LAS ARISTAS ###########################################
# Creo las aristas (los 0 significan que no hay nodos finales (pongo 5 pues estaba prefiltrado así en la base de datos))
aristas = []
for i in range(len(nodosi)):
    if Nf1[i] != 0:
        aristas.append((nodosi[i], Nf1[i]))
    if Nf2[i] != 0:
        aristas.append((nodosi[i], Nf2[i]))
    if Nf3[i] != 0:
        aristas.append((nodosi[i], Nf3[i]))
    if Nf4[i] != 0:
        aristas.append((nodosi[i], Nf4[i]))
    if Nf5[i] != 0:
        aristas.append((nodosi[i], Nf5[i]))

########################## DICCIONARIOS DE NODOS Y ARISTAS #############################################
# Dado que hay algún nombre de algún nodo que incluye letras, creo un diccionario que vincule cada nodo a un id, y hago lo mismo con las aristas
nodos_map = {str(nodo): i for i, nodo in enumerate(nodos)}
aristas_indices = [(nodos_map[str(nodo1)], nodos_map[str(nodo2)])
                   for nodo1, nodo2 in aristas]


########################## CREACIÓN DEL GRAFO ORIGINAL ##########################################
# Creo el grafo
G = nx.Graph()
G.add_nodes_from(range(1, len(nodos)))
G.add_edges_from(aristas_indices)



################# ANÁLISIS DE LOS GRUPOS Cocina, Viajes Y Deporte ################################
#Voy a hacer los grupos a los que pertenece cada nodo (C, S, T, CS, CT, ST, CST)
g1 = []
g2 = []
g3 = []
g4 = []

for i in range(len(Grupos)):
    if Grupos[i][1] == 1:
        g1.append(nodos_map[str(Grupos[i][0])])
    elif Grupos[i][1] == 2:
        g2.append(nodos_map[str(Grupos[i][0])])
    elif Grupos[i][1] == 3:
        g3.append(nodos_map[str(Grupos[i][0])])
    else:
        g4.append(nodos_map[str(Grupos[i][0])])
        
        
gi123 = list(set(g1) & set(g2) & set(g3))
gi12 = list((set(g1) & set(g2))-set(gi123))
gi13 = list((set(g1) & set(g3))-set(gi123))
gi23 = list((set(g2) & set (g3))-set(gi123))


gu12 = list(set(g1) | set(g2))
gu13 = list(set(g1) | set(g3))
gu23 = list(set(g2) | set (g3))

#Nodos que pertenecen a cada grupo
g1f = list(set(g1)-set(gu23)-set(gi123))
g2f = list(set(g2)-set(gu13)-set(gi123))
g3f = list(set(g3)-set(gu12)-set(gi123))


g1234 = [set(g1f) , set(g2f) , set(g3f) , set(g4) , set(gi12) , set(gi13) , set(gi23), set(gi123)]
grupos_map ={}
for i in range(len(g1234)):
    grupos_map.update(dict.fromkeys(g1234[i], i))

#Creación de grafo con leyenda por colores dependiendo a qué grupo pertenecen    
nx.set_node_attributes(G, grupos_map, 'group')
            
node_groups = nx.get_node_attributes(G, 'group')

# Aplicar colores a los nodos en el grafo
colores_grupos = {
    0: "red",      # Cooking
    1: "blue",     # Sports
    2: "yellow",    # Travel
    3: "white",     # No clasificados
    4: "purple",   # Cooking & Sports
    5: "orange",   # Cooking & Travel
    6: "green",    # Sports & Travel
    7: "black"    # Cooking, Sports & Travel
}


# Crear un diccionario de colores para cada nodo
node_colors = {nodo: colores_grupos.get(grupo, "black") for nodo, grupo in node_groups.items()}

nx.set_node_attributes(G, node_colors, 'color')
nw.visualize(G)
G.number_of_nodes()
G.number_of_edges()
# =============================================================================
# Hay 5725 nodos que pertenecen a Cooking
# Hay 6497 nodos que pertenecen a Sports
# Hay 7141 nodos que pertenecen a Travel
# Hay 30 nodos que pertenecen a Cooking y Sports
# Hay 183 nodos que pertenecen a Cooking y Travel
# Hay 1192 nodos que pertenecen a Sports y Travel
# Hay 20207 nodos que no pertenecen a ninguno de los 3 grupos
# =============================================================================


############### ELIMINACIÓN DE NODOS NO CLASIFICADOS Y ARISTAS QUE LOS CONTIENEN ####################
#Para ello, creamos subgrafo con los nodos que sólo están clasificados y que pertenecen a única categoría
nodosclasificadosvflist = list(nodosclasificadosvf)
nodosvf = []
for i in range(len(nodosclasificadosvflist)):
    if nodos_map[str(nodosclasificadosvflist[i])] in g1f:
        nodosvf.append(nodos_map[str(nodosclasificadosvflist[i])])
    elif nodos_map[str(nodosclasificadosvflist[i])] in g2f:
        nodosvf.append(nodos_map[str(nodosclasificadosvflist[i])])
    elif nodos_map[str(nodosclasificadosvflist[i])] in g3f:
        nodosvf.append(nodos_map[str(nodosclasificadosvflist[i])])
        
#Creación de subgrafo con los nodos que interesan para el estudio (clasificados y que pertencen a un único grupo)
Glimp = G.subgraph(nodosvf)

#Numero de nodos y aristas
Glimp.number_of_nodes()
Glimp.number_of_edges()

# Visualizar el grafo con Netwulf
nw.visualize(Glimp)
# =============================================================================
# Hay 5725 nodos que pertenecen a Cooking
# Hay 6497 nodos que pertenecen a Sports
# Hay 7141 nodos que pertenecen a Travel
# =============================================================================


############################### COMPONENTE CONEXA 3 grupos #####################################################
Gextra = G.subgraph(nodosvf)
esconexo = nx.is_connected(Gextra)

#No es conexo por lo que queremos usar la componente conexa más grande (13997 nodos)
componentesconexas = nx.connected_components(Gextra)
componentescon = []
for i, comp in enumerate(componentesconexas, start=1):
    componentescon.append(comp) 
    
componentes_len = [len(com) for com in componentescon]

#Hago gráfico de distribución del tamaño de las componentes conexas
conteo = Counter(componentes_len)

# Crear el gráfico de barras
tabla = pd.DataFrame(conteo.items(), columns=["Número", "Frecuencia"]).sort_values(by="Número")
print(tabla)    
    
nodos_conexos_CTS = list(max(componentescon, key=len))

Gextra = G.subgraph(nodos_conexos_CTS)
nw.visualize(Gextra)

colores_grupos2 = {
    "red":0,      # Cooking
    "blue":1,     # Sports
    "yellow":2,    # Travel
     "purple":3,   # Cooking & Sports
     "orange":4,   # Cooking & Travel
     "green":5,    # Sports & Travel
     "black":6    # Cooking, Sports & Travel
}

grupos_mapextra = {node: colores_grupos2[Gextra.nodes[node]['color']] for node in Gextra.nodes()}
color_count = {}
for node in Gextra.nodes():
    color = Gextra.nodes[node]['color']
    if color in color_count:
        color_count[color] += 1
    else:
        color_count[color] = 1

print(color_count)

#Este será el grafo que utilizaré para la implementación de algoritmos
#------------------------------------------------------------------------------
#Finalmente observamos que nos quedamos con:
# Hay 4730 nodos que pertenecen a Cooking
# Hay 4603 nodos que pertenecen a Sports
# Hay 4664 nodos que pertenecen a Travel
# En total hay 13997 nodos
#------------------------------------------------------------------------------
#Hago una análisis del grafo
n_nodosextra = Gextra.number_of_nodes()
n_aristasextra = Gextra.number_of_edges()
grado_medioextra = sum(dict(Gextra.degree()).values()) / n_nodosextra
coef_cluster_globalextra = nx.average_clustering(Gextra)

clustering_valsextra = list(nx.clustering(Gextra).values())
deg_centextra = nx.degree_centrality(Gextra)


print(f"Nodos: {n_nodosextra}")
print(f"Aristas: {n_aristasextra}")
print(f"Grado medio: {grado_medioextra:.2f}")
print(f"Coef. de agrupamiento (global): {coef_cluster_globalextra:.4f}")
top5_degextra = sorted(deg_centextra.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 betweenness centrality:", top5_degextra)


plt.figure()
plt.hist(clustering_valsextra, bins=30, edgecolor='black')
plt.title("Distribución del coeficiente de clustering")
plt.xlabel("Clustering local")
plt.ylabel("Número de nodos")
plt.show()


grados = [d for n, d in Gextra.degree()]
plt.figure(figsize=(8, 5))
plt.hist(grados, bins=30, edgecolor='black')
plt.title("Distribución de grados")
plt.xlabel("Grado")
plt.ylabel("Número de nodos")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

################################# LOUVAIN 3 grupos #####################################
SEED = 1
random.seed(SEED)
np.random.seed(SEED)

#Ejecuto el algoritmo de Louvain
comunidades = nx_comm.louvain_communities(G=Gextra, weight="weight", resolution = 0.0051, seed = SEED)
comunidades_map = {}
for i in range(len(comunidades)):
    comunidades_map.update(dict.fromkeys(comunidades[i], i))

num_comunidades = len(set(comunidades_map.values()))
colormap = plt.cm.get_cmap('tab20', num_comunidades)

# Creao un diccionario {node: color} usando la comunidad
color_map = {
    node: '#' + ''.join(f'{int(c*255):02x}' for c in colormap(group)[:3])
    for node, group in comunidades_map.items()
}

# Asigno el color como atributo 'color' del nodo
nx.set_node_attributes(Gextra, color_map, 'color')
nw.visualize(Gextra)

modLou3 = nx.algorithms.community.quality.modularity(Gextra, comunidades)
# Etiquetas reales: las de los grupos originales (Cooking, Sports, etc.)
y_true = []
# Etiquetas predichas: las comunidades detectadas por Louvain
y_pred = []

for node in Gextra.nodes():
    y_true.append(grupos_mapextra[node])      # grupo real
    y_pred.append(comunidades_map[node])   # comunidad Louvain

ari3 = adjusted_rand_score(y_true, y_pred)  # y_true: grupos reales, y_pred: comunidades Louvain
print("ARI:", ari3)
nmi3 = normalized_mutual_info_score(y_true, y_pred)
print("NMI:", nmi3)

gammas = np.arange(0.003, 0.006, 0.0001)
resultados = []
for γ in gammas:
    part = nx_comm.louvain_communities(Gextra, weight="weight", resolution=γ, seed = SEED)
    for i in range(len(part)):
        comunidades_map.update(dict.fromkeys(part[i], i))
    pred_labels = [comunidades_map[n] for n in Gextra.nodes()]
    real = [grupos_mapextra[n] for n in Gextra.nodes()]
    ari = adjusted_rand_score(real, pred_labels)
    nmi = normalized_mutual_info_score(real, pred_labels)
    mod = nx.algorithms.community.quality.modularity(Gextra, part) 
    resultados.append((γ, len(part), ari, nmi, mod))

# Tabla sencilla
print("γ\tcomunidades\tARI\tNMI\tmod")
for r in resultados:
    print(f"{r[0]:.4f}\t{r[1]}\t\t{r[2]:.3f}\t{r[3]:.3f}\t{r[4]:.3f}")

# Gráficos
γs, ncoms, aris, nmis, mod = zip(*resultados)
plt.figure()
plt.plot(γs, aris, marker='o', label='ARI')
plt.plot(γs, nmis, marker='s', label='NMI')
plt.xlabel('Resolución γ')
plt.ylabel('Métrica')
plt.legend()
plt.title('Sensibilidad de Louvain al parámetro de resolución')
plt.show()

############################ LEIDEN ##########################################################
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
#Convertimos el grafo de 3 grupos de nettworkx a Igraph, que es el que neesitamos para poder implemnetar el algoritmo de Leiden
# Creo un mapeo de nodos: node_id → index
node_mapping = {node: idx for idx, node in enumerate(Gextra.nodes())}
reverse_mapping = {idx: node for node, idx in node_mapping.items()}
#Creamos grafo igraph vacío con el mismo número de nodos
edges = [(node_mapping[u], node_mapping[v]) for u, v in Gextra.edges()]
G_ig = ig.Graph()
G_ig.add_vertices(len(Gextra.nodes()))
G_ig.add_edges(edges)

partition =la.find_partition(G_ig, la.CPMVertexPartition, resolution_parameter = 0.0000015, seed = SEED)
communities_sets = [set(community) for community in partition]
    
comunidadesl_map = {}
for i, community in enumerate(partition):
    for node_index in community:
        node_original = reverse_mapping[node_index]
        comunidadesl_map[node_original] = i

num_comunidadesl = len(set(comunidadesl_map.values()))
colormapl = plt.colormaps['tab20'].resampled(num_comunidadesl)

# Creo un diccionario {node: color} usando la comunidad
colorl_map = {
    node: '#' + ''.join(f'{int(c*255):02x}' for c in colormapl(group)[:3])
    for node, group in comunidadesl_map.items()
}

# Asignar el color como atributo 'color' del nodo
nx.set_node_attributes(Gextra, colorl_map, 'color')
nw.visualize(Gextra)


modLou3 = nx.algorithms.community.quality.modularity(Gextra, comunidades)
# Etiquetas reales: las de los grupos originales (Cooking, Sports, etc.)
y_true = []
# Etiquetas predichas: las comunidades detectadas por Louvain
y_pred = []

for node in Gextra.nodes():
    y_true.append(grupos_mapextra[node])      # grupo real
    y_pred.append(comunidadesl_map[node])   # comunidad Louvain

ari3 = adjusted_rand_score(y_true, y_pred)  # y_true: grupos reales, y_pred: comunidades Louvain
print("ARI:", ari3)
nmi3 = normalized_mutual_info_score(y_true, y_pred)
print("NMI:", nmi3)


gammas = np.arange(0.0000007, 0.0000020, 0.0000001)
resultadosl = []
for γ in gammas:
    partition =la.find_partition(G_ig, la.CPMVertexPartition, resolution_parameter = γ, seed = SEED)
    communities_sets = [set(community) for community in partition]
        
    comunidadesl_map = {}
    for i, community in enumerate(partition):
        for node_index in community:
            node_original = reverse_mapping[node_index]
            comunidadesl_map[node_original] = i
    pred_labels = [comunidadesl_map[n] for n in Gextra.nodes()]
    real = [grupos_mapextra[n] for n in Gextra.nodes()]
    ari = adjusted_rand_score(real, pred_labels)
    nmi = normalized_mutual_info_score(real, pred_labels)
    mod = partition.modularity
    resultadosl.append((γ, len(partition), ari, nmi, mod))
    
print("γ\tcomunidades\tARI\tNMI\tmod")
for r in resultadosl:
    print(f"{r[0]:.10f}\t{r[1]}\t\t{r[2]:.3f}\t{r[3]:.3f}\t{r[4]:.3f}")
    
    
###################### NODE TO VEC 3 grupos ##################################################
SEED = 3
random.seed(SEED)
np.random.seed(SEED)

start_time = time.time()
Gextra = G.subgraph(nodos_conexos_CTS)
node2vec3 = Node2Vec(Gextra, 
    dimensions=16,       # Dimensión del embedding (el mejor tras probar con varias muestras (128, 64, 32, 16, 8, 4, 2))
    walk_length=80,       # Longitud de cada paseo aleatorio
    num_walks=10,         # Número de paseos por nodo
    p=1,                  # Parámetro de regreso (back)
    q=1,                  # Parámetro de exploración (in-out)
    workers=1,            # Número de procesos paralelos
    weight_key=None,      # Si el grafo tiene pesos en las aristas
    seed=SEED)               # Para reproducibilidad)
model3 = node2vec3.fit(window=10, min_count=1, batch_words=4, seed = SEED)
embeddings3 = model3.wv
end_time = time.time()
elapsed_time = end_time - start_time    
print(f"Tiempo de ejecución Node to Vec 3 grupos: {elapsed_time:.4f} segundos")


#Tarda 10 minutos aprox
#Lo representamos en 2D usando PCA
nodos3 = embeddings3.index_to_key
E3 = [embeddings3[n] for n in sorted(nodos3)]

scaler = StandardScaler()
X_scaled=scaler.fit_transform(E3)

pca3 = PCA(n_components=2, random_state= SEED)
X_pca=pca3.fit_transform(X_scaled)
expl = pca3.explained_variance_ratio_
print(sum(expl))

#Color
y_true = []
for node in sorted(nodos3):
    y_true.append(grupos_mapextra[int(node)]) 
    
#Dibujar scatter plot
plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], s=1, cmap= 'viridis', alpha=0.7, c=y_true)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Embeddings Node2Vec (16→2 dim) mediante PCA')
plt.grid(True)
plt.show()
    
plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], s=1, cmap= 'viridis', alpha=0.7)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Embeddings Node2Vec (16→2 dim) mediante PCA')
plt.grid(True)
plt.show()

########################## Kmeans 3 grupos ##########################################

random.seed(SEED)
np.random.seed(SEED)

start_time = time.time()
nodes3 = model3.wv.index_to_key 
X3= np.array([embeddings3[node] for node in sorted(nodes3)])
k = 3
scaler = StandardScaler()
X_scaled=scaler.fit_transform(X3)
kmeans3 = KMeans(n_clusters=k, random_state=SEED)
labels_kmeans3 = kmeans3.fit_predict(X_scaled)
end_time = time.time()
elapsed_time = end_time - start_time    
print(f"Tiempo de ejecución Node to Vec 3 grupos: {elapsed_time:.4f} segundos")

y_true = []
for node in sorted(nodes3):
    y_true.append(grupos_mapextra[int(node)])
    
y_pred = list(labels_kmeans3)


pca3 = PCA(n_components=2, random_state = SEED)
X_pca=pca3.fit_transform(X_scaled)
expl = pca3.explained_variance_ratio_
print(expl)

plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis', c=y_true, s=1,alpha=0.7)
plt.title("K-Means sobre Node2Vec")
plt.grid(True)
plt.show()

plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis', c=y_pred, s=1, alpha=0.7)
plt.title("K-Means sobre Node2Vec")
plt.grid(True)
plt.show()
ari3 = adjusted_rand_score(y_true, y_pred)  # y_true: grupos reales, y_pred: comunidades Louvain
print("ARI:", ari3)
nmi3 = normalized_mutual_info_score(y_true, y_pred)
print("NMI:", nmi3)

###################### DBSCAN 3 grupos ###########################################
embeddings= np.array([embeddings3[node] for node in sorted(nodes3)])

def find_optimal_epsilon(data, k):
    """
    Encuentra el epsilon óptimo usando el método k-distance
    """
    # Calcular distancias al k-ésimo vecino más cercano
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    
    # Tomar la distancia al k-ésimo vecino (última columna)
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Graficar para encontrar el "codo"
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-Distance Graph')
    plt.xlabel('Puntos ordenados por distancia')
    plt.ylabel(f'Distancia al {k}-ésimo vecino más cercano')
    plt.grid(True)
    plt.show()
    
    return distances

def apply_dbscan_multiple_params(data, eps_values, min_samples_values):
    """
    Prueba múltiples combinaciones de parámetros para DBSCAN
    """
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Aplicar DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            
            # Calcular métricas
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'labels': labels
            })
            
            print(f"eps={eps:.2f}, min_samples={min_samples}: "
                  f"{n_clusters} clusters, {n_noise} puntos de ruido")
    
    return results

def visualize_dbscan_results(data_original, labels, eps, min_samples):
    """
    Visualiza resultados en 2D usando PCA UNA SOLA VEZ
    """
    plt.figure(figsize=(12, 8))
      
    # PCA aplicado una sola vez a los datos originales
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_original)
   
    print("Varianza explicada para clustering (2 componentes):")
    print(f"Total: {pca.explained_variance_ratio_.sum():.2%}")
    # Crear colores únicos para cada cluster
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = (0.5, 0.5, 0.5, 0.3)  # Gris translúcido para ruido
            marker = 'x'
            size = 8
        else:
            marker = 'o'
            size = 20
        
        mask = labels == k
        plt.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                    c=[col], marker=marker, s=size,
                    label=f'Cluster {k}' if k != -1 else 'Ruido')
    
    
    plt.title(f'DBSCAN (eps={eps:.2f}, min_samples={min_samples})\n'
              f'Clusters: {len(unique_labels)-1}, Ruido: {list(labels).count(-1)}')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.show()

# Estandarizar los datos
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# PCA para DBSCAN (reducción de dimensiones por sensibilidad en alta dimensionalidad) 
pca_clustering = PCA(n_components=8)  # Para clustering
embeddings_for_dbscan = pca_clustering.fit_transform(embeddings_scaled)

# Mostrar varianza explicada del PCA para clustering
print("Varianza explicada para clustering (2 componentes):")
print(f"Total: {pca_clustering.explained_variance_ratio_.sum():.2%}")

#Aplico DBSCAN en 8 dimensiones
distances = find_optimal_epsilon(embeddings_for_dbscan, k=16)
eps_values = np.linspace(1.5, 2, 80)  # Ajustado según tu K-distance
#eps_values = np.linspace(2.9, 2.98, 20)
min_samples_values = [200,250,300,350,400,425,450]

results = apply_dbscan_multiple_params(embeddings_for_dbscan, eps_values, min_samples_values)

#Encuentro mejor configuración
best_configs = [r for r in results if r['n_clusters'] == 3]

if best_configs:
    print(f"\nConfiguración(es) que producen 3 clusters:")
    for config in best_configs[:]:  # Mostrar las primeras 3
        print(f"eps={config['eps']}, min_samples={config['min_samples']}, "
              f"ruido={config['n_noise']} puntos", f"%ruido={config['n_noise']/len(nodes3)}")
    
    # Usar la primera configuración válida
    best_config = best_configs[0]
    best_eps = best_config['eps']
    best_min_samples = best_config['min_samples']
    best_labels = best_config['labels']
   
    
else:
    # Si ninguna configuración da exactamente 3 clusters, usar valores por defecto
    print("\nNo se encontró configuración para 3 clusters exactos. Usando parámetros por defecto.")
    best_eps = 1.8924050632911391
    best_min_samples = 350
    
    # Aplicar DBSCAN con parámetros por defecto
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    best_labels = dbscan.fit_predict(embeddings_scaled)

# Visualizo el mejor resultado
print(f"\nVisualizando resultado con eps={best_eps}, min_samples={best_min_samples}")
for i in range(len(best_configs)):
    best_config = best_configs[i]
    best_eps = best_config['eps']
    best_min_samples = best_config['min_samples']
    best_labels = best_config['labels']
    visualize_dbscan_results(embeddings_scaled, best_labels, best_eps, best_min_samples)
    ari3 = adjusted_rand_score(y_true, best_labels)  # y_true: grupos reales, y_pred: comunidades Louvain
    print("ARI:", ari3)
    nmi3 = normalized_mutual_info_score(y_true, best_labels)
    print("NMI:", nmi3)

#♦Sólo voy a tomar el ari y nmi de los puntos que relamente ha clasificado (no de los que ha calificado como ruido)

best_configs = [r for r in results if (r['n_clusters'] == 3 and r['min_samples'] == 300 and r['n_noise'] == 8905)]
best_config = best_configs[0]
best_eps = best_config['eps']
best_min_samples = best_config['min_samples']
best_labels = list(best_config['labels'])
visualize_dbscan_results(embeddings_scaled, best_labels, best_eps, best_min_samples)
y_pred = []
y_truedb = []
for i in range(len(best_labels)):
    if best_labels[i] != -1:
        y_pred.append(best_labels[i])
        y_truedb.append(y_true[i])
ari3 = adjusted_rand_score(y_true, best_labels)  # y_true: grupos reales, y_pred: comunidades Louvain
print("ARI:", ari3)
nmi3 = normalized_mutual_info_score(y_true, best_labels)
print("NMI:", nmi3)        
ari3 = adjusted_rand_score(y_truedb, y_pred)  # y_true: grupos reales, y_pred: comunidades Louvain
print("ARI:", ari3)
nmi3 = normalized_mutual_info_score(y_truedb, y_pred)
print("NMI:", nmi3)
