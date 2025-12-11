from pymatgen.core import Structure
from pymatgen.io.vasp import Xdatcar
from pymatgen.analysis.local_env import CrystalNN  # or you can use VoronoiNN
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from e3nn.nn import FullyConnectedNet
import torch.nn as nn
import copy
import seaborn as sns



# Get number of atoms
num_atoms = len(structure)
print("Total number of the atoms in the Hetero structure:", num_atoms)

# Build edges
edges = []
n1_idx, n2_idx = num_atoms - 2, num_atoms - 1
for site in graph.graph:
    for neighbor in graph.graph[site]:
        edges.append((site, neighbor))
        edges.append((neighbor, site))
print(len(edges))

for i in range(num_atoms - 2):  # Exclude N2 atoms
    edges.append((n1_idx, i))
    edges.append((i, n1_idx))
    edges.append((n2_idx, i))
    edges.append((i, n2_idx))

# Print the edge list length
print(len(edges))

# Convert to tensor
edge_index = torch.tensor(edges, dtype=torch.long).T
print(edge_index.shape)

# Extract features (positions) from all structures
features = []
for item in range(len(xdatcar.structures)):
    positions = torch.tensor([list(site.coords) for site in xdatcar.structures[item].sites], dtype=torch.float)
    features.append(positions)

print(np.array(features).shape)
print(features[-1])

# Function to compute edge attributes
def compute_edge_attr(pos, edge_index):
    """Computes edge attributes as relative position vectors."""
    row, col = edge_index
    edge_attr = pos[col] - pos[row]  
    return edge_attr  

# Build dataset
dataset = []
for i in range(5000):
    periodic_graph = Data(x=features[i],
                          edge_index=edge_index,
                          edge_attr=compute_edge_attr(features[i], edge_index))
    dataset.append(periodic_graph)

print(dataset[0])

# Split dataset
train_graphs, temp_graphs = train_test_split(dataset, test_size=0.33, shuffle=False)
val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, shuffle=False)

print("length of training sample:\t", len(train_graphs))
print("length of validation sample:\t", len(val_graphs))
print("length of test sample:\t", len(test_graphs))

# Model definition
class EQGATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=3):
        super().__init__(aggr="add")
        self.attn_mlp = torch.nn.Linear(2*in_channels,1)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.node_embed = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        self.attn_scores = []
        out = self.propagate(edge_index, x=x)
        return out, torch.cat(self.attn_scores, dim=0) if self.attn_scores else None  # Only return updated node features

    def message(self, x_i, x_j):
        x_cat = torch.cat([x_i, x_j], dim=1)
        attn_weights = self.attn_mlp(x_cat)
        self.attn_scores.append(attn_weights)
        return attn_weights * self.lin(x_j)

class EQGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.eqgat1 = EQGATLayer(in_dim, hidden_dim)
        self.eqgat2 = EQGATLayer(hidden_dim, out_dim)
        self.pred = torch.nn.Linear(out_dim, 3)

    def forward(self, x, edge_index, edge_attr):
        x, attn1 = self.eqgat1(x, edge_index, edge_attr)
        x = F.relu(x)

        x, attn2 = self.eqgat2(x, edge_index, edge_attr)
        if attn1 is not None and attn2 is not None:
            attn_values = (attn1 + attn2) / 2  # Average attention across layers
        else:
            attn_values = attn1 if attn1 is not None else attn2

        return self.pred(x), attn_values  # Return node features & attention

model = EQGAT(in_dim=3, hidden_dim=16, out_dim=16)
print(model)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,)
criterion = nn.MSELoss()
# criterion = nn.SmoothL1Loss()

# Evaluation function
def eval_model(graphs, target):
    """Trains the GNN model to predict nitrogen movement."""
    model.eval()
    with torch.no_grad():
      loss_sum = 0

      for t in range(len(graphs) - 1):

          h, edge_index, edge_attr = graphs[t].x, graphs[t].edge_index, graphs[t].edge_attr
          next_h = graphs[t+1].x  # Next step as target

          output,_ = model(h, edge_index, edge_attr)
          loss = criterion(output[-2:], next_h[-2:])
          loss_sum += loss.item()

    return loss_sum/len(graphs)

# Training function
def train_dynamic_model(graphs, target, epochs=150, lr=0.001):
    """Trains a DynamicGAT GNN model."""
    # print(len(graphs))
    best_model = None
    best_loss = 10000
    best_epoch = 0
    train_loss_list = []
    validate_loss_list = []
    attention_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        loss_sum = 0
        for t in range(len(graphs) - 1):  # Train on time sequences
            h, edge_index, edge_attr = graphs[t].x, graphs[t].edge_index, graphs[t].edge_attr
            next_h = graphs[t+1].x  # Next step as target

            output, attention_values = model(h, edge_index, edge_attr)
            loss = criterion(output[-2:], next_h[-2:])  # Only nitrogen atoms
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if attention_values is not None:
                attention_history.append(attention_values.detach().cpu().numpy())

        train_loss = eval_model(train_graphs, target=None)
        train_loss_list.append(train_loss)
        val_loss = eval_model(val_graphs, target=None)
        validate_loss_list.append(val_loss)
        if val_loss < best_loss:
          best_loss = val_loss
          best_model = copy.deepcopy(model)
          best_epoch = epoch

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
    print(f'Epoch: {best_epoch:02d}, '
        f'Validation set MAE: {best_loss:.4f} eV')

    return best_model

# Run training
print(train_graphs[0])
gnn_model = train_dynamic_model(train_graphs, target=None)

# Influence computation
def compute_ga_s_influence(edge_index, edge_attn, num_GaS=36):
    """
    Extracts attention scores for Ga-S atoms influencing N2 atoms.
    Returns attention summed over each Ga-S atom.
    """
    attn_scores = torch.zeros(num_GaS)  # Initialize attention storage

    for i, (src, dest) in enumerate(edge_index.t()):
        if src < num_GaS and dest >= num_GaS:  
            attn_scores[src] += edge_attn[i].item()

    return attn_scores

def compute_feature_importance(graphs, model):
    model.eval()
    feature_importance = []
    for graph in graphs:
        h, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        _, edge_attn = model(h, edge_index, edge_attr)
        importance_scores = compute_ga_s_influence(edge_index, edge_attn)
        feature_importance.append(importance_scores.numpy())
    return np.mean(feature_importance, axis=0)

attention = compute_feature_importance(test_graphs,gnn_model)
print(attention)
total_attention = np.sum(attention) 
percentage_influence = (attention / total_attention) * 100


# Plotting
plt.figure(figsize=(18, 6))
sns.set_style("whitegrid")

# Atom labels
atoms = ['Ga$_{1}$', 'Ga$_{2}$', 'Ga$_{3}$', 'Ga$_{4}$', 'Ga$_{5}$',
         'Ga$_{6}$', 'Ga$_{7}$', 'Ga$_{8}$', 'Ga$_{9}$', 'Ga$_{10}$',
         'Ga$_{11}$', 'Ga$_{12}$', 'Ga$_{13}$', 'Ga$_{14}$', 'Ga$_{15}$',
         'Ga$_{16}$', 'Ga$_{17}$', 'Ga$_{18}$', 'S$_{1}$', 'S$_{2}$',
         'S$_{3}$', 'S$_{4}$', 'S$_{5}$', 'S$_{6}$', 'S$_{7}$',
         'S$_{8}$', 'S$_{9}$', 'S$_{10}$', 'S$_{11}$', 'S$_{12}$',
         'S$_{13}$', 'S$_{14}$', 'S$_{15}$', 'S$_{16}$', 'S$_{17}$', 'S$_{18}$']

x = np.arange(36)  

# Normalize values for color intensity
norm_values = (percentage_influence - np.min(percentage_influence)) / (np.max(percentage_influence) - np.min(percentage_influence))

# Color gradients: Blue for Ga, Red for S
ga_colors = sns.color_palette("Blues", 18)  
s_colors = sns.color_palette("Reds", 18)  
colors = np.array(ga_colors + s_colors)  

# Find top 5 influential atoms
top_5_idx = np.argsort(percentage_influence)[-5:]  # Indices of top 5 values

# Plot bars
bars = plt.bar(x, percentage_influence, color=colors, edgecolor="black", alpha=0.85)

# Highlight **Top 5 influential atoms** in **darker red**
for idx in top_5_idx:
    bars[idx].set_color('darkred')
    bars[idx].set_edgecolor('black')
    bars[idx].set_alpha(1.0)

# Formatting
plt.xticks(ticks=x, labels=atoms, rotation=45, fontsize=12, ha='right')
plt.ylabel("Influence (%)", fontsize=14)
plt.ylim(2.2, 3.5)
plt.xlabel("Ga-S sheet", fontsize=14)

# Clean visualization
sns.despine()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.savefig("Vaccum-Ga-sheet.png", dpi=100, bbox_inches="tight", pad_inches=0.1)
plt.show()
