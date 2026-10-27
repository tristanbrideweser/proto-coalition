import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv

class HeMACGraphEncoder(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super(HeMACGraphEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        # asymmetric input projections
        self.agent_projection = nn.Sequential(
            nn.Linear(in_features=12, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        )

        self.task_projection = nn.Sequential(
            nn.Linear(in_features=6, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        )

        # heterogeneous message parsing layers
        self.hetero_conv1 = HeteroConv({
            ('agent', 'communicates', 'agent'): GATConv((-1, -1), hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('agent', 'tracks', 'task'):        GATConv((-1, -1), hidden_dim, heads=num_heads, concat=False, add_self_loops=False)
        }, aggr='mean')
        
        self.hetero_conv2 = HeteroConv({
            ('agent', 'communicates', 'agent'): GATConv((-1, -1), hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('agent', 'tracks', 'task'):        GATConv((-1, -1), hidden_dim, heads=num_heads, concat=False, add_self_loops=False)
        }, aggr='mean')

        # activation functions between graph layers
        self.activation = nn.ReLU()

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: Dictionary mapping node types to raw feature tensors.
                    e.g., {'agent': tensor_shape(num_agents, 12), 'task': tensor_shape(num_tasks, 6)}
            edge_index_dict: Dictionary mapping edge tuple types to PyG graph coordinate tensors.
                    e.g., {('agent', 'communicates', 'agent'): tensor_shape(2, num_edges)}
        Returns:
            z_dict: Uniform latent embeddings for all nodes.
        """
        # project asymmetric raw features into uniform latent dimensions
        h_dict = {
            'agent': self.agent_projection(x_dict['agent']),
            'task': self.task_projection(x_dict['task'])
        }

        # layer 1 heterogeneous message passing
        h_dict = self.hetero_conv1(h_dict, edge_index_dict)
        h_dict = {k: self.activation(v) for k, v in h_dict.items()}

        # layer 2 heterogeneous message parsing for deepened relational context
        z_dict = self.hetero_conv2(h_dict, edge_index_dict)

        return z_dict