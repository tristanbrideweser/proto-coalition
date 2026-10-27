import os
import sys
import torch
from torch_geometric.data import HeteroData


# 1. Path Resolution Hook
# This dynamically finds your project root directory and tells Python it can import from 'models'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Package Import
from models.hgnn import HeMACGraphEncoder


def test_graph_encoder():
    # Instantiate your encoder
    encoder = HeMACGraphEncoder(hidden_dim=64, num_heads=4)
    encoder.eval()
    
    # Simulate a HeMAC environment state with 5 agents and 2 active tasks
    data = HeteroData()
    data['agent'].x = torch.randn(5, 12)  # 5 agents, each with 12 features
    data['task'].x = torch.randn(2, 6)    # 2 tasks, each with 6 features
    
    # Create mock communication and tracking graph edges
    data['agent', 'communicates', 'agent'].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )
    data['agent', 'tracks', 'task'].edge_index = torch.tensor(
        [[0, 2], [0, 1]], dtype=torch.long
    )
    
    # Run the encoder forward pass
    with torch.no_grad():
        z_out = encoder(data.x_dict, data.edge_index_dict)
        
    print("Graph Encoder Sanity Check Passed!")
    print(f"-> Active Agent Embeddings Shape: {z_out['agent'].shape}") # Expected: torch.Size([5, 64])
    print(f"-> Target Task Embeddings Shape: {z_out['task'].shape}")   # Expected: torch.Size([2, 64])


if __name__ == "__main__":
    test_graph_encoder()
