import torch
import torch.nn as nn


class EGCL(nn.Module):
    def __init__(self, node_dim, message_dim, hidden_dim, activation=nn.SiLU):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*self.node_dim + 2, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.message_dim),
            activation(),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(self.message_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.message_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

    def forward(self, x, r, adj):
        M, N = x.shape
        dim = r.shape[1]

        # Calculate pairwise distances
        d = torch.cdist(r, r, p=2.0)

        # Create mask for non-diagonal elements
        mask = ~torch.eye(M, dtype=bool, device=x.device)

        # Efficient reshaping and expansion
        x_pairs = x.unsqueeze(1).expand(-1, M, -1)[mask].view(M, M-1, N)
        x_expanded = x.unsqueeze(1).expand(-1, M-1, -1)
        r_pairs = r.unsqueeze(1).expand(-1, M, -1)[mask].view(M, M-1, dim)
        r_expanded = r.unsqueeze(1).expand(-1, M-1, -1)
        
        # Indexing for d and adj
        d_pairs = d[mask].view(M, M-1)
        adj_pairs = adj[mask].view(M, M-1)
        
        # Concatenate node features with distance and adjacency info
        x_concat = torch.cat([
            x_expanded, 
            x_pairs,
            d_pairs.unsqueeze(-1),
            adj_pairs.unsqueeze(-1)
        ], dim=-1).view(-1, N*2 + 2)

        # Update the position of the nodes
        m = self.edge_mlp(x_concat)
        move_weights = self.coord_mlp(m).view(M, M-1, 1)
        r = r + (1/(M - 1)) * ((r_pairs - r_expanded) * move_weights).sum(dim=1)

        m = m.view(M, M-1, self.message_dim)
        m = m.sum(dim=1)
        h = x + self.node_mlp(torch.cat([x, m], dim=-1))

        return h, r


class EGNN(nn.Module):
    def __init__(self, num_layers, node_dim, message_dim, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([EGCL(node_dim, message_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x, r, adj):
        for layer in self.layers:
            x, r = layer(x, r, adj)
        return x, r
