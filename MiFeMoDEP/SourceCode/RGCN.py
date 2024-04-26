from torch_geometric.nn import RGCNConv, SAGPooling
from torch import nn

class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.sag_pool = SAGPooling(hidden_channels, ratio=0.8)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)
        self.lin = nn.Linear(out_channels*1600, 128)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x, edge_index, edge_type, _, _, _ = self.sag_pool(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        x = x.view(x.size(0)*x.size(1))
        x = self.lin(x)
        return x

