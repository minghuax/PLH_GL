import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv, ChebConv, GATv2Conv, GATConv, SAGEConv

class GCNConvLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

class SAGEConvLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

class FermiDiracDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=25, temperature=.5):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.abs(x)
        x = torch.clamp(x, min=0, max=40)        
        x = 1. / (torch.exp((x - 2.0) / self.temperature) + 1.0)
        return x

class Net(nn.Module):
    def __init__(self, x_dim, plh_dim, args):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        encoding_dim = 256
        self.encoder = None
        if args.name == 'GCN':
            self.encoder = GCNConvLayer(x_dim, args.hidden_channels, encoding_dim, dropout=args.emb_dropout)
        elif args.name == 'SAGE':
            self.encoder = SAGEConvLayer(x_dim, args.hidden_channels, encoding_dim, dropout=args.emb_dropout)
        decoder_input_dim = encoding_dim + plh_dim
        self.decoder = FermiDiracDecoder(decoder_input_dim, hidden_dim=args.mlp_hidden_channels, temperature=args.fd_temperature)
    
    @property
    def device(self):
        return self.dummy_param.device

    def forward(self, G_data, edges, x):
        emb = self.encoder(G_data.x, G_data.gnn_edge_index)
        torch.renorm(emb , 2, 0, 1)
        emb_in = emb[edges[:, 0]]
        emb_out = emb[edges[:, 1]]
        sqdist = (emb_in - emb_out).pow(2)
        if x is not None:
            x = x.to(self.device)
            # logging.debug("new_x: {}".format(x.shape))
            x = torch.cat((sqdist, x), dim=1)
        else:
            x = sqdist
        x = self.decoder(x)
        return x.reshape(-1)



def get_model(model, G_data, ph):
    x_dim = G_data.x.size(1)
    ph_dim = ph.resolution * ph.resolution * (ph.maxdim + 1) * 2 if ph.bool else 0
    if ph.both:
        ph_dim *= 2
    return Net(x_dim, ph_dim, model)
