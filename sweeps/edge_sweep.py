# System imports
import os
import sys
import argparse
import yaml

# Linking
sys.path.append('.')
sys.path.append('/global/common/cori_cle7/software/jupyter/19-11/lib/python3.7/site-packages')
sys.path.append('/global/homes/d/danieltm/.local/lib/python3.7/site-packages')

# External imports
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import wandb

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('sweep.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='config.yaml')
    return parser.parse_args()

def edge_validate(model, val_loader, val_size, device):
    model = model.eval()
    edge_correct, edge_total, loss = 0, 0, 0
    for batch in val_loader:
        data = batch.to(device)
#             print(len(data.y_params))
        edge_pred = model(data)
        edge_pred = torch.sigmoid(edge_pred)
        edge_correct += ((edge_pred > 0.5) == (data.y_edges > 0.5)).sum().item()
        edge_total += len(edge_pred)
        loss += F.binary_cross_entropy_with_logits(edge_pred, data.y_edges)
    acc = edge_correct / edge_total
    return acc, loss.item()/val_size

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train():
    
    print("Initialising W&B...")
    wandb.init()

    import torch_geometric.data
    from torch_geometric.data import Data
    from torch_geometric.data import DataLoader
    from torch_scatter import scatter_add
    
    # Local imports
    from utils.toy_utils import load_data, make_mlp

    # -------------------- Architecture Definition -----------------------
    
    def n_hop_front(x, e, edge_index, n):
        start, end = edge_index
    #     print(n, "-hop forward shapes: ", x.shape, e.shape, edge_index.shape) 
        if n is 1:
            return [scatter_add(e[:,None]*x[start], end, dim=0, dim_size=x.shape[0])]
        else:
            running_hop = n_hop_front(x, e, edge_index, (n-1))
            forward_hop = [scatter_add(e[:, None]*(running_hop[0])[start], end, dim=0, dim_size=x.shape[0])]
            return forward_hop + running_hop

    def n_hop_back(x, e, edge_index, n):
        start, end = edge_index
    #     print(n, "-hop back shapes: ", x.shape, e.shape, edge_index.shape)
        if n is 1:
            return [scatter_add(e[:,None]*x[end], start, dim=0, dim_size=x.shape[0])]
        else:
            running_hop = n_hop_back(x, e, edge_index, (n-1))
            back_hop = [scatter_add(e[:, None]*(running_hop[0])[end], start, dim=0, dim_size=x.shape[0])]
            return back_hop + running_hop
    
    class EdgeNetwork(nn.Module):
        """
        A module which computes weights for edges of the graph.
        For each edge, it selects the associated nodes' features
        and applies some fully-connected network layers with a final
        sigmoid activation.
        """
        def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh,
                     layer_norm=True):
            super(EdgeNetwork, self).__init__()
            self.network = make_mlp(input_dim*2,
                                    [hidden_dim, hidden_dim, hidden_dim, 1],
                                    hidden_activation=hidden_activation,
                                    output_activation=None,
                                    layer_norm=layer_norm)

        def forward(self, x, edge_index):
            # Select the features of the associated nodes
            start, end = edge_index
            x1, x2 = x[start], x[end]
            edge_inputs = torch.cat([x[start], x[end]], dim=1)
            return self.network(edge_inputs).squeeze(-1)
    
    class NHopAttNetwork(nn.Module):
        """
        A module which computes new node features on the graph.
        For each node, it aggregates the neighbor node features
        (separately on the input and output side), and combines
        them with the node's previous features in a fully-connected
        network to compute the new features.
        """
        def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation=nn.ReLU,
                     layer_norm=True, hops=1):
            super(NHopAttNetwork, self).__init__()
            self.network = make_mlp(input_dim*(hops*2 + 1), [hidden_dim, hidden_dim, hidden_dim, output_dim],
                                    hidden_activation=hidden_activation,
                                    output_activation=hidden_activation,
                                    layer_norm=layer_norm)
            self.hops = hops

        def forward(self, x, e, edge_index):
    #         print("Input shapes: ", x.shape, e.shape, edge_index.shape)       
            node_inputs = torch.cat(n_hop_front(x, e, edge_index, self.hops) + [x] + n_hop_back(x, e, edge_index, self.hops), dim=-1)
    #         print("Network shape: ", node_inputs.shape)
            return self.network(node_inputs)

    class NHop_Edge_Class_Net(nn.Module):
        """
        Segment classification graph neural network model.
        Consists of 
        an input network, an edge network, and a node network.
        """
        def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                     hidden_activation=nn.ReLU, layer_norm=False, hops=1):
            super(NHop_Edge_Class_Net, self).__init__()
            self.n_graph_iters = n_graph_iters
            # Setup the input network
            self.input_network = make_mlp(input_dim, [hidden_dim],
                                          output_activation=hidden_activation,
                                          layer_norm=layer_norm)
            # Setup the edge network
            self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                            hidden_activation, layer_norm=layer_norm)
            # Setup the node layers
            self.node_network = NHopAttNetwork(input_dim+hidden_dim, hidden_dim, hidden_dim,
                                            hidden_activation=nn.ReLU, layer_norm=True, hops=hops)

        def forward(self, inputs):
            """Apply forward pass of the model"""
            # Apply input network to get hidden representation
            x = self.input_network(inputs.x)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
            # Loop over iterations of edge and node networks
            for i in range(self.n_graph_iters):
                # Apply edge network
                e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
                # Apply node network
                x = self.node_network(x, e, inputs.edge_index)
                # Shortcut connect the inputs onto the hidden representation
                x = torch.cat([x, inputs.x], dim=-1)
            # Apply final edge network
            return self.edge_network(x, inputs.edge_index)
    
     # -------------------- ----------------------- ----------------------------
    
    print("Loading data...")
    train_dataset, val_dataset = load_data(train_size=wandb.config.get("train_size",0), test_size=12)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=1, shuffle=True), DataLoader(val_dataset, batch_size=1, shuffle=True)
    
#     print("config:", dict(wandb.config.user_items()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using ", device)
    
    m_dic = ["hidden_dim", "n_graph_iters", "hops"]
    m_configs = {k:wandb.config.get(k,0) for k in m_dic} 
    m_configs = {'input_dim': 3, **m_configs}
        
    model = NHop_Edge_Class_Net(**m_configs).to(device)
    wandb.watch(model, log='all')
    
    o_dic = ["lr", "weight_decay"]
    o_configs = {k:wandb.config.get(k,0) for k in o_dic} 
    optimizer_fn = getattr(torch.optim, wandb.config.get("optimizer",0))
#     optimizer_kwargs = {"Adam": {}, "SGD": {}}
    optimizer = optimizer_fn(model.parameters(), amsgrad=True, **o_configs)
    
#     s_dic = ["step_size", "gamma"]
#     s_configs = {k:wandb.config.get(k,0) for k in s_dic} 
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.33, patience=10)
    
    model.train()
    
    print("Training...")
    
    # Initialise some tracked quantities
    ep, best_acc, lr = 0, 0, 1
    
#   for epoch in range(wandb.config.get("epochs", 0)):
    while (lr > 6e-6):
        for batch in train_loader:
            optimizer.zero_grad()
            data = batch.to(device)
            edge_pred = model(data)
            loss = F.binary_cross_entropy_with_logits(edge_pred, data.y_edges)
            loss.backward()
            optimizer.step()
        ep += 1
        val_acc, val_loss = edge_validate(model, val_loader, 12, device)
        if (val_acc > best_acc): best_acc = val_acc
        scheduler.step(val_loss)
        lr = get_lr(optimizer)
#         print("Epoch: " , ep, ", validation loss: ", val_loss, ", validation accuracy: ", val_acc*100, "%, lr: ", lr)
        wandb.log({"Validation Accuracy": val_acc, "Best Accuracy": best_acc, "Validation Loss": val_loss, "Learning Rate": lr, "Epochs": ep})


def main():
    
    print("Started main")
    
    # Parse the command line
    args = parse_args()
    
    # Load config YAML
    with open(args.config) as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
        
    # Instantiate WandB sweep ID
    sweep_id = wandb.sweep(sweep_config, entity= "murnanedaniel", project= "edge_classification_sweep")
    
    # Run WandB weep agent
    wandb.agent(sweep_id, function=train)
    
if __name__ == '__main__':
    main()
