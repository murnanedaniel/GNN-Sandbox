# System imports
import os
import sys
import argparse
import yaml

# Linking
sys.path.append('..')
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



def validate(model, val_loader, val_size, device):
    model = model.eval()
    node_correct, node_total, loss = 0, 0, 0
    for batch in val_loader:
        data = batch.to(device)
#            print(len(data.y_params))
        node_pred = model(data)
        node_correct += (((node_pred - data.y_params)/data.y_params)**2 < 0.1**2).sum().item()
        node_total += len(node_pred)
        loss += F.mse_loss(node_pred, data.y_params)
    acc = node_correct / node_total
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
    
    class TwoHopAttNetwork(nn.Module):
        """
        A module which computes new node features on the graph.
            For each node, it aggregates the neighbor node features
            (separately on the input and output side), and combines
            them with the node's previous features in a fully-connected
            network to compute the new features.
        """
        def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation=nn.ReLU,
                        layer_norm=True):
            super(TwoHopAttNetwork, self).__init__()
            self.network = make_mlp(input_dim*5, [hidden_dim, hidden_dim, hidden_dim, output_dim],
                                        hidden_activation=hidden_activation,
                                        output_activation=hidden_activation,
                                        layer_norm=layer_norm)

        def forward(self, x, e, edge_index):
            start, end = edge_index
            # Aggregate edge-weighted incoming/outgoing features
            mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
            mi2 = scatter_add(e[:, None]*scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])[start], end, dim=0, dim_size=x.shape[0])
            mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
            mo2 = scatter_add(e[:, None]*scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])[end], start, dim=0, dim_size=x.shape[0])
            node_inputs = torch.cat([mi, mi2, mo, mo2, x], dim=1)
            return self.network(node_inputs)

    class Edge_Track_Truth_Net(nn.Module):
        """
         Segment classification graph neural network model.
        Consists of an input network, an edge network, and a node network.
        """
        def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                         output_dim=3, hidden_activation=nn.ReLU, layer_norm=True):
            super(Edge_Track_Truth_Net, self).__init__()
            self.n_graph_iters = n_graph_iters

            # Setup the input network
            self.input_network = make_mlp(input_dim, [hidden_dim],
                                              hidden_activation=nn.ReLU,
                                              layer_norm=False)
             # Setup the node layers
            self.node_network = TwoHopAttNetwork(input_dim+hidden_dim, hidden_dim, hidden_dim,
                                                hidden_activation=nn.ReLU, layer_norm=False)

            self.output_network = make_mlp(input_dim+hidden_dim, [hidden_dim, hidden_dim, hidden_dim, output_dim],
                                               hidden_activation=nn.ReLU,
                                              output_activation=None,
                                              layer_norm=False)

        def forward(self, inputs):
            """Apply forward pass of the model"""
            # Apply input network to get hidden representation
            x = self.input_network(inputs.x)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
            # Loop over iterations of edge and node networks
            for i in range(self.n_graph_iters):
                # Apply edge network
                e = inputs.y_edges
                # Apply node network
                x = self.node_network(x, e, inputs.edge_index)
                # Shortcut connect the inputs onto the hidden representation
                x = torch.cat([x, inputs.x], dim=-1)
            # Apply final edge network
            return self.output_network(x)
    
     # -------------------- ----------------------- ----------------------------
    
    print("Loading data...")
    train_dataset, val_dataset = load_data(train_size=wandb.config.get("train_size",0), test_size=20)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=2, shuffle=True), DataLoader(val_dataset, batch_size=1, shuffle=True)
    
#     print("config:", dict(wandb.config.user_items()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using ", device)
    
    m_dic = ["hidden_dim", "n_graph_iters"]
    m_configs = {k:wandb.config.get(k,0) for k in m_dic} 
    m_configs = {'input_dim': 3, **m_configs, 'output_dim': 1}
        
    model = Edge_Track_Truth_Net(**m_configs).to(device)
    wandb.watch(model, log='all')
    
    o_dic = ["lr", "weight_decay"]
    o_configs = {k:wandb.config.get(k,0) for k in o_dic} 
    optimizer_fn = getattr(torch.optim, wandb.config.get("optimizer",0))
#     optimizer_kwargs = {"Adam": {}, "SGD": {}}
    optimizer = optimizer_fn(model.parameters(), amsgrad=True, **o_configs)
    
#     s_dic = ["step_size", "gamma"]
#     s_configs = {k:wandb.config.get(k,0) for k in s_dic} 
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.33, patience=20)
    
    model.train()
    
    print("Training...")
    
    # Initialise some tracked quantities
    ep, best_acc, lr = 0, 0, 1
    
#   for epoch in range(wandb.config.get("epochs", 0)):
    while (lr > 6e-6):
        for batch in train_loader:
            optimizer.zero_grad()
            data = batch.to(device)
            node_pred = model(data)
            loss = F.mse_loss(node_pred, data.y_params)
            loss.backward()
            optimizer.step()
        ep += 1
        val_acc, val_loss = validate(model, val_loader, 20, device)
        if (val_acc > best_acc): best_acc = val_acc
        scheduler.step(val_loss)
        lr = get_lr(optimizer)
        print("Epoch: " , ep, ", validation loss: ", val_loss, ", validation accuracy: ", val_acc*100, "%, lr: ", lr)
        wandb.log({"Validation Accuracy": val_acc, "Best Accuracy": best_acc, "Validation Loss": val_loss, "Learning Rate": lr, "Epochs": ep})


def main():
    
    # Parse the command line
    args = parse_args()
    
    # Load config YAML
    with open(args.config) as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
        
    # Instantiate WandB sweep ID
    sweep_id = wandb.sweep(sweep_config, entity= "murnanedaniel", project= "node_regression_sweep")
    
    # Run WandB weep agent
    wandb.agent(sweep_id, function=train)
    
if __name__ == '__main__':
    main()
