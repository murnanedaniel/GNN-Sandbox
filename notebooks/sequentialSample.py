import torch
import torch.nn as nn
import wandb

class Edge_Track_Net(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 output_dim=3, hidden_activation=nn.ReLU, layer_norm=True):
        super(Edge_Track_Truth_Net, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim, hidden_dim, output_dim],
                                      hidden_activation=nn.ReLU,
                                      layer_norm=False)

    def forward(self, inputs):
            """Apply forward pass of the model"""
            # Apply input network to get hidden representation
            x = self.input_network(inputs.x)
            
def make_mlp(input_size, sizes,
             hidden_activation=nn.ReLU,
             output_activation=nn.ReLU,
             layer_norm=False):
    """Construct an MLP with specified fully-connected layers."""
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)

------------- Jupyter code -------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m_configs = {'input_dim': 3, 'hidden_dim': 16, 'n_graph_iters': 3, 'output_dim': 1}
model = Edge_Track_Truth_Net(**m_configs).to(device)
o_configs = {'lr': 0.001, 'weight_decay': 1e-4}

optimizer = torch.optim.Adam(model.parameters(), **o_configs)
s_configs = {'step_size': 20, 'gamma': 0.9}
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **s_configs)

hyperconfig = {**m_configs, **t_configs, **s_configs, **o_configs}
wandb.init(project="node_regression", config=hyperconfig)
wandb.watch(model, log='all')



model.train()
ep = 0
for epoch in range(100):
    ep += 1
    for batch in train_loader:
        optimizer.zero_grad()
        data = batch.to(device)
        node_pred = model(data)
        loss = F.mse_loss(node_pred, data.y_params)
        loss_v_node.append(loss)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print("Epoch: " , ep, ", loss: ", loss.item(), ", node accuracy: ", node_acc*100, "%, lr: ", scheduler.get_lr())
    acc_v_node.append(node_acc)
    wandb.log({"Test Accuracy": node_acc, "Test Loss": loss.item(), "Learning Rate": scheduler.get_lr()[0]})