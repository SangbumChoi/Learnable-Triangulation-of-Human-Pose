# import torch
#
# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)
#
#     def forward(self, x):
#
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred
#
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
#
# model = TwoLayerNet(D_in, H, D_out)
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
# for t in range(500):
#     y_pred = model(x)
#
#     loss = criterion(y_pred, y)
#     if t % 100 == 99:
#         print(t, loss)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# print(model)
#
# a = ([1,1,2],[1,1,2])
# print(a[1:], *a[1:])

import torch
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

cuboid_side = 10
base_point = [0, 0, 0]
sides = np.array([cuboid_side, cuboid_side, cuboid_side])
position = base_point - sides / 2

volume_size = 10
xxx, yyy, zzz = torch.meshgrid(torch.arange(volume_size), torch.arange(volume_size), torch.arange(volume_size))
grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
print(grid, grid.shape)
grid = grid.reshape((-1, 3))
print(grid, grid.shape)
grid_coord = torch.zeros_like(grid)
print(grid_coord, grid_coord.shape)
grid_coord[:, 0] = position[0] + (sides[0] / (volume_size - 1)) * grid[:, 0]
grid_coord[:, 1] = position[1] + (sides[1] / (volume_size - 1)) * grid[:, 1]
grid_coord[:, 2] = position[2] + (sides[2] / (volume_size - 1)) * grid[:, 2]
print(grid_coord, grid_coord.shape)

coord_volume = grid_coord.reshape(volume_size, volume_size, volume_size, 3)
print(coord_volume.shape)