from optimizer_collection import Optimizer_Collection
import torch

#?#################### Input Variables #####################

#General parameters
iterations = 50
iteration_pause = 0.00001 #Iteration_pause cannot be 0
x_start = -6.0 #float
y_start = -2.0 #float
loss_function = lambda x, y: torch.sin(y) * torch.exp(1-torch.cos(x))**2 + torch.cos(x) * torch.exp(1-torch.sin(y))**2 + (x-y)**2

#Start and end values for the axes in the plot, default is x_start, y_start +- 5
x_axis_start = -10
x_axis_end = 0  
y_axis_start = -10
y_axis_end = 0

#Parameters for optimizers
gd_active = True
gd_learning_rate = 0.05
gd_learning_rate_decay = 0.95

momentum_active = True
momentum_learning_rate = 0.02
momentum_learning_rate_decay = 0.95
momentum_velocity_loss_rate = 0.8

nesterov_active = False
nesterov_learning_rate = 0.02
nesterov_learning_rate_decay = 0.95
nesterov_velocity_loss_rate = 0.8

rms_prop_active = False
rms_prop_learning_rate = 0.4
rms_prop_learning_rate_decay = 0.95
rms_prop_squared_grad_forget_rate = 0.8

adam_active = False
adam_learning_rate = 0.2
adam_learning_rate_decay = 0.95
adam_exponential_decay_rate_first_moment = 0.9
adam_exponential_decay_rate_second_moment = 0.999

#?##########################################################


optimizer_collection = Optimizer_Collection(iterations, iteration_pause, x_start, y_start,
    gd_active, gd_learning_rate, gd_learning_rate_decay,
     momentum_active, momentum_learning_rate, momentum_learning_rate_decay, momentum_velocity_loss_rate, 
     nesterov_active, nesterov_learning_rate, nesterov_learning_rate_decay, nesterov_velocity_loss_rate, 
     rms_prop_active, rms_prop_learning_rate, rms_prop_learning_rate_decay, rms_prop_squared_grad_forget_rate, 
     adam_active, adam_learning_rate, adam_learning_rate_decay, adam_exponential_decay_rate_first_moment, adam_exponential_decay_rate_second_moment,
     loss_function, x_axis_start, x_axis_end, y_axis_start, y_axis_end)

optimizer_collection.plot()

