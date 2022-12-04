from optimizer import Optimizer
import torch

class Nesterov_Optimizer(Optimizer):
    def __init__(self, nesterov_learning_rate: float, nesterov_learning_rate_decay: float, nesterov_velocity_loss_rate: float, x_start: float, y_start: float, loss_function) -> None:
        super().__init__(color = '#D65780', learning_rate = nesterov_learning_rate, learning_rate_decay = nesterov_learning_rate_decay, x_start = x_start, y_start = y_start, loss_function = loss_function)
        self.velocity_loss_rate = nesterov_velocity_loss_rate
        self.x_velocity = self.y_velocity = 0.0

    def calculate_step(self, current_iteration) -> None:
        self.overwriteLastValues()
        x_tensor_after_momentum = torch.tensor(self.x_last + self.velocity_loss_rate * self.x_velocity, requires_grad=True)
        y_tensor_after_momentum = torch.tensor(self.y_last + self.velocity_loss_rate * self.y_velocity, requires_grad=True)
        z_tensor_after_momentum = Optimizer.loss_function_tensor(x_tensor_after_momentum, y_tensor_after_momentum, self.loss_function)
        z_tensor_after_momentum.backward()
        x_grad_after_momentum = x_tensor_after_momentum.grad.item()
        y_grad_after_momentum = y_tensor_after_momentum.grad.item()
        self.x_velocity = self.velocity_loss_rate * self.x_velocity - self.learning_rate * x_grad_after_momentum
        self.y_velocity = self.velocity_loss_rate * self.y_velocity - self.learning_rate * y_grad_after_momentum
        self.x_current = self.x_last + self.x_velocity
        self.y_current = self.y_last + self.y_velocity
        self.z_current = Optimizer.loss_function_value(self.x_current, self.y_current, self.loss_function)
        self.learning_rate = self.learning_rate * self.learning_rate_decay