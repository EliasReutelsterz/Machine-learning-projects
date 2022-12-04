from optimizer import Optimizer
import torch

class GD_Optimizer(Optimizer):
    def __init__(self, gd_learning_rate: float, gd_learning_rate_decay: float, x_start: float, y_start: float, loss_function):
        super().__init__(color = '#94E5FF', learning_rate = gd_learning_rate, learning_rate_decay = gd_learning_rate_decay, x_start = x_start, y_start = y_start, loss_function = loss_function)

    def calculate_step(self, current_iteration):
        self.overwriteLastValues()
        x_tensor = torch.tensor(self.x_last, requires_grad=True)
        y_tensor = torch.tensor(self.y_last, requires_grad=True)
        z_tensor = Optimizer.loss_function_tensor(x_tensor, y_tensor, self.loss_function)
        z_tensor.backward()
        x_grad = x_tensor.grad.item()
        y_grad = y_tensor.grad.item()
        self.x_current = self.x_last - self.learning_rate * x_grad
        self.y_current = self.y_last - self.learning_rate * y_grad
        self.z_current = Optimizer.loss_function_value(self.x_current, self.y_current, self.loss_function)
        self.learning_rate = self.learning_rate * self.learning_rate_decay