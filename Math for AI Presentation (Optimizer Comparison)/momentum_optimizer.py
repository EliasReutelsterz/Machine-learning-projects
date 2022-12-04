from optimizer import Optimizer
import torch

class Momentum_Optimizer(Optimizer):
    def __init__(self, momentum_learning_rate: float, momentum_learning_rate_decay: float, momentum_velocity_loss_rate: float, x_start: float, y_start: float, loss_function) -> None:
        super().__init__(color = '#6FD08C', learning_rate = momentum_learning_rate, learning_rate_decay = momentum_learning_rate_decay, x_start = x_start, y_start = y_start, loss_function = loss_function)
        self.velocity_loss_rate = momentum_velocity_loss_rate
        self.x_velocity = self.y_velocity = 0.0
    
    def calculate_step(self, current_iteration) -> None:
        self.overwriteLastValues()
        x_tensor = torch.tensor(self.x_last, requires_grad=True)
        y_tensor = torch.tensor(self.y_last, requires_grad=True)
        z_tensor = Optimizer.loss_function_tensor(x_tensor, y_tensor, self.loss_function)
        z_tensor.backward()
        x_grad = x_tensor.grad.item()
        y_grad = y_tensor.grad.item()
        self.x_velocity = self.velocity_loss_rate * self.x_velocity - self.learning_rate * x_grad
        self.y_velocity = self.velocity_loss_rate * self.y_velocity - self.learning_rate * y_grad
        self.x_current = self.x_last + self.x_velocity
        self.y_current = self.y_last + self.y_velocity
        self.z_current = Optimizer.loss_function_value(self.x_current, self.y_current, self.loss_function)
        self.learning_rate = self.learning_rate * self.learning_rate_decay

        
