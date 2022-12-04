from optimizer import Optimizer
import torch
import numpy as np

class RMS_Prop_Optimizer(Optimizer):
    def __init__(self, rms_prop_learning_rate: float, rms_prop_learning_rate_decay: float, rms_prop_squared_grad_forget_rate: float, x_start: float, y_start: float, loss_function) -> None:
        super().__init__(color = '#DD7230', learning_rate = rms_prop_learning_rate, learning_rate_decay = rms_prop_learning_rate_decay, x_start = x_start, y_start = y_start, loss_function = loss_function)
        self.squared_grad_forget_rate = rms_prop_squared_grad_forget_rate
        self.x_squared_grad_mean = self.y_squared_grad_mean = 0
        self.epsilon_in_denominator = 1e-8

    def calculate_step(self, current_iteration) -> None:
        self.overwriteLastValues()
        x_tensor = torch.tensor(self.x_last, requires_grad=True)
        y_tensor = torch.tensor(self.y_last, requires_grad=True)
        z_tensor = Optimizer.loss_function_tensor(x_tensor, y_tensor, self.loss_function)
        z_tensor.backward()
        x_grad = x_tensor.grad.item()
        y_grad = y_tensor.grad.item()
        self.x_squared_grad_mean = self.squared_grad_forget_rate * self.x_squared_grad_mean + (1 - self.squared_grad_forget_rate) * x_grad**2
        self.y_squared_grad_mean = self.squared_grad_forget_rate * self.y_squared_grad_mean + (1 - self.squared_grad_forget_rate) * y_grad**2
        self.x_current = self.x_last - self.learning_rate * x_grad / (np.sqrt(self.x_squared_grad_mean) + self.epsilon_in_denominator)
        self.y_current = self.y_last - self.learning_rate * y_grad / (np.sqrt(self.y_squared_grad_mean) + self.epsilon_in_denominator)
        self.z_current = Optimizer.loss_function_value(self.x_current, self.y_current, self.loss_function)
        self.learning_rate = self.learning_rate * self.learning_rate_decay