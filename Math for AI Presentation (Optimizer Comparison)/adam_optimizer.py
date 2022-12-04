from optimizer import Optimizer
import torch
import numpy as np

class Adam_Optimizer(Optimizer):
    def __init__(self, adam_learning_rate: float, adam_learning_rate_decay: float, adam_exponential_decay_rate_first_moment: float, adam_exponential_decay_rate_second_moment: float, x_start: float, y_start: float, loss_function) -> None:
        super().__init__(color = '#9F1B04', learning_rate = adam_learning_rate, learning_rate_decay = adam_learning_rate_decay, x_start = x_start, y_start = y_start, loss_function = loss_function)        
        self.exponential_decay_rate_first_moment = adam_exponential_decay_rate_first_moment
        self.exponential_decay_rate_second_moment = adam_exponential_decay_rate_second_moment
        self.x_first_moment_estimate = self.y_first_moment_estimate = 0
        self.x_second_moment_estimate = self.y_second_moment_estimate = 0
        self.epsilon_in_denominator = 1e-8

    def calculate_step(self, current_iteration) -> None:
        self.overwriteLastValues()
        x_tensor = torch.tensor(self.x_last, requires_grad=True)
        y_tensor = torch.tensor(self.y_last, requires_grad=True)
        z_tensor = Optimizer.loss_function_tensor(x_tensor, y_tensor, self.loss_function)
        z_tensor.backward()
        x_grad = x_tensor.grad.item()
        y_grad = y_tensor.grad.item()
        self.x_first_moment_estimate = self.exponential_decay_rate_first_moment * self.x_first_moment_estimate + (1 - self.exponential_decay_rate_first_moment) * x_grad
        self.y_first_moment_estimate = self.exponential_decay_rate_first_moment * self.y_first_moment_estimate + (1 - self.exponential_decay_rate_first_moment) * y_grad
        self.x_second_moment_estimate = self.exponential_decay_rate_second_moment * self.x_second_moment_estimate + (1 - self.exponential_decay_rate_second_moment) * x_grad**2
        self.y_second_moment_estimate = self.exponential_decay_rate_second_moment * self.y_second_moment_estimate + (1 - self.exponential_decay_rate_second_moment) * y_grad**2
        x_bias_corrected_first_moment_estimate = self.x_first_moment_estimate / (1 - self.exponential_decay_rate_first_moment**(current_iteration + 1))
        y_bias_corrected_first_moment_estimate = self.y_first_moment_estimate / (1 - self.exponential_decay_rate_first_moment**(current_iteration + 1))
        x_bias_corrected_second_moment_estimate = self.x_second_moment_estimate / (1 - self.exponential_decay_rate_second_moment**(current_iteration + 1))
        y_bias_corrected_second_moment_estimate = self.y_second_moment_estimate / (1 - self.exponential_decay_rate_second_moment**(current_iteration + 1))
        self.x_current = self.x_current - self.learning_rate * x_bias_corrected_first_moment_estimate / (np.sqrt(x_bias_corrected_second_moment_estimate) + self.epsilon_in_denominator)
        self.y_current = self.y_current - self.learning_rate * y_bias_corrected_first_moment_estimate / (np.sqrt(y_bias_corrected_second_moment_estimate) + self.epsilon_in_denominator)
        self.z_current = Optimizer.loss_function_value(self.x_current, self.y_current, self.loss_function)
        self.learning_rate = self.learning_rate * self.learning_rate_decay        
