import torch

class Optimizer():
    def __init__(self, color: str, learning_rate: float, learning_rate_decay: float, x_start: float, y_start: float, loss_function) -> None:
        self.color = color
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.x_last = self.x_current = x_start
        self.y_last = self.y_current = y_start
        self.z_last = self.z_current = Optimizer.loss_function_value(x_start, y_start, loss_function)
        self.loss_function = loss_function
    
    def overwriteLastValues(self) -> None:
        self.x_last = self.x_current
        self.y_last = self.y_current
        self.z_last = self.z_current

    def loss_function_value(x: float, y: float, loss_function) -> float:
        return Optimizer.loss_function_tensor(torch.tensor(x), torch.tensor(y), loss_function).item()
    
    def loss_function_tensor(x: float, y: float, loss_function) -> torch.tensor:
        return loss_function(x, y)
