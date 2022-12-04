import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from optimizer import Optimizer
from gd_optimizer import GD_Optimizer
from momentum_optimizer import Momentum_Optimizer
from nesterov_optimizer import Nesterov_Optimizer
from rms_prop_optimizer import RMS_Prop_Optimizer
from adam_optimizer import Adam_Optimizer
from gif_maker import GifMaker
from datetime import datetime
import os
from dill.source import getsource


class Optimizer_Collection():
    def __init__(self, iterations: int, iteration_pause: float, x_start: float, y_start: float,
    gd_active: bool, gd_learning_rate: float, gd_learning_rate_decay: float,
    momentum_active: bool, momentum_learning_rate: float, momentum_learning_rate_decay:float,  momentum_velocity_loss_rate: float, 
    nesterov_active: bool, nesterov_learning_rate: float, nesterov_learning_rate_decay: float, nesterov_velocity_loss_rate: float, 
    rms_prop_active: bool, rms_prop_learning_rate: float, rms_prop_learning_rate_decay: float, rms_prop_squared_grad_forget_rate: float, 
    adam_active: bool, adam_learning_rate: float, adam_learning_rate_decay: float, adam_exponential_decay_rate_first_moment: float, adam_exponential_decay_rate_second_moment: float,
    loss_function = lambda x, y: x**2 + y**2, x_axis_start:float = None, x_axis_end:float = None, y_axis_start:float = None, y_axis_end:float = None) -> None:
        
        self.iterations = iterations
        self.iteration_pause = iteration_pause
        self.x_start = x_start
        self.y_start = y_start
        self.loss_function = loss_function
        if x_axis_start is None:
            self.x_axis_start = x_start - 5
        else:
            self.x_axis_start = x_axis_start
        if x_axis_end is None:
            self.x_axis_end = x_start + 5
        else:
            self.x_axis_end = x_axis_end
        if y_axis_start is None:
            self.y_axis_start = y_start - 5
        else:
            self.y_axis_start = y_axis_start
        if y_axis_end is None:
            self.y_axis_end = y_start + 5
        else:
            self.y_axis_end = y_axis_end

        self.optimizers = {}
        if gd_active:
            self.optimizers['GD'] = GD_Optimizer(gd_learning_rate, gd_learning_rate_decay, x_start, y_start, loss_function)
        if momentum_active:
            self.optimizers['Momentum'] = Momentum_Optimizer(momentum_learning_rate, momentum_learning_rate_decay, momentum_velocity_loss_rate, x_start, y_start, loss_function)
        if nesterov_active:
            self.optimizers['Nesterov'] = Nesterov_Optimizer(nesterov_learning_rate, nesterov_learning_rate_decay, nesterov_velocity_loss_rate, x_start, y_start, loss_function)
        if rms_prop_active:
            self.optimizers['RMS_Prop'] = RMS_Prop_Optimizer(rms_prop_learning_rate, rms_prop_learning_rate_decay, rms_prop_squared_grad_forget_rate, x_start, y_start, loss_function)
        if adam_active:
            self.optimizers['Adam'] = Adam_Optimizer(adam_learning_rate, adam_learning_rate_decay, adam_exponential_decay_rate_first_moment, adam_exponential_decay_rate_second_moment, x_start, y_start, loss_function)
    
    def plot(self) -> None:
        now = datetime.now()
        start_time = now.strftime('%Y-%m-%d(%H-%M)')
        os.mkdir(f'figures/{start_time}')
        self.create_text_file(start_time)
        self.set_pyplot_settings()
        plot_cache = [None] * len(self.optimizers)
        for iteration in range(self.iterations):
            for index, optimizer in enumerate(self.optimizers.values()):
                optimizer.calculate_step(iteration)
                self.ax.plot([optimizer.x_last, optimizer.x_current], [optimizer.y_last, optimizer.y_current], [optimizer.z_last, optimizer.z_current], linewidth=0.3, color=optimizer.color, alpha = 1.0)
                if plot_cache[index]:
                    plot_cache[index].remove()
                plot_cache[index] = self.ax.scatter(optimizer.x_current, optimizer.y_current, optimizer.z_current, s = 1, depthshade=True, color=optimizer.color)  
                if iteration == 0:
                    plt.legend(plot_cache, self.optimizers.keys())       
                plt.pause(self.iteration_pause)
            plt.savefig(f'figures/{start_time}/' + start_time + '-' + str(iteration) + '.png')
        GifMaker.makegif(self.iterations, start_time)


    def set_pyplot_settings(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize = (3, 2), dpi = 300)
        self.ax = self.fig.add_subplot(111, projection='3d', computed_zorder=False)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        params = {'legend.fontsize': 3,
                'legend.handlelength': 3}
        plt.rcParams.update(params)
        x_val = np.linspace(self.x_axis_start, self.x_axis_end, 5000, dtype=np.float32)
        y_val = np.linspace(self.y_axis_start, self.y_axis_end, 5000, dtype=np.float32)
        x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
        x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
        y_val_mesh_flat = y_val_mesh.reshape([-1, 1])
        z_val_mesh_flat = Optimizer.loss_function_tensor(torch.tensor(x_val_mesh_flat), torch.tensor(y_val_mesh_flat), self.loss_function)
        z_val_mesh = z_val_mesh_flat.reshape(x_val_mesh.shape)
        self.ax.plot_surface(x_val_mesh, y_val_mesh, z_val_mesh, alpha=.8, cmap=cm.cividis)
        self.ax.tick_params(axis='both', which='major', labelsize=6)
        self.ax.tick_params(axis='both', which='minor', labelsize=6)
        plt.draw()

    def create_text_file(self, start_time:str) -> None:
        optimizer_string = ''
        if "GD" in self.optimizers.keys():
            gd_learning_rate = self.optimizers["GD"].learning_rate
            gd_learning_rate_decay = self.optimizers["GD"].learning_rate_decay
            optimizer_string += f" GD: learning_rate: {gd_learning_rate}, learning_rate_decay: {gd_learning_rate_decay}\n"
        if "Momentum" in self.optimizers.keys():
            momentum_learning_rate = self.optimizers["Momentum"].learning_rate
            momentum_learning_rate_decay = self.optimizers["Momentum"].learning_rate_decay
            momentum_velocity_loss_rate = self.optimizers["Momentum"].velocity_loss_rate
            optimizer_string += f" Momentum: learning_rate: {momentum_learning_rate}, learning_rate_decay: {momentum_learning_rate_decay}, velocity_loss_rate = {momentum_velocity_loss_rate}\n"
        if "Nesterov" in self.optimizers.keys():
            nesterov_learning_rate = self.optimizers["Nesterov"].learning_rate
            nesterov_learning_rate_decay = self.optimizers["Nesterov"].learning_rate_decay
            nesterov_velocity_loss_rate = self.optimizers["Nesterov"].velocity_loss_rate
            optimizer_string += f" Nesterov: learning_rate: {nesterov_learning_rate}, learning_rate_decay: {nesterov_learning_rate_decay}, velocity_loss_rate = {nesterov_velocity_loss_rate}\n"
        if "RMS_Prop" in self.optimizers.keys():
            rms_prop_learning_rate = self.optimizers["RMS_Prop"].learning_rate
            rms_prop_learning_rate_decay = self.optimizers["RMS_Prop"].learning_rate_decay
            rms_prop_squared_grad_forget_rate = self.optimizers["RMS_Prop"].squared_grad_forget_rate
            optimizer_string += f" RMS Prop: learning_rate: {rms_prop_learning_rate}, learning_rate_decay: {rms_prop_learning_rate_decay}, squared_grad_forget_rate = {rms_prop_squared_grad_forget_rate}\n"
        if "Adam" in self.optimizers.keys():
            adam_learning_rate = self.optimizers["Adam"].learning_rate
            adam_learning_rate_decay = self.optimizers["Adam"].learning_rate_decay
            adam_exponential_decay_rate_first_moment = self.optimizers["Adam"].exponential_decay_rate_first_moment
            adam_exponential_decay_rate_second_moment = self.optimizers["Adam"].exponential_decay_rate_second_moment
            optimizer_string += f" Adam: learning_rate: {adam_learning_rate}, learning_rate_decay: {adam_learning_rate_decay}, exponential_decay_rate_first_moment = {adam_exponential_decay_rate_first_moment}, adam_exponential_decay_rate_second_moment = {adam_exponential_decay_rate_second_moment}\n"

        try:
            loss_function_string = getsource(self.loss_function)
        except:
            loss_function_string = f"loss_function could not converted to string\n"

        with open(f'figures/{start_time}/param.txt', 'w') as f:
            f.write(f"Those are the parameters used for the corresponding plots:\n {loss_function_string} x_start: {self.x_start}, y_start: {self.y_start}\n iterations: {self.iterations}, iteration_pause: {self.iteration_pause}\n x_axis: [{self.x_axis_start, self.x_axis_end}]\n y_axis: [{self.y_axis_start, self.y_axis_end}]\n" + optimizer_string)
        
