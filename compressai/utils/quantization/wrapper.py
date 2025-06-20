import torch
import torch.nn as nn
import random

from aimet_torch.quantsim import QuantizationSimModel

random.seed(4040)

MAX_BUFFER_SIZE = 100000000

class InputLoggerWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.inputs = []

        #Buffer Management
        self.counter = 0
        self.freeze_pos = 0
        self.freeze_idx = 0

    def forward(self, *args, **kwargs):        
        with torch.no_grad():
            self.append_input((args, kwargs))            

        return self.model(*args, **kwargs)

    
    def forward_part1(self, *args, **kwargs):
        with torch.no_grad():
            self.append_input((args, kwargs))
        return self.model.forward_part1(*args, **kwargs)
    
    def forward_part2(self, *args, **kwargs):
        return self.model.forward_part2(*args, **kwargs)
        
    
    def append_input(self, input):
        # print("append")
        if len(self.inputs) == 5000 :
            print("Reached 5000 inputs!")

        if len(self.inputs) < MAX_BUFFER_SIZE :            
            self.inputs.append(input)
        else:
            self.counter += 1
            if (self.counter == MAX_BUFFER_SIZE):
                self.counter = 0
                if (self.freeze_idx) < 9:
                    self.freeze_idx += 1
                    self.freeze_pos = int(float(self.freeze_idx / 10) * MAX_BUFFER_SIZE)
                    print(self.freeze_pos)
                

            random_index = random.randint(self.freeze_pos, len(self.inputs) - 1)
            self.inputs[random_index] = input

    def get_logged_inputs(self):
        return self.inputs

    def clear_logged_inputs(self):
        self.inputs = []


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)
    

class SIMWrapper(nn.Module):
    def __init__(self, module : QuantizationSimModel):
        super().__init__()
        self.sim_module = module

    def forward(self, *args, **kwargs): 
        return self.sim_module.model(*args, **kwargs)
    
    def forward_part1(self, *args, **kwargs):        
        return self.sim_module.model.forward_part1(*args, **kwargs)
    
    def forward_part2(self, *args, **kwargs):
        return self.sim_module.model.forward_part2(*args, **kwargs)