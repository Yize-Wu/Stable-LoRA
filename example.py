from torch import nn

model: nn.Module # model with lora adapter inserted

# get param A and param B
params_A = []
params_B = []
params_dynamic_scales = []
for name,parameter in model.named_parameters():
    if parameter.requires_grad is True:
        if 'lora_A' in name:
            params_A.append(parameter)
        elif 'lora_B' in name:
            params_B.append(parameter)

shrink_ratio = 0.0005
beta1, beta2 = 0.9, 0.999
lr = 1e-4
weight_decay = 0.01
from stable_lora import StableLora
param_groups = [{"params": params_A}, {"params": params_B}]
optimizers = (
    StableLora(param_groups, shrink_ratio=shrink_ratio, betas=(beta1,beta2), lr=lr, weight_decay=weight_decay,
            ),
    None # Your learning rate scheduler here
)
