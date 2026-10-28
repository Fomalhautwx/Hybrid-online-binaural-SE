from ptflops import get_model_complexity_info
from torchinfo import summary
from torch.utils.flop_counter import FlopCounterMode
from models.Paris_enhancement import GTCRN_PARIS
from models.gtcrn_binaural import GTCRN_Binaural
import torch
import yaml

with open("/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    # model = GTCRN_PARIS(num_channels=32, mask_type="mono", bss=None, params=
    #                     config["network"])
model = GTCRN_Binaural(num_channels=32)
model.eval()

x = torch.randn(16, 2, 257, 160, dtype=torch.complex64)
enh = model(x)
print(enh.shape)

#flops, params = get_model_complexity_info(model, (2, 16000), as_strings=True,
                                            #print_per_layer_stat=False, verbose=True)
#print(flops, params)

summary(model,
    input_data=[x],
    depth=3,
    verbose=1,
    col_width=16,   
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"])