import os
import sys
print(os.getcwd())
print(sys.path)
# import evaluate.context
from project.mask_propagation import MaskPropagationModule
import numpy as np

tst = np.empty((3000, 2000, 3), dtype=np.float32)

mp = MaskPropagationModule('./pwc_net/pwc_net/pwc_net.pth.tar')
mp.infer_mask(tst, tst, None)
