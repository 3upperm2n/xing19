#!/usr/bin/env python

import sys
import numpy as np

dd_gpu0 = np.load('../02_dedicate/homedesktop-gpu0.npy').item()
dd_gpu1 = np.load('../02_dedicate/homedesktop-gpu1.npy').item()

if len(dd_gpu0.keys()) <> len(dd_gpu1.keys()):
    print("keys number does not matchi")
    sys.exit(1)

app_fastdev = {} # dict
for k, rt0 in dd_gpu0.items():
    rt1 = dd_gpu1[k]
    select_dev = 1 if rt1 < rt0 else 0 
    print("app={0:>35}, \tgpu0={1:8.2f}, \tgpu1={2:8.2f}, \t select={3:3d}".format(k, rt0, rt1, select_dev))
    # save to dd
    app_fastdev[k] = select_dev

# save to a file
np.save('home-fastdev.npy',app_fastdev)
