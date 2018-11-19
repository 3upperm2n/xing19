#!/usr/bin/env python
import os,sys
import numpy as np

sys.path.append('../prepare')
from app_info import *  # app and app2dir_dd

#sys.path.append('../utils')
#from magic_common import cd


folders = ['./homedesktop-gpu0', './homedesktop-gpu1', 
        './hoyi-gpu0', './hoyi-gpu1']

def main():

    print("Total GPU applications = {}".format(len(app)))

    for fold in folders:
        print("---------------------------------------------")
        print("{}".format(fold))
        print("---------------------------------------------")
        dedicate_file = "../02_dedicate/" + fold[2:] + ".npy"
        dedicate_dd = np.load(dedicate_file).item()
        for name in app:
            # step1: read dedicated runtime for the app
            dedicate_rt = dedicate_dd[name] 

            # step2: read the perf for run2 cases
            fullpath = str(fold) + "/" + str(name) + ".npy"
            #print fullpath
            
            app_run2_dd = np.load(fullpath).item()

            # step3: compute the slowdown for all the combinations
            slowdown_list = []
            for key, corun_rt in app_run2_dd.iteritems():
                #print key, corun_rt 
                slowdown = dedicate_rt / corun_rt 
                slowdown_list.append(slowdown)

            # step4: compute avg slowdown
            avg_slowdown = sum(slowdown_list) / len(slowdown_list)

            if avg_slowdown >=1:
                print("[Warning] {0:<30} \t {1:.3f}".format(name, avg_slowdown))
            else:
                print("{0:<30} \t {1:.3f}".format(name, avg_slowdown))

            #break

        print("---------------------------------------------")
        break
        



if __name__ == "__main__":
    main()
