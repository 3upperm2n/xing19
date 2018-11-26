#!/usr/bin/env python

import os,sys
import operator, copy, random, time, ctypes
import numpy as np
import socket # hostname

import multiprocessing as mp
from multiprocessing import Process, Lock, Manager, Value, Pool

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("[server]")

# read app info
sys.path.append('../prepare')
from app_info import * 

# read common api
sys.path.append('../utils')
from magic_common import * 

lock = Lock()
manager = Manager()


#-----------------------------------------------------------------------------#
# arguments 
#-----------------------------------------------------------------------------#
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-c', dest='maxCoRun', default=1,        help='max collocated jobs per gpu')
parser.add_argument('-s', dest='seed',     default=31415926, help='random seed for shuffling the app order')
parser.add_argument('-f', dest='ofile',    default=None,     help='output file to save the app timing trace')
args = parser.parse_args()


#=============================================================================#
# main program
#=============================================================================#
def main():
    timer = Timer()
    timer.start()

    #--------------------------------------------------------------------------
    # input: app, app2dir_dd in app_info.py
    #--------------------------------------------------------------------------
    if len(app) <> len(app2dir_dd):
        print "Error: app number wrong, check ../prepare/app_info.py!"
        sys.exit(1)

    #print app
    for appName, appDir in app2dir_dd.iteritems():
        print appName
        gen_program_flow(appDir)
        break



    ## step 1:
    ## obtain host name, gpuID
    ## create the corresponding folder (no need)
    #hostName = socket.gethostname()
    #foldName = str(hostName) + '-gpu' + str(gpu2Use)
    #path = os.getcwd()
    #targetDir = path + "/" + foldName

    ##if not os.path.exists(targetDir):
    ##    try:
    ##        os.mkdir(targetDir)
    ##    except OSError:
    ##        print("creation of %s failed!" % targetDir)
       






    #----------------------------------------------------------------------
    #
    #----------------------------------------------------------------------




    # three random sequences
    #app_s1 = genRandSeq(app, seed=RANDSEED) # pi

    ##apps_num = len(app)
    ##logger.debug("Total GPU Applications = {}.".format(apps_num))

    #--------------------------------------------------------------------------
    # dedicated runtime
    #--------------------------------------------------------------------------

    ### exclude: rodinia-heartwall, lonestar_sssp, dmr
    ##appPool = app
    ##
    ##app_dedicate_dd = {}
    ##ofile = foldName + ".npy"



    ##for idx, app1 in enumerate(appPool):
    ##    #ofile = targetDir + '/' + app1 + ".npy"
    ##    #print ofile
    ##    print("idx={}, appName={}".format(idx, app1))

    ##    if idx >= 0: # NOTE: modify if program hangs
    ##        app1_runtime = []

    ##        # run N times, get the fastest
    ##        for ii in xrange(3):
    ##            workers = [] # for mp processes

    ##            jobID = 0
    ##            id2name[jobID] = app1 
    ##            process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[app1]))
    ##            process.daemon = False
    ##            workers.append(process)
    ##            process.start()

    ##            for p in workers: p.join()

    ##            total_jobs = 1
    ##            appRT_dd = getGpuJobTiming(AppStat, total_jobs, id2name)
    ##            #print appRT_dd

    ##            app1_runtime.append(appRT_dd[app1])

    ##        #print app1_runtime
    ##        app1_best = min(app1_runtime)
    ##        #print app1_best

    ##        # NOTE: only remember the app1 runtime  when running with app2
    ##        app_dedicate_dd[app1] = app1_best
    ##        #break

    ###--- end of 1st for loop ---# 
    ##np.save(ofile, app_dedicate_dd)
    
    timer.end()
    print("total elapsed time = {} (s)".format(timer.total()))

if __name__ == "__main__":
    main()
