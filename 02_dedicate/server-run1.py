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

#
# gpu to use
#
#gpu2Use=0
gpu2Use=1

#-----------------------------------------------------------------------------#
# arguments 
#-----------------------------------------------------------------------------#
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-c', dest='maxCoRun', default=1,        help='max collocated jobs per gpu')
parser.add_argument('-s', dest='seed',     default=31415926, help='random seed for shuffling the app order')
parser.add_argument('-f', dest='ofile',    default=None,     help='output file to save the app timing trace')
args = parser.parse_args()

#-----------------------------------------------------------------------------#
# Run incoming workload
#-----------------------------------------------------------------------------#
def run_work(jobID, AppStat, appDir):

    #
    #    jobid      gpu     status      starT       endT
    #

    AppStat[jobID, 0] = jobID 
    # avoid tagging gpu since we simulate with 1 gpu
    AppStat[jobID, 2] = 0 

    # run the application 
    #[startT, endT] = run_remote(app_dir=appDir, devid=0)
    [startT, endT] = run_remote(app_dir=appDir, devid=gpu2Use)

    # logger.debug("jodID:{0:5d} \t start: {1:.3f}\t end: {2:.3f}\t duration: {3:.3f}".format(jobID, startT, endT, endT - startT))


    #=========================#
    # update gpu job table
    #
    # 5 columns:
    #    jobid      gpu     starT       endT
    #=========================#
    # mark the job is done, and update the timing info
    AppStat[jobID, 2] = 1   # done
    AppStat[jobID, 3] = startT 
    AppStat[jobID, 4] = endT 


#=============================================================================#
# main program
#=============================================================================#
def main():
    timer = Timer()
    timer.start()

    # step 1:
    # obtain host name, gpuID
    # create the corresponding folder (no need)
    hostName = socket.gethostname()
    foldName = str(hostName) + '-gpu' + str(gpu2Use)
    path = os.getcwd()
    targetDir = path + "/" + foldName

    #if not os.path.exists(targetDir):
    #    try:
    #        os.mkdir(targetDir)
    #    except OSError:
    #        print("creation of %s failed!" % targetDir)
       




    MAXCORUN = int(args.maxCoRun)    # max jobs per gpu
    RANDSEED = int(args.seed)
    gpuNum = 1

    #logger.debug("MaxCoRun={}\trandseed={}\tsaveFile={}".format(MAXCORUN, RANDSEED, args.ofile))

    #----------------------------------------------------------------------
    # 1) application status table : 5 columns
    #----------------------------------------------------------------------
    #
    #    jobid      gpu     status      starT       endT    
    #       0       0           1       1           2
    #       1       1           1       1.3         2.4
    #       2       0           0       -           -
    #       ...
    #----------------------------------------------------------------------
    maxJobs = 10000
    rows, cols = maxJobs, 5  # note: init with a large prefixed table
    d_arr = mp.Array(ctypes.c_double, rows * cols)
    arr = np.frombuffer(d_arr.get_obj())
    AppStat = arr.reshape((rows, cols))

    id2name = {}

    #----------------------------------------------------------------------
    #
    #----------------------------------------------------------------------



    #--------------------------------------------------------------------------
    # input: app, app2dir_dd in app_info.py
    #--------------------------------------------------------------------------
    if len(app) <> len(app2dir_dd):
        print "Error: app number wrong, check ../prepare/app_info.py!"
        sys.exit(1)

    # three random sequences
    #app_s1 = genRandSeq(app, seed=RANDSEED) # pi

    apps_num = len(app)
    logger.debug("Total GPU Applications = {}.".format(apps_num))

    #--------------------------------------------------------------------------
    # dedicated runtime
    #--------------------------------------------------------------------------

    # exclude: rodinia-heartwall, lonestar_sssp, dmr
    appPool = app
    
    app_dedicate_dd = {}
    ofile = foldName + ".npy"



    for idx, app1 in enumerate(appPool):
        #ofile = targetDir + '/' + app1 + ".npy"
        #print ofile
        print("idx={}, appName={}".format(idx, app1))

        if idx >= 0: # NOTE: modify if program hangs
            app1_runtime = []

            # run N times, get the fastest
            for ii in xrange(3):
                workers = [] # for mp processes

                jobID = 0
                id2name[jobID] = app1 
                process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[app1]))
                process.daemon = False
                workers.append(process)
                process.start()

                for p in workers: p.join()

                total_jobs = 1
                appRT_dd = getGpuJobTiming(AppStat, total_jobs, id2name)
                #print appRT_dd

                app1_runtime.append(appRT_dd[app1])

            #print app1_runtime
            app1_best = min(app1_runtime)
            #print app1_best

            # NOTE: only remember the app1 runtime  when running with app2
            app_dedicate_dd[app1] = app1_best
            #break

    #--- end of 1st for loop ---# 
    np.save(ofile, app_dedicate_dd)
    
    timer.end()
    print("total elapsed time = {} (s)".format(timer.total()))

if __name__ == "__main__":
    main()
