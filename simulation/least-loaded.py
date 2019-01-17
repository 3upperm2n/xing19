#!/usr/bin/env python

import os,sys
import operator, copy, random, time, ctypes
import numpy as np
import pandas as pd

import multiprocessing as mp
from multiprocessing import Process, Lock, Manager, Value, Pool

import pickle
from sklearn.externals import joblib  # to save/load model to disk


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
parser.add_argument('-g', dest='gpuNum',   default=2,        help='GPUs on current system')
parser.add_argument('-c', dest='maxCoRun', default=2,        help='max collocated jobs per gpu')
parser.add_argument('-s', dest='seed',     default=31415926, help='random seed for shuffling the app order')
parser.add_argument('-f', dest='ofile',    default=None,     help='output file to save the app timing trace')
args = parser.parse_args()

#-----------------------------------------------------------------------------#
# select which gpu to run 
#-----------------------------------------------------------------------------#
def select_gpu(gpuStat):
    gpuNum = len(gpuStat)

    node_dd = {}
    for gid, activeJobs in enumerate(gpuStat):
        node_dd[gid] = activeJobs

    # sort dd by value  =>  (gpuID, jobs) ascending order
    sorted_x = sorted(node_dd.items(), key=operator.itemgetter(1))

    (TargetDev, _) = sorted_x[0]

    return TargetDev


#-----------------------------------------------------------------------------#
# Check Dispatch or Not 
#-----------------------------------------------------------------------------#
def has_slot(gpuStat, MAXCORUN):
    gpuNum = len(gpuStat)

    gpuFull = 0
    for activeJobs in gpuStat: # gpuStat keeps track of active jobs
        if activeJobs >= MAXCORUN: 
            gpuFull += 1

    #print("gpuStat: {}, {}".format(gpuStat[0], gpuStat[1]))
    #print("gpuFull={}".format(gpuFull))

    if gpuFull == gpuNum: # all nodes are fully loaded 
        Dispatch = False 
    else:
        Dispatch = True

    return Dispatch

#-----------------------------------------------------------------------------#
# check work queue empty or not
#-----------------------------------------------------------------------------#
def hasworkloads(gpuWorkq):
    haswork = False
    if len(gpuWorkq) > 0:
        haswork = True 
    return haswork


#-----------------------------------------------------------------------------#
#
#-----------------------------------------------------------------------------#
def predict_appclass(app2metric, bestmodel):
    app2class_dd = {}
    for cur_app, metric in app2metric.iteritems():
        metric = metric.values.reshape(-1, metric.shape[0]) # col to row
        #print metric.shape
        app_class = bestmodel.predict(metric) # 1 as robust , 0 as sensitive
        # print cur_app, app_class[0]
        app2class_dd[cur_app] = app_class[0]
        #break
    return app2class_dd


#-----------------------------------------------------------------------------#
# Run incoming workload
#-----------------------------------------------------------------------------#
def run_work(jobID, AppStat, appDir, targetGPU):

    #
    #    jobid      gpu     status      starT       endT
    #

    AppStat[jobID, 0] = jobID 
    # avoid tagging gpu since we simulate with 1 gpu
    AppStat[jobID, 1] = targetGPU 

    # run the application 
    [startT, endT] = run_remote(app_dir=appDir, devid=targetGPU)

    logger.debug("jodID:{0:5d} \t gpu: {1:5d} \t start: {2:.3f}\t end: {3:.3f}\t duration: {4:.3f}".format(jobID, targetGPU, startT, endT, endT - startT))


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

    MAXCORUN = int(args.maxCoRun)    # max jobs per gpu
    RANDSEED = int(args.seed)
    gpuNum   = int(args.gpuNum) 

    logger.debug("GPUs(-g)={}\tMaxCoRun(-c)={}\trandseed(-s)={}\tsaveFile(-f)={}".format(
        gpuNum, MAXCORUN, RANDSEED, args.ofile))

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
    # 2) gpu node status: 1 columns
    #----------------------------------------------------------------------
    #
    #    GPU_Node(rows)     ActiveJobs
    #       0               0
    #       1               0
    #       2               0
    #       ...
    #----------------------------------------------------------------------
    #GpuStat = manager.dict()
    #for i in xrange(gpuNum):
    #    GpuStat[i] = 0
    gpuStat = [0 for i in xrange(gpuNum)]


    #--------------------------------------------------------------------------
    # input: app, app2dir_dd in app_info.py
    #--------------------------------------------------------------------------
    if len(app) <> len(app2dir_dd):
        print "Error: app number wrong, check ../prepare/app_info.py!"
        sys.exit(1)

    #
    # randomize the input sequences
    #
    app_s1 = genRandSeq(app, seed=RANDSEED)

    apps_num = len(app)
    logger.debug("Total GPU Workloads = {}.".format(apps_num))

    gpuWorkq = copy.deepcopy(app_s1) 

    #--------------------------------------------------------------------------
    # Run
    #--------------------------------------------------------------------------
    jobID = -1
    workers = [] # for mp processes
    current_jobid_list = [] # keep track of current application 

    while hasworkloads(gpuWorkq):
        Dispatch = has_slot(gpuStat,MAXCORUN)

        if Dispatch:
            targetGPU = select_gpu(gpuStat)

            workloadName = gpuWorkq[0]
            gpuWorkq.remove(workloadName)

            gpuStat[targetGPU] += 1 # increase the active jobs on the target
            jobID += 1
            id2name[jobID] = workloadName 
            current_jobid_list.append(jobID)

            process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[workloadName], targetGPU))
            process.daemon = False
            workers.append(process)
            process.start()

        else:
            # spinning, waiting for a free spot
            while True:
                break_loop = False

                #current_running_jobs = 0
                jobs2del = []

                # figure out the jobs that have ended 
                for jid in current_jobid_list:
                    if AppStat[jid, 2] == 1: # check the status, if one is done
                        jobs2del.append(jid)
                        break_loop = True
                        break

                if break_loop:
                    for id2del in jobs2del:
                        current_jobid_list.remove(id2del) # del ended jobs 
                        gpuInUse = int(AppStat[id2del, 1])
                        gpuStat[gpuInUse] -= 1 # update gpu active jobs

                    # break the spinning
                    break

            #------------------------------------
            # after spinning, schedule the work
            #------------------------------------
            targetGPU = select_gpu(gpuStat)

            workloadName = gpuWorkq[0]
            gpuWorkq.remove(workloadName)

            gpuStat[targetGPU] += 1 # increase the active jobs on the target
            jobID += 1
            id2name[jobID] = workloadName 
            current_jobid_list.append(jobID)

            process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[workloadName], targetGPU))
            process.daemon = False
            workers.append(process)
            process.start()

        #break
        #if jobID == 10: break


    #=========================================================================#
    # end of running all the jobs
    #=========================================================================#
    for p in workers:
        p.join()

    total_jobs = jobID + 1
    if total_jobs <> apps_num:
        logger.debug("[Warning] job number doesn't match.")

    
    # print out / save trace table 
    if args.ofile:
        PrintGpuJobTable(AppStat, total_jobs, id2name, saveFile=args.ofile)
    else:
        PrintGpuJobTable(AppStat, total_jobs, id2name)



if __name__ == "__main__":
    main()
