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
# check work queue empty or not
#-----------------------------------------------------------------------------#
def hasworkloads(gpuWorkq):
    haswork = False
    for i in gpuWorkq:
        if len(i) > 0:
            haswork = False
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
def run_work(jobID, AppStat, appDir):

    #
    #    jobid      gpu     status      starT       endT
    #

    AppStat[jobID, 0] = jobID 
    # avoid tagging gpu since we simulate with 1 gpu
    AppStat[jobID, 2] = 0 

    # run the application 
    [startT, endT] = run_remote(app_dir=appDir, devid=0)

    logger.debug("jodID:{0:5d} \t start: {1:.3f}\t end: {2:.3f}\t duration: {3:.3f}".format(jobID, startT, endT, endT - startT))


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


    #--------------------------------------------------------------------------
    # 3) model for predicting the best gpu to use
    #--------------------------------------------------------------------------
    logger.debug("Loading a Neural Net model to predict best GPU to use.")

    # metrics
    df_app_metrics = pd.read_csv('../07_nn/appmetrics_with_appname.csv')
    df_app_metrics = df_app_metrics.drop(df_app_metrics.columns[0], axis=1) # drop the 1st column

    # delete rodinia_heartwall
    df_app_metrics.drop(df_app_metrics[df_app_metrics.AppName == 'rodinia_heartwall'].index, inplace=True)

    df_rows = df_app_metrics.shape[0]
    logger.debug("Total profiling metrics = {}.".format(df_rows))

    #--------------------------------------------------------------------------
    # find out the mismatch
    #--------------------------------------------------------------------------
    count_same = 0
    count_diff = 0
    if df_rows > apps_num:
        print "\n[Warning] input metrics has more apps than the input sequence"
        app_in_df = list(df_app_metrics['AppName'])
        for i in app_in_df:
            if i not in app_s1:
                print("[Warning] {} not in app_s1".format(i))
        print "[Warning] Please fix the error before running!"
        sys.exit(1)


    if df_rows < apps_num:
        print "\n[Warning] input metrics has fewer apps than the input sequence"
        app_in_df = list(df_app_metrics['AppName'])
        for i in app_s1:
            if i not in app_in_df:
                print("[Warning] not in input metrics.".format(i))
        print "[Warning] Please fix the error before running!"
        sys.exit(1)

    if df_rows == apps_num:        
        if set(app_s1) == set(list(df_app_metrics['AppName'])):
            logger.debug("Great! The app lists match.")
        else:
            logger.debug("Bummer! The app lists between input sequence and metrics are not equal. Please fix the error.")
            sys.exit(1)

    #--------------------------------------------------------------------------
    # load trained model 
    #--------------------------------------------------------------------------
    nn_model = pickle.load(open('../07_nn/output_model.pkl', 'rb'))

    #
    # predict the best device to use  / assign to each GPU work queue
    #

    gpuWorkq = [ [] for i in xrange(gpuNum) ]
    app_dev = {}
    for appname in app_s1:
        df_current_app = df_app_metrics.loc[df_app_metrics['AppName'] == appname]
        df_current_app = df_current_app.drop(df_current_app.columns[0], axis=1) # drop the 1st column : "AppName"
        targetdev = nn_model.predict(df_current_app) 
        targetdev = int(targetdev[0])
        app_dev[appname] = targetdev 
        #print("{0:<40}:\t {1:2d}".format(appname, targetdev[0]))
        gpuWorkq[targetdev].append(appname)

    #print gpuWorkq[0]
    #print "\n----------\n"
    #print gpuWorkq[1]
    #print "\n----------\n"

    #--------------------------------------------------------------------------
    # 4) model for interference analysis 
    #--------------------------------------------------------------------------
    logger.debug("Loading model to predict co-running interference.")

    app2metric = np.load('../prepare/app2metric_featAll.npy').item()  # featAll
    bestmodel = joblib.load('../00_classification_interference/featall_bestmodel.pkl') # load model, predict app class 

    app2class_dd = predict_appclass(app2metric, bestmodel) 

    #print app2class_dd

    #--------------------------------------------------------------------------
    # 5) prioritize the interference-insensitive workloads
    #--------------------------------------------------------------------------

    gpuWorkq_new = [ [] for i in xrange(gpuNum) ]
    for gid, que in enumerate(gpuWorkq):
        #print gid, que
        robust_list = []
        sensitive_list = []
        for curapp in que:
            if app2class_dd[curapp] == 0:
                #print("sensitive : {}".format(curapp))
                sensitive_list.append(curapp)
            else:
                #print("in-sensitive : {}".format(curapp))
                robust_list.append(curapp)
        gpuWorkq_new[gid].extend(robust_list)
        gpuWorkq_new[gid].extend(sensitive_list)

    # update work queue order
    gpuWorkq = gpuWorkq_new


    #--------------------------------------------------------------------------
    # Run
    #--------------------------------------------------------------------------
    def select_gpu(gpuStat, gpuWorkq, MAXCORUN):
        gpuNum = len(gpuStat)

        gpuFull = 0
        for gid, activeJobs in enumerate(gpuStat): # gpuStat keeps track of active jobs
            if activeJobs >= MAXCORUN: 
                gpuFull += 1

        if gpuFull == gpuNum: # all nodes are fully loaded 
            Dispatch, targetGPU, workloadName = False, None, None
            return Dispatch, targetGPU, workloadName


        # when there is free slot




    while hasworkloads(gpuWorkq):
        Dispatch, targetGPU, workloadName = select_gpu(gpuStat, gpuWorkq, MAXCORUN)




    #--------------------------------------------------------------------------
    # Run
    #--------------------------------------------------------------------------

    ##appQueList = app_s1 
    #appQueList = new_seq 

    #workers = [] # for mp processes


    ##==================================#
    ## run the apps in the queue 
    ##==================================#
    #activeJobs = 0
    #jobID = -1

    #current_jobid_list = [] # keep track of current application 

    #for i in xrange(apps_num):
    #    Dispatch = True if activeJobs < MAXCORUN else False 
    #    #print("iter {} dispatch={}".format(i, Dispatch))

    #    if Dispatch:
    #        activeJobs += 1
    #        jobID += 1
    #        current_jobid_list.append(jobID)

    #        appName = appQueList[i] 
    #        id2name[jobID] = appName
    #        process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[appName]))

    #        process.daemon = False
    #        #logger.debug("Start %r", process)
    #        workers.append(process)
    #        process.start()

    #    else:
    #        # spin
    #        while True:
    #            break_loop = False

    #            current_running_jobs = 0
    #            jobs2del = []

    #            for jid in current_jobid_list:
    #                if AppStat[jid, 2] == 1: # check the status, if one is done
    #                    jobs2del.append(jid)
    #                    break_loop = True

    #            if break_loop:
    #                activeJobs -= 1

    #                # update
    #                if jobs2del:
    #                    for id2del in jobs2del:
    #                        del_idx = current_jobid_list.index(id2del)
    #                        del current_jobid_list[del_idx]
    #                break

    #        #------------------------------------
    #        # after spinning, schedule the work
    #        #------------------------------------

    #        #print("iter {}: activeJobs = {}".format(i, activeJobs))
    #        activeJobs += 1
    #        jobID += 1
    #        current_jobid_list.append(jobID)
    #        #print("iter {}: activeJobs = {}".format(i, activeJobs))

    #        appName = appQueList[i] 
    #        id2name[jobID] = appName
    #        process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[appName]))

    #        process.daemon = False
    #        workers.append(process)
    #        process.start()


    ##=========================================================================#
    ## end of running all the jobs
    ##=========================================================================#
    #for p in workers:
    #    p.join()

    #total_jobs = jobID + 1
    #if total_jobs <> apps_num:
    #    logger.debug("[Warning] job number doesn't match.")

    #
    ## print out / save trace table 
    #if args.ofile:
    #    PrintGpuJobTable(AppStat, total_jobs, id2name, saveFile=args.ofile)
    #else:
    #    PrintGpuJobTable(AppStat, total_jobs, id2name)


if __name__ == "__main__":
    main()
