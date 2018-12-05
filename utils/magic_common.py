#!/usr/bin/env python

import os,sys
import random
import time
import re

import subprocess
from subprocess import check_call, STDOUT, CalledProcessError,call
DEVNULL = open(os.devnull, 'wb', 0)  # no std out


import pandas as pd
import numpy as np
#from math import *
#import operator


## cudaMemcpyHostToDevice 
## cudaMemcpy2D
#CudaAPIs=['cudaMalloc', 'cudaFree', 'cudaMemcpy', "<<<", "cudaMemcpy2D",
#        "cufft", "cufftDestroy", "cufftXtSetCallback"]
#
## ./common/inc/dynlink/cuda_drvapi_dynlink_cuda.h
#tcuTexRefSetFilterMode
#
#
#tcuMemcpyHtoA
#tcuMemcpyAtoA
#tcuMemcpy3D
#tcuMemcpy2DUnaligned
#tcuMemcpyHtoDAsync
#
#tcuMemAlloc
#
#tcuMemAllocPitch
#tcuMemcpyPeer
#tcuD3D9Begin
#
##./common/inc/dynlink/cuda_drvapi_dynlink_d3d.h:typedef
#tcuD3D9ResourceGetMappedArray
#
#
#cudaMemset
#cudaFreeHost
#
#
#cudaEventSynchronize
#cudaEventRecord
#cudaEventDestroy
#
#cudaDeviceSynchronize
#
#cudaBindTexture
#cudaBindTexture2D
#cudaBindTextureToArray
#
#
#cufftPlan2d
#cufftExecC2C
#
#
#cudaStream_t
#cudaStreamCreate
#cudaStreamWaitEvent
#






#-----------------------------------------------------------------------------#
# shuffle the input app order 
#-----------------------------------------------------------------------------#
def genRandSeq(applist, seed = 99999999):
    random.seed(seed)
    appNum = len(applist)
    idx = [ i for i in xrange(0, appNum)]
    random.shuffle(idx)
    applist_new = [applist[i] for i in idx] 
    return applist_new

#-----------------------------------------------------------------------------#
# go to target dir for each process 
#-----------------------------------------------------------------------------#
class cd:
    """
    Context manager for changing the current working directory
    """

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

#-----------------------------------------------------------------------------#
# measure the elapsed time
#-----------------------------------------------------------------------------#
class Timer:
    """
    get the timer
    """
    def __init__(self):
        self.startT = 0.
        self.endT   = 0.
        self.totRT = 0.

    def start(self):
        self.startT = time.time()

    def end(self):
        self.endT = time.time()

    def total(self):
        return self.endT - self.startT
        

#-----------------------------------------------------------------------------#
# Run incoming workload
#-----------------------------------------------------------------------------#
def run_remote(app_dir, devid=0):
    cmd_str = "./run.sh " + str(devid)

    startT = time.time()
    with cd(app_dir):
        # print os.getcwd()
        # print app_dir
        # print cmd_str
        try:
            check_call(cmd_str, stdout=DEVNULL, stderr=STDOUT, shell=True)
        except CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {} ({})".format(
                    e.cmd, e.returncode, e.output, app_dir))

    endT = time.time()

    return [startT, endT]

#-----------------------------------------------------------------------------#
# check whether a file contains a string 
#-----------------------------------------------------------------------------#
def Check_str_in_file(f, mystr="", useRE=True):
    if useRE == False:
        if not mystr:
            print("Error in {}".format(__FILE__))
            sys.exit(1)

        if mystr in open(f).read():
            # if my str in the file, check whether  => may need double check!
            return True
        else:
            return False

    else: # use regular expression
        data = None
        with open(f, 'r') as myfile:
            data = myfile.read()
        outlist = re.findall(r'main\s*\(',data)
        if len(outlist) > 0:
            return True
        else:
            return False


#-----------------------------------------------------------------------------#
# save file 
#-----------------------------------------------------------------------------#
def save(filename, contents):  
    fh = open(filename, 'w')  
    fh.write(contents)  
    fh.close()  

#-----------------------------------------------------------------------------#
# remove comments  
#-----------------------------------------------------------------------------#
def Remove_comments(input_file, ofile="", as_newline=True):
    data = None
    with open(input_file, 'r') as myfile:
        data = myfile.read()

    if data <> None:
        if as_newline:
            # remove all occurance streamed comments (/*COMMENT */) from string
            data = re.sub(re.compile("/\*.*?\*/",re.DOTALL),"\n" ,data)
            # remove all occurance singleline comments (//COMMENT\n ) from string
            data = re.sub(re.compile("//.*?\n" ) ,"\n" ,data)
        else:
            data = re.sub(re.compile("/\*.*?\*/",re.DOTALL),"" ,data)
            data = re.sub(re.compile("//.*?\n" ) ,"" ,data)

        if ofile:
            save(ofile, data) 
        else:
            save(input_file + ".new", data) # save the org file as backups
         
#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
def keepFuncDecl(input_file, ofile=""):
    data = ""
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            IGN=True
            newline = line.strip()
            if len(newline) > 0:
                if newline[0] <> "#" and newline[-1] <> ";":
                    #print i, line 
                    IGN=False

            if IGN == False:
                data += line

    if len(ofile) > 0: 
        save(ofile, data) 



#-----------------------------------------------------------------------------#
# comment out lines (s1, s2) 
#-----------------------------------------------------------------------------#
def Comment_out_lines(infile, outfile, s1, s2):
    data = ""
    with open(infile, "r") as f:
        for i, line in enumerate(f, 1):
            if s1+1 <= i <= s2 -1:
                data += "//\n"
            else:
                data += line
    save(outfile, data)


#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
def find_funcName(input_file):
    func_name_list=[]
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            if line and "(" in line and (not line[0] in [" ", "\t"]):
                splitline = line.split("(")
                #print splitline[0]
                #print splitline
                func_name_list.append(splitline[0])
    return func_name_list




#-----------------------------------------------------------------------------#
# identify functions in the source file 
#-----------------------------------------------------------------------------#
def Genfunctionlist(src, retLineNum=False):
    # (1) remove comments
    # NOTE: remove line with extern "C" { 
    # #ifdef __cplusplus, find this line, remove the 2 line below

    data = ""
    with open(src, "r") as f:
        cplus = False 
        counter = 0
        for i, line in enumerate(f):
            EmptyLine = False
            noleading = line.strip()
            if noleading in ["\n","\r\n",""]: # symboles for empty line
                EmptyLine = True

            if "__cplusplus" in line:
                cplus = True
                counter = 3

            if cplus: 
                counter -= 1
                if counter == -1:
                    cplus = False

            if (EmptyLine == False) and (cplus == False):
                data += str(line)

    file01=src + "_01_rmEmptyLine"
    save(file01, data)






    # (2) find out the where the function begin and ends 
    # working on file01

    # NOTE: if there is a main(), return the beginning and ending line number

    lev = -1; progress = []; result=[];
    prev_lev = None
    mainFound = False
    with open(file01, "r") as f:
        for i, line in enumerate(f):
            if "main" in line and line[0] <> "#": mainFound=True; # main() line 
            for char in line:
                if char == "{":
                    #print i, line
                    if lev == -1: # at the root level
                        if mainFound: mainStartPos = i; # rmb where main starts
                        progress = []
                        result.append(progress)
                    progress.append([i+1, None]) # store the line num
                    lev += 1
                elif char == "}":
                    #print i, line
                    if lev > -1:
                        #print("current lev = {}".format(lev))
                        if not prev_lev:
                            prev_lev = lev # update prev lev
                        
                        #print("prev_lev = {}".format(prev_lev))

                        if prev_lev == lev:
                            USE_LAST = True 
                        elif prev_lev > lev:
                            prev_lev = lev  # update prev lev
                            USE_LAST = False 


                        if USE_LAST: # still on the same level
                            lastone = len(progress) - 1
                            progress[lastone][1] = i + 1
                        else:
                            progress[lev][1] = i + 1
                        #print("=> after updating : {} \t lev = {} \t prevLev = {}".format(progress, lev, prev_lev))

                        if lev == 0 and mainFound: 
                            mainEndPos=i; # rmb where main ends
                        lev -= 1

    if mainFound:
        print("main() is found in ({}). \t line#: {} - {}.".format(file01, mainStartPos, mainEndPos))


    #(3) comment the content of the function
    # NOTE: the 1st element of each item in the result indicates the function's
    # outmost curly bracket line nuber, we can remove the contents to obtain the function name
    file02=src + "_02_funcs"
    subprocess.call("cp " + file01 + " " + file02, shell=True) # use shell for copying

    for r in result:
        [s1, s2] = r[0]
        #print s1,s2
        Comment_out_lines(file02, file02, s1, s2)


    #(4) remove the comemnted line in (3)
    file03=src + "_03_funcDecls"
    Remove_comments(file02, ofile=file03)

    # (5) del lines which are not fund declaration
    file04=src + "_04_funcDecls_clean"
    keepFuncDecl(file03, ofile=file04)

    # (6) NOTE: read line with "(", consider the string before "(" as the func names
    func_name_list = find_funcName(file04)
    #print func_name_list

    if retLineNum and mainFound:
        return func_name_list, {file01: [mainStartPos, mainEndPos]} 
    else:
        return func_name_list


def parsing_file(targetfile, l1, l2):
    with open(targetfile, "r") as f:
        for i, line in enumerate(f):
            if l1 <= i <= l2:
                #print i, line 

                # for loop
                found = re.findall(r'for\s*\(', line) 
                if len(found) > 0:
                    print("for loop at {}".format(i))

                # while loop
                found = re.findall(r'while\s*\(', line) 
                if len(found) > 0:
                    print("while loop at {}".format(i))

                # fopen
                found = re.findall(r'fopen\s*\(', line) 
                if len(found) > 0:
                    print("fopen at {}".format(i))

                # switch 
                found = re.findall(r'switch\s*\(', line) 
                if len(found) > 0:
                    print("switch at {}".format(i))

                # malloc 
                found = re.findall(r'malloc\s*\(', line) 
                if len(found) > 0:
                    print("malloc at {}".format(i))

#-----------------------------------------------------------------------------#
# Analyze program flow 
#-----------------------------------------------------------------------------#
def gen_program_flow(app_dir):
    from os import walk
    with cd(app_dir):
        #------------------------------------------
        # step1: go the dir and list all the files 
        #------------------------------------------
        print("checking the files under the folder :  {}".format(app_dir))

        f = []
        for (dirpath, dirnames, filenames) in walk("."):
            #print("dirpath={}".format(dirpath))
            #print("dirnames={}".format(dirnames))
            #print("filenames={}".format(filenames))

            if dirpath==".":
                for localfile in filenames:
                    f.append("./" + localfile)
                #print f
            else:
                # sub folder
                if filenames and (not dirnames):
                    for fname in filenames:
                        srcfile = str(dirpath) + "/" + str(fname)
                        #print srcfile
                        f.append(srcfile)

        #------------------------------------------
        # step2: find all the source files ending with .c/.cu/.cpp
        #        (ignore headers)
        #------------------------------------------
        myfiles= []
        for targetfile in f:
            #print targetfile,targetfile[-2:]
            if targetfile[-2:] in [".c"] or targetfile[-3:] in [".cu"] or targetfile[-4:] in [".cpp"]:
                myfiles.append(targetfile)

        print("source files:")
        print("\t{}\n".format(myfiles))

        #------------------------------------------
        # step3: find main function file
        #------------------------------------------
        main_file = []
        for eachfile in myfiles:
            if Check_str_in_file(eachfile, "main("):
                print("found file with main() = {}".format(eachfile));
                main_file = eachfile;
                break;

        print("main file = {}\n".format(main_file))


        #------------------------------------------
        # step 4: remove the comments in the files
        #           each modified file ending with .new
        #------------------------------------------
        for eachfile in myfiles:
            #print("remove comments for = {}".format(eachfile))
            Remove_comments(eachfile)


        #------------------------------------------
        # step 5: find out all the functions in the files
        #------------------------------------------
        #Genfunctionlist(main_file + ".new", retLineNum=True)
        functionlist, mainfile_linenum_dd = Genfunctionlist(main_file + ".new", retLineNum=True)
        #print mainfile_linenum_dd 
        #print functionlist




        for f in myfiles:
            if f <> main_file:
                f_new = f + ".new"
                print("generating funcs for {}\n".format(f_new))
                functionlist = Genfunctionlist(f_new)
                #Genfunctionlist(f_new)

                print functionlist 
                print "\n\n"



        ##
        ## step 6: 
        ##
        #related_files = [i for i in myfiles if i <> main_file]
        #
        ##
        ## read file without empty line
        ##
        #print("reading the file to parse the program structure")
        #targetfile = mainfile_linenum_dd.keys()[0]
        #[startLine, endLine] = mainfile_linenum_dd[targetfile]

        ## figure out when the main()) starts and ends and read codes inbetween
        #print targetfile, startLine, endLine, '\n'

        #parsing_file(targetfile, startLine, endLine)


        #with open(current_file, "r") as f:
        #    for i, line in enumerate(f):
        #        if line[0] <> "#" and "main" in line:



        # fopen:  cpu read input file
        # malloc: cpu
        # check whether there is a function


#-----------------------------------------------------------------------------#
# GPU Job Table 
#-----------------------------------------------------------------------------#
def PrintGpuJobTable(GpuJobTable, total_jobs, id2name, saveFile=None):
    """Print application trace."""

    #
    # gpujobtable cols:   
    #            jobid / gpu / status / startT / endT
    #

    # To save, jobID / jobName / start / end / duration  (5 columns)
    if saveFile:
        traceCols = ['jobID', 'appName', 'start', 'end', 'duration (s)']
        df_trace = pd.DataFrame(index=np.arange(0, total_jobs),columns=traceCols)

    print("JobID\tStart\tEnd\tDuration\tAppName")
    start_list = []
    end_list = []
    for row in xrange(total_jobs):
        jobid    = GpuJobTable[row,0]
        startT   = GpuJobTable[row,3]
        endT     = GpuJobTable[row,4]
        duration = endT - startT 
        appName  = id2name[jobid] 

        if saveFile:
            df_trace.loc[row, 'jobID']         = jobid
            df_trace.loc[row, 'appName']       = appName 
            df_trace.loc[row, 'start']         = startT 
            df_trace.loc[row, 'end']           = endT 
            df_trace.loc[row, 'duration (s)']  = duration 

        print("{}\t{}\t{}\t{}\t{}".format(jobid, startT, endT, duration, appName))

        start_list.append(startT)
        end_list.append(endT)

    total_runtime = max(end_list) - min(start_list) 
    print("total runtime = {} (s)".format(total_runtime))

    if saveFile:
        df_trace.to_csv(saveFile, index=False, encoding='utf-8')
        print("Done! Check the saved file {}.".format(saveFile))


#-----------------------------------------------------------------------------#
# GPU Job Table 
#-----------------------------------------------------------------------------#
def getGpuJobTiming(GpuJobTable, total_jobs, id2name):
    """Print application timing."""

    #
    # gpujobtable cols:   
    # jobID / jobName / start / end / duration  (5 columns)

    #print("JobID\tStart\tEnd\tDuration\tAppName")
    start_list = []
    end_list = []
    ret = {} 
    for row in xrange(total_jobs):
        jobid    = GpuJobTable[row,0]
        startT   = GpuJobTable[row,3]
        endT     = GpuJobTable[row,4]
        duration = endT - startT 
        appName  = id2name[jobid] 

        #print("{}\t{}\t{}\t{}\t{}".format(jobid, startT, endT, duration, appName))

        start_list.append(startT)
        end_list.append(endT)

        ret[appName] = duration

    total_runtime = max(end_list) - min(start_list) 
    #print("total runtime = {} (s)".format(total_runtime))

    return ret
