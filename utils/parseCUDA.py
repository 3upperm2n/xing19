#!/usr/bin/env python

import os,sys
import subprocess
from subprocess import check_call,STDOUT,CalledProcessError,call,Popen,PIPE
import re

def print_error(*args):
    print("[Error]: {} \nExiting Program!\n".format(''.join(map(str, *args))))
    sys.exit(1)


def FindPattern(cmd, showoutput=False):
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read()
    if showoutput: print output;

    if len(output) > 0 : 
        return True
    else:
        return False

def FindPattern_linenum(cmd):
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read()
    if len(output) > 0 : 
        linenum = output.split()[0]
        return int(linenum)
    else:
        print_error("No matching!")
        


def FindMajorFuncBrac(infile):
    lev = -1; progress = []; result=[];
    prev_lev = None
    with open(infile, "r") as f:
        for i, line in enumerate(f):
            for char in line:
                if char == "{":
                    #print i, line
                    if lev == -1: # at the root level
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

                        lev -= 1

    return result


#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
def FindFuncName(cudafile, read_region):
    cuda_funcs = []
    for pos in read_region:
        [start, end] = pos
        BeginRead = False 

        # NOTE: not efficient
        with open(cudafile, 'r') as f:
            for i, line in enumerate(f,1):
                if i >= start:
                    BeginRead = True

                if i >= end:
                    BeginRead = False

                if BeginRead:
                    newline = line.strip()
                    if "(" in newline:
                        header = newline.split("(")
                        funcname = header[0].split()[-1] # last one before (
                        #print funcname
                        cuda_funcs.append(funcname)

    if len(cuda_funcs) >= 1:
        return cuda_funcs
    else:
        print_error("cuda functions not found!")


#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
def GenApiSeq(cudafile, brac_start, brac_end):
    print os.getcwd()
    print cudafile

    #--- api list ---#
    forloop = []
    whileloop = []
    #cudastream = []
    cudamalloc = []
    cudafree = []
    #cudamemset= []
    cudaconst = []
    cudatex = []
    kernel = []
    #cudacopy = []
    h2d = []
    d2h = []
    d2d = []

    #cufft= []
    cudasync = []
    

    # where it should start reading
    FindFor = False 
    FindBrac4For = False
    SkipCurFor = False

    FindWhl = False 
    FindBrac4Whl = False
    SkipCurWhl = False


    with open(cudafile, "r") as f:
        for i, line in enumerate(f,1):
            if brac_start <= i <= brac_end:

                #print line

                #--------------------------------------------------------------#
                # for loop 
                #--------------------------------------------------------------#
                match = re.findall(r'for\s*\(', line); 
                if match:
                    print("for() at {}".format(i))
                    forloop.append([i, -1]);  # [startPos, endPos]
                    FindFor = True
                    FindBrac4For = False
                    SkipCurFor = False  # restart / init
                    left_bracs, right_bracs = 0, 0

                if FindFor:
                    lastforline = forloop[-1][0] # find the current for line num

                    if "{" in line: FindBrac4For = True;

                    if (i - lastforline) >= 1  and (not FindBrac4For):
                        SkipCurFor = True  # case 1: propably for in one line 

                    if FindBrac4For and (not SkipCurFor): 
                        if "{" in line:
                            left_bracs += 1
                        if "}" in line:
                            right_bracs += 1

                        if left_bracs == right_bracs:
                            # we found where for() ends
                            print("for() ends at {}".format(i))
                            forloop[-1][1] = i
                            FindFor = False # end searching
                        


                    

                # while loop 
                match = re.findall(r'while\s*\(', line); 
                if match:
                    print("while() at {}".format(i))
                    whileloop.append([i, -1]); # [start, end]
                    FindWhl = True
                    FindBrac4whl = False
                    SkipCurWhl = False
                    left_brac_whl, right_brac_whl = 0,0
                    #leiming







                ## cudastream
                #match = re.findall(r'cudaStreamCreate', line); if match: cudastream.append(i); 

                # cudamalloc 
                match = re.findall(r'cudaMalloc[A-Fa-f\s]*\(', line); 
                if match:
                    print("cudaMalloc() at {}".format(i))
                    cudamalloc.append(i); 

                # cudafree
                match = re.findall(r'cudaFree[A-Fa-f\s]*\(', line); 
                if match:
                    print("cudaFree() at {}".format(i))
                    cudafree.append(i); 

                ## cudaMemset
                #match = re.findall(r'cudaMemset\s*\(', line); if match: cudamemset.append(i); 

                # constant memory 
                match = re.findall(r'cudaMemcpy(From|To)Symbol[A-Fa-f\s]*\(', line);
                if match:
                    print("Constant Memory copy at {}".format(i))
                    cudaconst.append(i); 

                # texture memory 
                match = re.findall(r'cuda(Bind|Unbind|Get)Texture[A-Fa-f\s]*\(', line);
                if match:
                    print("Constant Memory copy at {}".format(i))
                    cudatex.append(i);

                # cuda kernel 
                match = re.findall(r'<<<', line);
                if match:
                    print("CUDA Kernel at {}".format(i))
                    kernel.append(i); 

                # data copy
                match = re.findall(r'cudaMemcpy[A-Fa-f\s]*\(', line);
                if match: 
                    #print i, line
                    if "cudaMemcpyHostToDevice" in line:
                        print("h2d at {}".format(i))
                        h2d.append(i)
                    elif "cudaMemcpyDeviceToHost" in line:
                        print("d2h at {}".format(i))
                        d2h.append(i)
                    elif "cudaMemcpyDeviceToDevice" in line:
                        print("d2d at {}".format(i))
                        d2d.append(i)
                    #cudacopy.append(i); 

                ## cufft
                #match = re.findall(r'cufft[A-Fa-f\s]*\(', line); if match: cufft.append(i); 

                # sync
                match = re.findall(r'cuda(Event|Device)Synchronize\s*\(', line);
                if match:
                    print("sync at {}".format(i))
                    cudasync.append(i);
        
    #
    #
    #

    print "for() loop"
    print forloop

    api_dd = {}
    if len(forloop) > 0: api_dd['for']= forloop;
    if len(whileloop) > 0: api_dd['while']= whileloop;
    if len(cudamalloc) > 0: api_dd['cudamalloc']= cudamalloc;
    if len(cudafree) > 0: api_dd['cudafree']= cudafree;
    if len(cudaconst) > 0: api_dd['cudaconst']= cudaconst;
    if len(cudatex) > 0: api_dd['cudatex']= cudatex;
    if len(kernel) > 0: api_dd['kernel']= kernel;

    #if len(cudacopy) > 0: api_dd['cudacopy']= cudacopy;
    if len(h2d) > 0: api_dd['h2d']= h2d;
    if len(d2h) > 0: api_dd['d2h']= d2h;
    if len(d2d) > 0: api_dd['d2d']= d2d;

    if len(cudasync) > 0: api_dd['cudasync']= cudasync;

    # TODO: check kernel: whether in a for loop or while loop
    #if forloop:
    #    print "[Error]checking for() is not implemented!"
    #    sys.exit(1)

    if whileloop:
        print "[Error]checking while() is not implemented!"
        sys.exit(1)


    #print api_dd

    # what is the related function call?



#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
def parseCUDAfile(cudafile):
    "parsing cuda files ending with .new_01_rmEmptyLine"


    #
    # find the function name 
    #
    func_bracket = FindMajorFuncBrac(cudafile)
    print "\nmajor functions (start/end) line positions in current files:"
    #print func_bracket
    print func_bracket
    #print len(func_bracket[0])

    #
    # fig out where to read the func header 
    #
    prev_end = 0
    read_region = []
    for i, fc in enumerate(func_bracket, 1):
        [b_start, b_end] = fc[0]
        if i == 1:
            read_region.append([1, b_start])
        else: # i > 1
            read_region.append([prev_end, b_start])
        prev_end = b_end
    #print read_region

    #
    # find out the func name 
    #
    funclist = FindFuncName(cudafile, read_region)
    funcstart_dd = {}
    for f in funclist:
        cmd = "awk '/" + f +".*\(/{print NR, $0}' " + cudafile 
        #print cmd
        linenum = FindPattern_linenum(cmd)
        funcstart_dd[f] = linenum
        #print linenum
        #print "--- done find ---\n"

    print "function start postion:"
    print funcstart_dd



    #----------#
    # parse the cuda api call sequence for the target function
    #----------#
    idx = 0
    for fname, startpos in funcstart_dd.iteritems():
        [brac_start, brac_end] = func_bracket[idx][0]
        print("func={}\t func_start_at={}\t reading_region= {} to {}".format(
            fname, startpos, brac_start, brac_end))

        #api_seq = GenApiSeq(cudafile, brac_start, brac_end)
        GenApiSeq(cudafile, brac_start, brac_end)


        print "--- done ---\n"
        idx += 1


    #for i, fc in enumerate(func_bracket, 1):
    #    [b_start, b_end] = fc[0]



    #return api_dd


    # api of interest

#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
def findCUDA_rt(targetfile):
    """
    Check cuda runtime api
    """

    # use dd to track apis
    apiDD = {}
    apiDD['cudaMalloc'] = 0
    apiDD['cudaFree'] = 0
    apiDD['cudaMemset'] = 0
    apiDD['cudaConst'] = 0
    apiDD['cudaTex'] = 0
    apiDD['kernel'] = 0
    apiDD['cudaMemcpy'] = 0
    apiDD['cufft'] = 0
    apiDD['cudaSync'] = 0
    apiDD['cudaStream'] = 0


    #
    # cudamalloc
    #
    cmd = "awk '/cudaMalloc[A-Fa-f]*\(/{print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cudaMalloc'] = 1;

    #
    # cudafree
    #
    cmd = "awk '/cudaFree[A-Fa-f\s]*\(/{print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cudaFree'] = 1;

    #
    # cudaMemset
    #
    cmd = "awk '/cudaMemset\s*\(/{print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cudaMemset'] = 1;

    #
    # constant memory 
    #
    cmd = "awk '/cudaMemcpy(From|To)Symbol[A-Fa-f\s]*\(/ {print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cudaConst'] = 1;

    #
    # texture memory 
    #
    cmd = "awk '/cuda(Bind|Unbind|Get)Texture[A-Fa-f\s]*\(/ {print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cudaTex'] = 1;


    #
    # cuda kernel 
    #
    cmd = "awk '/<<</{print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['kernel'] = 1;

    #
    # data transfer: check cuda_runtime.h
    # NOTE: direction
    #
    cmd = "awk '/cudaMemcpy[A-Fa-f\s]*\(/ {print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cudaMemcpy'] = 1;

    #
    # cuff support: cufft.h
    #
    cmd = "awk '/cufft[A-Fa-f\s]*\(/ {print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cufft'] = 1;

    #
    # synchronization 
    #
    cmd = "awk '/cuda(Event|Device)Synchronize\s*\(/ {print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cudaSync'] = 1;

    #
    # cudastream 
    #
    cmd = "awk '/cudaStream[A-Fa-f\s]*\(/ {print $0}' " + targetfile 
    if FindPattern(cmd): apiDD['cudaStream'] = 1;


    #
    # NOTE: check drv api, reported, to-do 
    # check : ./common/inc/dynlink/cuda_drvapi_dynlink_cuda.h
    #
    cmd = "awk '/tcuMem[A-Fa-f\s]*\(/ {print $0}' " + targetfile 
    if FindPattern(cmd): print("[Warning] Found driver API"); sys.exit(1);


    found_cudaapi = False
    if sum(apiDD.values()) > 0:
        found_cudaapi = True
        print("\n---------\nFor file = {}".format(targetfile))

    for key, value in apiDD.iteritems():
        if value == 1:
            if key == "cudaMalloc": print("found cuda malloc");
            if key == "cudaFree": print("found cuda free");
            if key == "cudaMemset": print("found cuda memset");
            if key == "cudaConst": print("found cuda constant");
            if key == "cudaTex": print("found cuda tex memory");
            if key == "kernel": print("found cuda kernel");
            if key == "cudaMemcpy": print("found cuda data transfer");
            if key == "cufft": print("found cufft");
            if key == "cudaSync": print("found cuda sync");
            if key == "cudaStream": print("found cuda stream");

    return found_cudaapi









#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
