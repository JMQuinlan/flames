#!/usr/bin/env python3
import sys
import argparse
import os, glob, subprocess
import configparser, io
from collections import OrderedDict
from datetime import datetime
import socket
import time
import re
import pathlib

from sympy import capture

class color:
    reset = "\033[0m"
    red   = "\033[31m"
    green   = "\033[32m"
    blue   = "\033[34m"
    magenta   = "\033[35m"
    boldblue   = "\033[1m\033[34m"
    boldgreen   = "\033[1m\033[32m"
    boldyellow   = "\033[1m\033[33m"
    bold = "\033[1m"
    lightgray = "\033[37m"
    darkgray = "\033[90m"

#
# RE tool to strip out color escapes
# 
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

#
# Get a unique string ID to label all output files
#
now = datetime.now()
testid = now.strftime("output_%Y-%m-%d_%H.%M.%S_"+socket.gethostname())
print("Test ID = ",testid)

#
# Special order from SO - dictionary allows for keys to be specified multiple times and 
# config parser will read it all.
class MultiOrderedDict(OrderedDict):
    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super().__setitem__(key, value)


#
# Provide some simple command line arguments for filtering the types of 
# regression tests to run. For instance, if you need to run without
# mpirun, you can use the --serial flag.
#
parser = argparse.ArgumentParser(description='Configure ALAMO');
parser.add_argument('tests', default=None, nargs='*', help='Spatial dimension [3]')
parser.add_argument('--serial',action='store_true',default=False,help='Run in serial only (no mpi)')
parser.add_argument('--dim',default=None,type=int,help='Specify dimensions to run in')
parser.add_argument('--cmd',default=False,action='store_true',help="Print out the exact command used to run each test")
parser.add_argument('--sections',default=None, nargs='*', help='Specific sub-tests to run')
parser.add_argument('--exe',default=None, nargs='*', help='Run only certain executables')
parser.add_argument('--debug',default=False,action='store_true',help='Use the debug version of the code')
parser.add_argument('--profile',default=False,action='store_true',help='Use the profiling version of the code')
parser.add_argument('--coverage',default=False,action='store_true',help='Use the gcov version of the code for all tests')
parser.add_argument('--only-coverage',default=False,action='store_true',help='Gracefully skip non-coverage tests')
parser.add_argument('--only-non-coverage',default=False,action='store_true',help='Gracefully skip coverage tests')
parser.add_argument('--no-coverage',default=False,action='store_true',help='Prevent coverage version of the code from being used')
parser.add_argument('--benchmark',default=socket.gethostname(),help='Current platform if testing performance')
parser.add_argument('--dryrun',default=False,action='store_true',help='Do not actually run tests, just list what will be run')
parser.add_argument('--comp', default="g++", help='Compiler. Options: [g++], clang++, icc')
parser.add_argument('--timeout', default=10000, help='Timeout value in seconds (default: 10000)')
parser.add_argument('--post', default=None, help='Use a post script to post results')
parser.add_argument('--clean', dest='clean', default=True, action='store_true', help='Clean up output files if test is successful (on by default)')
parser.add_argument('--no-clean', dest='clean', default=False, action='store_false', help='Keep all output files')
parser.add_argument('--permissive', dest='permissive', default=False, action='store_true', help='Option to run without erroring out (if at all possible)')
parser.add_argument('--permit-timeout', dest='permit_timeout', default=False, action='store_true', help='Permit timeouts without failing')
args=parser.parse_args()

if args.coverage and args.no_coverage:
    raise Exception("Cannot specify both --coverage and --no-coverage")
if args.only_coverage and args.no_coverage:
    raise Exception("Cannot specify both --only-coverage and --no-coverage")

if args.post:
    if not os.path.isfile(args.post):
        raise Exception(args.post,"is not a file")
    sys.path.append(str(pathlib.Path(args.post).parent))
    import post
    postdata = post.init()



class DryRunException(Exception):
    pass

def test(testdir):
    """
    Run tests using alamo on the input file stored in "testdir"
    We assume the following convention: for a test called "MyTest",

    test directory:      ./tests/MyTest          # Directory containig all test information
    alamo input file:    ./tests/MyTest/input    # Input file with config file comments 
    output file:         ./tests/MyTest/output   # All alamo output
    check script:        ./tests/test            # Executable script that returns 0 for successful test
    
    """

    # Counters to track failed tests, skipped tests, passed tests, passed checks
    fails = 0
    skips = 0
    tests = 0
    checks = 0
    fasters = 0
    slowers = 0
    timeouts = 0
    records = []

    # Parse the input file ./tests/MyTest/input containing #@ comments.
    # Everything commeneted with #@ will be interpreted as a "config" file
    # The variable "config" is a dict of dicts where each item corresponds to
    # a test configuration.
    cfgfile = io.StringIO()
    input = open(testdir + "/input")
    for line in input.readlines():
        if line.startswith("#@"):
            cfgfile.write(line.replace("#@",""))
    cfgfile.seek(0)
    config = configparser.ConfigParser(dict_type=MultiOrderedDict,strict=False)
    config.read_file(cfgfile)

    sections = config.sections()
    if args.sections:
        if len(args.sections)>0:
            sections = list(set(config.sections()).intersection(set(args.sections)))

    # If there are no runs specified, then we will not test anything
    # in this directory, and will print an "ignore" message.
    # (Eventually, everything in ./tests should be tested!)
    if not len(sections):
        print("{}IGNORE {}{}".format(color.darkgray,testdir,color.reset))
        return 0,0,0,0,0,0,0,[]
        
    # Otherwise let the user know that we are in this directory
    print("RUN    {}{}{}".format(color.bold,testdir,color.reset))

    # Iterate through all test configurations
    for desc in sections:

        record = dict()
        record['testdir'] = testdir
        record['section'] = desc
        record['test-section'] = record['testdir']+'/'+record['section']
        record['section'] = desc
        record['testid']  = testid
        record['path'] = "{}/{}_{}".format(testdir,testid,desc)


        p = subprocess.check_output('git rev-parse --abbrev-ref HEAD'.split(),stderr=subprocess.PIPE)
        record['branch'] = p.decode('utf-8').replace('\n','')

        # In some cases we want to run the exe but can't check it.
        # Skipping the check can be done by specifying the "check" input.
        check = True
        if 'check' in config[desc].keys(): 
            if config[desc]['check'] in {"no","No","false","False","0"}:
                check = False
            elif config[desc]['check'] in {"yes","Yes","true","True","1"}:
                check = True
            else:
                raise(Exception("Invalid value for check: {}".format(config[desc]['check'])))

        # Determine if we want to use the coverage version of the code.
        coverage = args.coverage
        if 'coverage' in config[desc].keys():
            if config[desc]['coverage'] in {"no","No","false","False","0"}:
                coverage = False
            elif config[desc]['coverage'] in {"yes","Yes","true","True","1"}:
                coverage = True
            else:
                raise(Exception("Invalid value for coverage: {}".format(config[desc]['coverage'])))
        if args.only_coverage and not coverage:
            continue
        if args.only_non_coverage and coverage:
            continue
        if args.no_coverage:
            coverage = False

        timeout = int(args.timeout)
        if 'timeout' in config[desc].keys():
            timeout = int(config[desc]['timeout'])

        dobenchmark = False
        benchmark = None
        if "benchmark-{}".format(args.benchmark) in config[desc].keys():
            dobenchmark = True
            benchmark = float(config[desc]["benchmark-{}".format(args.benchmark)])

        # Build the command to run the script. This can be done in two ways:
        #
        # 1. with the 'cmd' input where 'cmd' is the precise run command.
        #    This is NOT PORTABLE and useful only for initial testing. You should
        #    generally use option 2.
        # 2. with 'dim', 'nprocs', 'args', etc keywords. See current tests for
        #    examples
        command = ""
        if 'cmd' in config[desc].keys():
            command = config[desc]['cmd']
            if len(config[desc].keys()) > 1:
                raise Exception("If 'cmd' is specified no other parameters can be set. Received " + ",".join(config[desc].keys))
        else:
            exe = 'alamo'
            if 'exe' in config[desc].keys(): exe = config[desc]['exe']
            dim = 3 # Dimension of alamo to use
            if 'dim' in config[desc].keys(): dim = int(config[desc]['dim'])
            nprocs = 1 # Number of MPI processes, if 1 then will run without mpirun
            if 'nprocs' in config[desc].keys(): nprocs = int(config[desc]['nprocs'])
            cmdargs = "" # Extra arguments to pass to alamo in addition to input file
            if 'args' in config[desc].keys(): cmdargs = config[desc]['args'].replace('\n',' ')

            cmdargs += " plot_file={}/{}_{}".format(testdir,testid,desc)

            if 'ignore' in config[desc].keys():
                cmdargs += " ignore={}".format(config[desc]['ignore'])

            # Quietly ignore this one if running in serial mode.
            if nprocs > 1 and args.serial: 
                continue
            # If not running in serial, specify mpirun command
            if nprocs > 1: command += "mpirun -np {} ".format(nprocs)
            # Specify alamo command.
            
            exestr = "./bin/{}-{}d".format(exe,dim)
            if args.debug: exestr += "-debug"
            if args.profile: exestr += "-profile"
            if coverage: exestr += "-coverage"
            exestr += "-"+args.comp
            
            #
            # Sometimes we don't have a coverage version built; in that case,
            # use the non-coverage version.
            #
            if not os.path.isfile(exestr):
                exestr=exestr.replace("-coverage","")

            #if args.debug and args.profile: exestr = "./bin/alamo-{}d-profile-debug-{}".format(dim,args.comp)
            #elif args.debug: exestr = "./bin/alamo-{}d-debug-{}".format(dim,args.comp)
            #elif args.profile: exestr = "./bin/alamo-{}d-profile-{}".format(dim,args.comp)
            #else: exestr = "./bin/alamo-{}d-{}".format(dim,args.comp)

            # If we specified a CLI dimension that is different, quietly ignore.
            if args.dim and not args.dim == dim:
                continue

            if args.exe:
                if not exe in args.exe:
                    continue

            # If the exestr doesn't exist, exit noisily. The script will continue but will return a nonzero
            # exit code.
            if not os.path.isfile(exestr):
                print("  ├ {}{} (skipped - no {} executable){}".format(color.boldyellow,desc,exestr,color.reset))
                skips += 1
                continue

            # If we have enabled "skipping," exit noisily. The script will continue but will return a nonzero
            # exit code.
            if 'skip' in config[desc].keys():
                if config[desc]['skip'].lower() in ['true','yes','1']:
                    print("  ├ {}{} (skip indicated in input){}".format(color.boldyellow,desc,color.reset))
                    skips += 1
                    continue

            command += exestr + " "
            command += "{}/input ".format(testdir)
            command += cmdargs
        
        # Run the actual test.
        print("  ├ " + desc)
        if args.cmd: print("  ├      " + command)
        print("  │      Running test............................................",end="",flush=True)
        # Spawn the process and wait for it to finish before continuing.
        try:
            if args.dryrun: raise DryRunException()
            timeStarted = time.time()
            p = subprocess.run(command.split(),capture_output=True,check=True,timeout=timeout)
            executionTime = time.time() - timeStarted
            record['executionTime'] = str(executionTime)
            fstdout = open("{}/{}_{}/stdout".format(testdir,testid,desc),"w")
            fstdout.write(ansi_escape.sub('',p.stdout.decode('utf-8')))
            fstdout.close()
            fstderr = open("{}/{}_{}/stderr".format(testdir,testid,desc),"w")
            fstderr.write(ansi_escape.sub('',p.stderr.decode('utf-8')))
            fstderr.close()
            print("[{}PASS{}]".format(color.boldgreen,color.reset), "({:.2f}s".format(executionTime),end="")
            record['runStatus'] = 'PASS'
            if dobenchmark:
                if abs(executionTime - benchmark) / (executionTime + benchmark) < 0.01: print(", no change)")
                elif abs(executionTime < benchmark):
                    print(",{} {:.2f}% faster{})".format(color.blue,100*(benchmark-executionTime)/executionTime,color.reset))
                    fasters += 1
                else:
                    print(",{} {:.2f}% slower{})".format(color.magenta,100*(executionTime-benchmark)/executionTime,color.reset))
                    slowers += 1
            else: print(")")

            tests += 1
        # If an error is thrown, we'll go here. We will print stdout and stderr to the screen, but 
        # we will continue with running other tests. (Script will return an error)
        except subprocess.CalledProcessError as e:
            print("[{}FAIL{}]".format(color.red,color.reset))
            record['runStatus'] = 'FAIL'
            print("  │      {}CMD   : {}{}".format(color.red,' '.join(e.cmd),color.reset))
            for line in e.stdout.decode('utf-8').split('\n'): print("  │      {}STDOUT: {}{}".format(color.red,line,color.reset))
            for line in e.stderr.decode('utf-8').split('\n'): print("  │      {}STDERR: {}{}".format(color.red,line,color.reset))
            fails += 1
            continue
        # If an error is thrown, we'll go here. We will print stdout and stderr to the screen, but 
        # we will continue with running other tests. (Script will return an error)
        except subprocess.TimeoutExpired as e:
            print("[{}TIMEOUT{}]".format(color.red,color.reset))
            record['runStatus'] = 'TIMEOUT'
            print("  │      {}CMD   : {}{}".format(color.red,' '.join(e.cmd),color.reset))
            try:
                stdoutlines = e.stdout.decode('utf-8').split('\n')
                if len(stdoutlines) < 10:
                    for line in stdoutlines: print("  │      {}STDOUT: {}{}".format(color.red,line,color.reset))
                else:
                    for line in stdoutlines[:5]:  print("  │      {}STDOUT: {}{}".format(color.red,line,color.reset))
                    for i in range(3):            print("  │      {}        {}{}".format(color.red,"............",color.reset))
                    for line in stdoutlines[-5:]: print("  │      {}STDOUT: {}{}".format(color.red,line,color.reset))
                #for line in e.stderr.decode('utf-8').split('\n'): print("  │      {}STDERR: {}{}".format(color.red,line,color.reset))
            except Exception as e1:
                for line in str(e1).split('\n'):
                    print("  │      {}EXCEPT: {}{}".format(color.red,line,color.reset))
            timeouts += 1
            continue
        except DryRunException as e:
            print("[----]")
            record['runStatus'] = '----'
            
        # Catch-all handling so that if something else odd happens we'll still continue running.
        except Exception as e:
            print("[{}FAIL{}]".format(color.red,color.reset))
            record['runStatus'] = 'FAIL'
            for line in str(e).split('\n'): print("  │      {}{}{}".format(color.red,line,color.reset))
            fails += 1
            continue
        
        # If we have specified that we are doing a check, use the 
        # ./tests/MyTest/test 
        # script to determine if the run was successful.
        # The exception handling is basically the same as for the above test.
        if check:
            print("  │      Checking result.........................................",end="",flush=True)
            try:
                if args.dryrun: raise DryRunException()
                cmd = ["./test","{}_{}".format(testid,desc)]
                if "check-file" in config[desc].keys():
                    cmd.append(config[desc]['check-file'])
                if args.cmd: 
                    print("  ├      " + ' '.join(cmd))
                p = subprocess.check_output(cmd,cwd=testdir,stderr=subprocess.PIPE)
                checks += 1
                print("[{}PASS{}]".format(color.boldgreen,color.reset))
                record['checkStatus'] = 'PASS'
            except subprocess.CalledProcessError as e:
                print("[{}FAIL{}]".format(color.red,color.reset))
                record['checkStatus'] = 'FAIL'
                print("  │      {}CMD   : {}{}".format(color.red,' '.join(e.cmd),color.reset))
                for line in e.stdout.decode('utf-8').split('\n'): print("  │      {}STDOUT: {}{}".format(color.red,line,color.reset))
                for line in e.stderr.decode('utf-8').split('\n'): print("  │      {}STDERR: {}{}".format(color.red,line,color.reset))
                fails += 1
                continue
            except DryRunException as e:
                print("[----]")
                record['checkStatus'] = "----"
            except Exception as e:
                print("[{}FAIL{}]".format(color.red,color.reset))
                record['checkStatus'] = 'FAIL'
                for line in str(e).split('\n'): print("  │      {}{}{}".format(color.red,line,color.reset))
                fails += 1
                continue
        else:
            record['checkStatus'] = 'NONE'


        #
        # Scan metadata file for insteresting things to include in the record
        #
        if args.post:
            try:
                metadatafile = open("{}/{}_{}/metadata".format(testdir,testid,desc),"r")
                metadata = dict()
                for line in metadatafile.readlines():
                    if line.startswith('#'): continue;
                    if '::' in line:
                        ### skip these for now ...
                        continue
                        #line = re.sub(r'\([^)]*\)', '',line)
                        #line = line.replace(" :: ", " = ").replace('[','').replace(',','').replace(']','').replace(' ','')
                    if len(line.split(' = ')) != 2: continue;
                    col = line.split(' = ')[0]#.replace('.','_')
                    val = line.split(' = ')[1].replace('\n','')#.replace('  ','').replace('\n','').replace(';','')
                    metadata[col] = val
                metadatafile.close()
                record['git_commit_hash'] = metadata['Git_commit_hash']
                record['platform'] = metadata['Platform']
                record['test-section'] = record['testdir'] + '/' + record['section']
                p = subprocess.run('git show --no-patch --format=%ci {}'.format(record['git_commit_hash'].split('-')[0]).split(),capture_output=True)
                record['git_commit_date'] = p.stdout.decode('utf-8').replace('\n','')
            except Exception as e:
                if not args.permissive:
                    print("Problem getting metadata, here it is:")
                    print(record)
                    raise Exception()
                True # permissive

            try:
                post.updateDatabase(postdata,[record])
            except Exception as e:
                print("  │      [{}POST ERROR{}] : {}".format(color.red,color.reset,e))
        records.append(record)


        #
        # Clean up all of the node and cell file 
        #
        ok_to_clean = False
        if args.clean and record['runStatus'] == 'PASS':
            if not 'checkStatus' in record.keys():
                ok_to_clean = True # successful run and no testing done
            elif record['checkStatus'] == 'PASS':
                ok_to_clean = True # successful run and testing completed
            else:
                ok_to_clean = False # successful run but testing failed
        else:
            ok_to_clean = False

        if ok_to_clean:
            path = "{}/{}_{}".format(testdir,testid,desc)
            p = subprocess.run(f'rm -rf *cell *node',capture_output=True,cwd=path,shell=True)
            
        
            
    # Print a quick summary for this test family.
    summary = "  └ "
    sums = []
    if tests: sums.append("{}{} tests run{}".format(color.blue,tests,color.reset))
    if checks: sums.append("{}{} checks passed{}".format(color.green,checks,color.reset))
    if fails: sums.append("{}{} tests failed{}".format(color.red,fails,color.reset))
    if skips: sums.append("{}{} tests skipped{}".format(color.boldyellow,skips,color.reset))
    if timeouts: sums.append("{}{} tests timed out{}".format(color.red,timeouts,color.reset))
    print(summary + ", ".join(sums))
    return fails, checks, tests, skips, fasters, slowers, timeouts, records

# We may wish to pass in specific test directories. If we do, then test those only.
# Otherwise look at everything in ./tests/
if args.tests: tests = sorted(args.tests)
else: tests = sorted(glob.glob("./tests/*"))

tests = [str(pathlib.Path(f)) for f in tests]

class stats:
    fails = 0   # Number of failed runs - script errors if this is nonzero
    skips = 0   # Number of tests that were unexpectedly skipped - script errors if this is nonzero
    checks = 0  # Number of successfully passed checks
    tests = 0   # Number of successful checks
    fasters = 0
    slowers = 0
    timeouts = 0
    records = []

# Iterate through all test directories, running the above "test" function
# for each.
for testdir in tests:
    if (not os.path.isdir(testdir)) or (not os.path.isfile(testdir + "/input")):
        print("{}IGNORE {} (no input){}".format(color.darkgray,testdir,color.reset))
        continue
    f, c, t, s, fa, sl, to, re = test(testdir)
    stats.fails += f
    stats.tests += t
    stats.checks += c
    stats.skips += s
    stats.fasters += fa
    stats.slowers += sl
    stats.timeouts += to
    stats.records += re

# Print a quick summary of all tests
print("\nTest Summary")
print("{}{} tests run{}".format(color.blue,stats.tests,color.reset))
print("{}{} tests run and verified{}".format(color.boldgreen,stats.checks,color.reset))
if not stats.fails: print("{}0 tests failed{}".format(color.boldgreen,color.reset))
else:         print("{}{} tests failed{}".format(color.red,stats.fails,color.reset))
if stats.skips: print("{}{} tests skipped{}".format(color.boldyellow,stats.skips,color.reset))
if stats.fasters: print("{}{} tests ran faster".format(color.blue,stats.fasters,color.reset))
if stats.slowers: print("{}{} tests ran slower".format(color.magenta,stats.slowers,color.reset))
if stats.timeouts: print("{}{} tests timed out".format(color.red,stats.timeouts,color.reset))
print("")

return_code = stats.fails + stats.skips
if not args.permit_timeout:
    return_code += stats.timeouts

# Return nonzero only if no tests failed or were unexpectedly skipped
exit(return_code)
