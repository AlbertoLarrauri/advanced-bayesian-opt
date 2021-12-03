import numpy as np
import re
import itertools
import os, getpass
import time, datetime
import csv


class Tester:

    def __init__(self, mt=None):
        # the parameter mt defines the configuration multi-threading when the execution of the simulations is started;
        # e.g. mt='mt2' will reserve 2 cores for the simulations, mt=None will use no multi-threading, etc.

        # self.limits = { 'name_property_1': [ lower_bound_1, upper_bound_1 ], ..., 'name_property_n' : [ lower_bound_n, upper_bound_n' }
        # self.limits stores the monitored output parameters with their allowed min/max values; after an experiment
        # is performed, the actual measurements of the output parameters are checked against their allowed min/max values
        self.limits = dict()

        # self.in_values = { 'in_bit_vector': [0, 127], 'temperature': [-40, 100, 150], ... }
        # self.in_values stores the input parameters with their values specified by the user; the simlist file required
        # to run an experiment is generated out of the input parameters values; if for every input parameter exactly
        # one possible value is specified then the generated simlist file will contain exactly one run line, otherwise
        # one run line for each possible combination of the various input parameters values
        self.in_values = dict()

        # dictionary of run output results; the output results of a run is a dictionary of the form
        # { 'name_property_1': value_1, ..., 'name_property_n' : value_n }
        self.out_results = dict()

        # dictionary of runs; a run is a dictionary of input parameters
        self.runs = dict()

        self.mt = mt
        
        self.timeout = False

		

    def load_limits(self, filename):
        """
        Populate self.limits with the out-parameters and their variation ranges specified in 'filename'.
        """

        try:
            f = open(filename, "r")
            lines = f.read()
            spec = re.compile(r"%spec\s*=\s*([^;]*)")
            spec_search = spec.search(lines)
            if spec_search:
                spec_contents = spec_search.group(1)
            else:
                print(
                    "Could not parse spec"
                )
                return

            parameter_re = re.compile(r"\w*\s*=>\s*{([^}]*)}")
            parameters = parameter_re.findall(spec_contents)

            label_re = re.compile(r"label\s*=>\s*\"([^\"]*)\"\s*,")
            min_re = re.compile(r"min\s*=>\s*([^,]*)\s*")
            max_re = re.compile(r"max\s*=>\s*([^,]*)\s*")

            # populate self.limits
            for parameter in parameters:
                label_search = label_re.search(parameter)
                if not label_search:
                    print(
                        "Could not parse spec"
                    )
                    return
                label = label_search.group(1)
                max = max_re.search(parameter).group(1) if max_re.search(parameter) else 0
                min = min_re.search(parameter).group(1) if min_re.search(parameter) else 0
                self.limits[label] = [float(min), float(max)]

            f.close()

        except IOError:
            print(
                'There is no file named', filename
            )

    def load_in_values(self, filename, create_simlist=True):
        self.in_values = {}
        """
        Load values for the input parameters into self.in_values from the file 'filename'.
        NOTE: set 'create_simlist' to False when the simulation with the specifications from limits.txt and
              input_configuration.csv has been already performed and you just want now to load the output results
              without running the simulation again.
        """

        try:
            f = open(filename, "r")
            lines = f.readlines()
            for line in lines:
                if not (line.lstrip().startswith('#') or line.lstrip() == ''):
                    # 'line' is not a comment line and it is not empty

                    tokens = line.split(',')
                    name = tokens[0]
                    type = tokens[1]
                    ds = tokens[2]

                    values = []
                    if type == 'string':
                        if ds == 'enum':
                            for i in range(3, len(tokens)):
                                values.append(tokens[i].strip())
                        else:
                            print("Could not parse in-values")
                            return
                    elif type == 'bitvector':
                        if ds == 'interval':
                            begin = int(tokens[3])
                            end = int(tokens[4])
                            for i in range(begin, end + 1):  # end+1 because range(a,b) generates the numbers up to b-1
                                values.append(i)
                        else:
                            print("Could not parse in-values")
                            return
                    else:
                        print("Could not parse in-values")
                        return
                    if (name == 'vdda_evr') and ('vdda_hpbg' in self.in_values):
                        # we are loading 'vdda_evr' and 'vdda_hpbg' has been already loaded: store the union of
                        # the values of the two in-params because they must always have the same value
                        self.in_values[name] = self.in_values['vdda_hpbg'] = sorted(
                            list(set(values).union(set(self.in_values['vdda_hpbg']))), key=float)
                    elif (name == 'vdda_hpbg') and ('vdda_evr' in self.in_values):
                        # we are loading 'vdda_hpbg' and 'vdda_evr' has been already loaded: store the union of
                        # the values of the two in-params because they must always have the same value
                        self.in_values[name] = self.in_values['vdda_evr'] = sorted(
                            list(set(values).union(set(self.in_values['vdda_evr']))), key=float)
                    else:
                        self.in_values[name] = values
            f.close()
        except IOError:
            print(
                'There is no file named', filename
            )

        # generate runs from the input parameter values loaded above
        self._generate_runs(create_simlist)

    def _generate_runs(self, create_simlist):
        """
        Generate a set of simulation runs out of the input parameter values loaded by self.load_in_values() and
        populate self.runs with the generated runs.

        If create_simlist is set to True then this method also creates in the 'pattern' folder the corresponding
        'simlist' perl-script necessary to invoke the simulator.
        If a simlist file already exists, this is renamed by appending the last modify date at its name before creating
        a new simlist file.

        This is a private method which is invoked by self.load_in_values().
        """

        assert "TRIMBG" in self.in_values, "missing TRIMBG input values"
        assert "TRIMCUR" in self.in_values, "missing TRIMCUR input values"
        assert "T" in self.in_values, "missing Temperature input values"
        assert "models" in self.in_values, "missing Corner input values"

        if create_simlist:
            simlist_path = os.path.join('.', 'pattern', 'simlist')

            # check if a simlist exists already and, if it does, make a backup
            do_backup(simlist_path)

            # create a new simlist-file to be filled in with the run-lines
            f_simlist = open(simlist_path, "w")

            # add the (fixed) header to the simlist-file
            f_simlist.write('#!/bin/perl\n'
                            '\n'
                            '$ENV{FCV_POWER_SUPPLY}="VDDPD!;VDDA_HPBG!;VDDA_EVR!;VSS!;VREF!";\n'
                            'my ($ip_filelist) = gf_ip_filelist("hpbg");\n'
                            '\n'
                            'patterns: hpbg_startup_trimall\n'
                            '\n')

        # compute the number of possible runs; this is the product of the numbers of possible values of the in-parameters;
        # 'vdda_evr' and 'vdda_hpbg', if both present, have always the same value, hence only one of them has to be considered
        # in the product
        n_runs = 1
        in_values_ = dict()
        wildcard_key = '_wildcard_'
        if 'vdda_evr' in self.in_values:
            n_runs = len(self.in_values['vdda_evr'])
            in_values_[wildcard_key] = self.in_values['vdda_evr']
        elif 'vdda_hpbg' in self.in_values:
            n_runs = len(self.in_values['vdda_hpbg'])
            in_values_[wildcard_key] = self.in_values['vdda_hpbg']
        for in_param_name in self.in_values:
            # update n_runs only if the current in_parameter is different from 'vdda_evr' and 'vdda_hpbg' since
            # these parameters have always the same value and they have been considered above at the n_runs initialization
            if (in_param_name != 'vdda_evr') and (in_param_name != 'vdda_hpbg'):
                n_runs *= len(self.in_values[in_param_name])
                in_values_[in_param_name] = self.in_values[in_param_name]

        assert n_runs == 1, "Error, multiple runs defined. Can only perform one run at a time"

        # compute and add a run-line for every combination of values from the specified value ranges of the input params

        count = 0
        keys = list(in_values_.keys())
        index_TRIMBG = keys.index('TRIMBG')
        index_TRIMCUR = keys.index('TRIMCUR')
        index_T = keys.index('T')
        index_models = keys.index('models')
        index_wildcard = keys.index(wildcard_key) if wildcard_key in keys else None
        # compute the cartesian product of the input parameter values and process each tuple one at a time to generate
        # a run-line
        for run_tuple in itertools.product(*in_values_.values()):
            count += 1
            run_name = 'RUN{:0{}d}'.format(count, len(str(n_runs)))

            # initialize entry for a new run in the runs-dictionary
            self.runs[run_name] = {"TRIMBG": bit_format(run_tuple[index_TRIMBG], 7),
                                   "TRIMCUR": bit_format(run_tuple[index_TRIMCUR], 5),
                                   "models": run_tuple[index_models],
                                   "T": run_tuple[index_T]}

            # initialize run-line for the simlist file
            run_line = 'T {} HPBGnl("TRIMBG_{}","TRIMCUR_{}") models_{} T{}C'.format(
                run_name,
                bit_format(run_tuple[index_TRIMBG], 7),
                bit_format(run_tuple[index_TRIMCUR], 5),
                run_tuple[index_models],
                run_tuple[index_T]
            )

            # add all other specified parameters to the new runs-dictionary entry and run-line
            for i, param_value in enumerate(run_tuple):
                if i not in [index_TRIMBG, index_TRIMCUR, index_T, index_models]:
                    if i == index_wildcard:
                        # handle the special case of 'vdda_evr' and 'vdda_hpbg' having both the same value
                        if 'vdda_evr' in self.in_values:
                            self.runs[run_name]['vdda_evr'] = param_value
                            run_line = run_line + ' vdda_evr_{}'.format(param_value)
                        if 'vdda_hpbg' in self.in_values:
                            self.runs[run_name]['vdda_hpbg'] = param_value
                            run_line = run_line + ' vdda_hpbg_{}'.format(param_value)
                    else:
                        self.runs[run_name][keys[i]] = param_value
                        run_line = run_line + ' {}_{}'.format(keys[i], param_value)

            # finally, add all (fixed) configuration file names to the run-line
            run_line = run_line + ' $ip_filelist ip_cfg_hpbg save_sigs_lvl1{}\n'.format('' if self.mt is None else (' ' + self.mt))

            if create_simlist:
                # add the run-line to the simlist-file
                f_simlist.write(run_line)

        # make sure that all runs have been generated
        assert count == n_runs

        if create_simlist:
            f_simlist.close()

    def run_simulation(self):
        # the shell command to run the simulation

        assert self.runs, "No input parameters loaded, thus no available runs to be executed!"

        # check if a measurements file 'parameters.csv' from a previous experiment is available and, if it does,
        # make a backup
        do_backup(os.path.join('.', 'RESULTS', 'TRAIN_BGP', 'TITAN', 'parameters.csv'))

        # check if individual measurements files of the single runs are available and, if yes, make a backup of the
        # folder containing all individual measurements files
        user_name = getpass.getuser()
        pattern_results_folder = '/opt/fwtmp/{0}/fcv/advanced/ip_hpbg_c40fla/nodm/default/ws_{0}/fcv_c40fla_bgp/RESULTS/TRAIN_BGP/TITAN/hpbg_startup_trimall'.format(
            user_name)
        do_backup(pattern_results_folder)

        do_backup('simulation.log')

        cmd = "nohup fcv -titan 'hpbg_startup_trimall/.*' > simulation.log 2>&1 &"   # '> simulation.log 2>&1' will redirect the output from stdout (and stderr) to 'simulation.log'
        #cmd = "nohup fcv -titan 'hpbg_startup_trimall/.*' > simulation.log 2>&1"   # removed '&' from the end of the command since according to Roland fcv is run anyway in the background

        return_code = os.system(cmd)
        if return_code:
            print('Oooops... it seems that the simulator has not been launched!')
        else:
            print('Simulator launched successfully!')

        print('\nCheck "./simulation.log" for logs of the fcv execution')

    def load_out_results(self, timeout=900, waiting_tolerance_threshold=100):
        # load the measurements of the monitored output parameters during/after the execution of a simulation
        #
        # 'waiting_tolerance_threshold' indicates the number of missing/uncompleted simulation runs below which the user
        # is asked whether to re-launch the fcv-simulator for these runs; useful when the simulation fails for some
        # runs due to some (unclear) IT reasons

        # 'parameters.csv' stores the output measurements (and the input params) of *all* runs at one place;
        # the output measurements (but not also the input params) are also stored individually for each single run
        # separately in the folders ..ws_<username>/simulation/fcv_c40fla_bgp/RESULTS/TRAIN_BGP/TITAN/hpbg_startup_trimall/<RUN_NAME from simlist>/titan.tr.measure
        delay = 0
        output_measurements_path = os.path.join('.', 'RESULTS', 'TRAIN_BGP', 'TITAN', 'parameters.csv')

        while True:
            if os.path.exists(output_measurements_path):
                try:
                    with open(output_measurements_path, 'r') as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=';')
                        count = 0
                        # completed_runs = set()
                        for row in reader:
                            if row['Run'] not in self.out_results:
                                self.out_results[row['Run']] = dict()
                            if row['Parameter Label'] in self.limits:
                                if not row['Parameter Label'] in self.out_results[row['Run']]:
                                    count += 1
                                    # completed_runs.add(row['Run'])
                                    self.out_results[row['Run']][row['Parameter Label']] = float(row['Value'])
                            else:
                                print('\nWARNING: Parameter "{}" measured in "parameters.csv" for run="{}" but not specified in "limits.txt"!'.format(row['Parameter Label'], row['Run']))

                        # check if all out-params specified in "limits.txt" have been measured in every run
                        # for out_param in self.limits:
                        #     for run in sorted(self.out_results):
                        #         if out_param not in self.out_results[run]:
                        #             print('\nWARNING: out-parameter "{}" specified by the user in "limits.txt" but not available for run="{}" in the measurements file "parameters.csv"!'.format(out_param, run))

                        if count < len(self.runs)*len(self.limits):
                            delay = delay + 10
                            if delay >= timeout:
                                print('Simulation took longer than timeout \n')
                                
                                self.timeout = True

                                return
                            time.sleep(10)
                            # (re)initialize self.out_results for a new read
                            self.out_results = dict()
                            continue
                        else:
                            # count == len(self.runs)*len(self.limits)
                            # we are done, all measurements have been logged and loaded
                            return

                # error handling
                except Exception as e:
                    print(repr(e))
                    return
            else:
                # user_input = raw_input('\nMeasurements file "parameters.csv" not even initialized yet!\nAbort waiting? (y/n): ')
                # if user_input.lower() == 'y' or user_input.lower() == 'yes':
                #     print('Waiting aborted.\n')
                #     return
                delay = delay + 10
                if delay >= timeout:
                    print('Simulation took longer than timeout \n')
                    self.timeout = True
                    return
                time.sleep(10)
                continue

    def evaluate_out_results(self, with_logging=True):
        # compare self.out_results to self.limits;
        # log results evaluation to 'simulation.log' if 'with_logging' is set to True.

        if with_logging:
            logFile = open('simulation.log', 'a')
        else:
            logFile = None

        count_passed = 0

        myPrint('\n\n############################## Output Parameters Evaluation ##############################\n',
                logFile)
        for run in sorted(self.out_results):
            myPrint('Run = {} ({}):'.format(run, self.runs[run]), logFile)
            run_passed = True
            for out_param in self.out_results[run]:
                if self.limits[out_param][0] <= self.out_results[run][out_param] <= self.limits[out_param][1]:
                    myPrint('  [passed]: {} = {} in [{}, {}]'.format(out_param, self.out_results[run][out_param],
                                                                     self.limits[out_param][0],
                                                                     self.limits[out_param][1]), logFile)
                else:
                    myPrint('  [failed]: {} = {} NOT in [{}, {}]'.format(out_param, self.out_results[run][out_param],
                                                                         self.limits[out_param][0],
                                                                         self.limits[out_param][1]), logFile)
                    run_passed = False
            myPrint('{}!\n'.format("Passed" if run_passed else "Failed"), logFile)
            count_passed += run_passed
        myPrint('Total: {}/{} runs passed\n'.format(count_passed, len(self.out_results)), logFile)
        myPrint('##################################### Evaluation End #####################################\n', logFile)

        if with_logging:
            logFile.close()


def bit_format(val, n):
    # convert 'val' into a binary string with n bits if not already in that format
    # 'val' can be an integer or a string containing either an integer or already a binary value
    if type(val) is int:
        assert 0 <= val < 2 ** n, '{} cannot be represented by {} bits'.format(val, n)
        val = '0b{:0{}b}'.format(val, n)
    elif type(val) is str:
        if val.isdigit():
            if len(val) == n:
                assert set(val) == {'0', '1'}, '{} cannot be represented by {} bits'.format(val, n)
                val = '0b' + val
            else:
                val = int(val)
                assert 0 <= val < 2 ** n, '{} cannot be represented by {} bits'.format(val, n)
                val = '0b{:0{}b}'.format(val, n)
        else:
            assert val.startswith('0b') and set(val[2:]).union({'0', '1'}) == {'0',
                                                                               '1'}, 'invalid in-parameter value: {}'.format(
                val)
    else:
        assert False, 'cannot convert {} to {}-bits binary'.format(val, n)

    return val


def do_backup(fileorfolder_path):
    # check if a file or a directory exists already under the path 'fileorfolder_path' and, if it does, make a backup
    # by appending the last modify date to the file/folder name
    if os.path.exists(fileorfolder_path):
        modify_time = time.strftime("_%Y%m%d_%H%M%S", time.strptime(time.ctime(os.path.getmtime(fileorfolder_path))))
        # separate file name from file extension (of course if it's a file)
        split_path = os.path.splitext(fileorfolder_path)
        # rename file/folder name
        os.rename(fileorfolder_path, split_path[0] + modify_time + split_path[1])


def myPrint(string, logFile):
    print(string)
    if logFile is not None:
        logFile.write(string + '\n')
