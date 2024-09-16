def read_most_recent_snopt_hist():

    # import
    import numpy as np
    import os

    
    # =========================== Iteration data =========================== :
    current = 0
    snopt_files = [f for f in os.listdir('.') if f.endswith('SNOPT_summary.out')]
    if len(snopt_files) != 1:
        raise ValueError('There should be only one SNOPT_summary.out file in the current directory')

    snopt_file = snopt_files[0]

    # snopt_print_files = [f for f in os.listdir('.') if f.endswith('SNOPT_print.out')]
    # if len(snopt_print_files) != 1:
    #     raise ValueError('There should be only one SNOPT_print.out file in the current directory')
    # snopt_print_file = snopt_print_files[0]

    # with open(snopt_files[0], 'r', errors='ignore') as f:
    with open(snopt_files[0], 'r', encoding='latin-1') as f:
    # with open(snopt_files[0], 'a+') as f:
    #     # f.write('S')
    #     f.flush()
    #     os.fsync(f.fileno())
    #     # f.close()
        lines = f.readlines()
    # f = open(snopt_files[0], 'a+')
    # f.flush()
    # os.fsync(f.fileno())
    # import sys
    # sys.stdout.flush()
    # import time
    # print('SLEEP')
    # time.sleep(10)

    # with open(snopt_files[0], 'r') as f:
    #     lines = f.readlines()
    lines.reverse()
    rev_lines = lines
    # print('ldkfngld',lines)
    for i, line in enumerate(rev_lines, 0):
        if ('SNOPTC EXIT' in line):
            line_exit = line
            current = i + 2
            line = rev_lines[current]
            try:
                major = int(line[:6].strip())
            except:
                major = -1
            break
    rev_opt = []
    rev_feas = []
    rev_major = []
    rev_nnf = []
    # rev_nsup = []
    rev_merit = []
    rev_pen = []
    unconstrained = False
    # print(major)
    # exit()

    while (major > 0):
        line = rev_lines[current]
        # print(line)
        # print(line[:6])
        try:
            major = int(line[:6].strip())
        except:
            current = current + 1
            print('ERROR:', line)
            continue
        rev_major.append(major)
        rev_nnf.append(int(line[25:29]))
        try:
            rev_feas.append(float(line[31:38]))
        except:
            unconstrained = True
            rev_feas.append(0.0)
        rev_opt.append(float(line[40:47]))
        # print(float(line[40:47]))
        rev_merit.append(float(line[49:62]))
        # rev_nsup.append(int(line[62:69]))
        # pen = line[70:77]
        # # print(len(pen))
        # if pen in (' '*7, ''):
        #     rev_pen.append(0.)
        # else:
        #     rev_pen.append(float(line[70:77]))
        current = current + 1
        if major % 10 == 0:
            current = current + 2

    rev_major.reverse()
    major = np.array(rev_major)
    rev_feas.reverse()
    feas = np.array(rev_feas)
    rev_opt.reverse()
    opt = np.array(rev_opt)
    rev_merit.reverse()
    merit = np.array(rev_merit)
    # rev_nsup.reverse()
    # nsup = np.array(rev_nsup)
    # rev_pen.reverse()
    # pen = np.array(rev_pen)
    rev_nnf.reverse()
    nnf = np.array(rev_nnf)
    nnf = np.append([1, ], np.ediff1d(nnf))
    # POSTPROCESSING FOR PENALTY NEEDS TO BE INCLUDE IN THE LATER PART OF THE SCRIPT

    # print(np.cumsum(nnf))
    # print(nsup, len(nsup))
    # print(merit, len(merit))
    # print(pen, len(pen))
    # exit()

    # dash = CaddeeDash()
    # dash.save_python_object('optimality', opt)
    # dash.save_python_object('feasibility', feas)
    # dash.save_python_object('num_superbasics', nsup)
    # dash.save_python_object('merit', merit)
    print('MAJOR',major)
    save_dict = {}

    try:
        major_repeated = np.repeat(major, nnf)
        feas_repeated = np.repeat(feas, nnf)
        opt_repeated = np.repeat(opt, nnf)
        merit_repeated = np.repeat(merit, nnf)
        # nsup_repeated = np.repeat(nsup, nnf)

        major_repeated = np.append([0, 0, 0], major_repeated)
        major_repeated = np.append(major_repeated, [major_repeated[-1]])
        feas_repeated = np.append([feas_repeated[0]]*3, feas_repeated)
        feas_repeated = np.append(feas_repeated, [feas_repeated[-1]])
        opt_repeated = np.append([opt_repeated[0]]*3, opt_repeated)
        opt_repeated = np.append(opt_repeated, [opt_repeated[-1]])
        merit_repeated = np.append([merit_repeated[0]]*3, merit_repeated)
        merit_repeated = np.append(merit_repeated, [merit_repeated[-1]])
        # nsup_repeated = np.append([nsup_repeated[0]]*3, nsup_repeated)
        # nsup_repeated = np.append(nsup_repeated, [nsup_repeated[-1]])
        # print(major_repeated)
        # print(nnf)
        # print(len(major_repeated))

        if unconstrained:
            major_repeated = major_repeated[1:]
            feas_repeated = feas_repeated[1:]
            opt_repeated = opt_repeated[1:]
            merit_repeated = merit_repeated[1:]

        LINE_EXIT = line_exit
        OPTIMALITY = opt_repeated
        FEASIBILITY = feas_repeated
        MERIT = merit_repeated
        MAJOR = major_repeated
    except:
        NUM_F_CALLS = [-1]
        NUM_VECTORIZED_F_CALLS = [-1]
        NUM_DF_CALLS = [-1]
        NUM_VECTORIZED_DF_CALLS = [-1]
        OPTIMALITY = False
        FEASIBILITY = False
        MERIT = False
        MAJOR = False
        LINE_EXIT = line_exit
        save_dict['NUM_F_CALLS'] = NUM_F_CALLS
        save_dict['NUM_F_CALLS'] = NUM_F_CALLS
        save_dict['NUM_VECTORIZED_F_CALLS'] = NUM_VECTORIZED_F_CALLS
        save_dict['NUM_DF_CALLS'] = NUM_DF_CALLS
        save_dict['NUM_VECTORIZED_DF_CALLS'] = NUM_VECTORIZED_DF_CALLS


    # OPTIMIZATION
    save_dict['OPTIMALITY'] = OPTIMALITY
    save_dict['FEASIBILITY'] = FEASIBILITY
    save_dict['MERIT'] = MERIT
    save_dict['MAJOR'] = MAJOR
    save_dict['LINE_EXIT'] = LINE_EXIT

    # os.remove(snopt_print_file)
    # os.remove(snopt_file)

    return OPTIMALITY, LINE_EXIT

def pad_opt_data(history_f_calls, history_opt):
    if abs(len(history_opt) - len(history_f_calls)) == 0:
        pass
    elif abs(len(history_opt) - len(history_f_calls)) > 50:
        no = len(history_opt)
        nm = len(history_f_calls)
        raise ValueError(f'# opt != # major ({no} != {nm})')
    else:
        case = len(history_opt) > len(history_f_calls)
        if case:
            shorter = history_f_calls
            longer = history_opt
        else:
            longer = history_f_calls
            shorter = history_opt

        while len(shorter) != len(longer):
            shorter.insert(0, shorter[0])
            # print('lskdmfls')

        if case:
            history_f_calls = shorter
            history_opt = longer
        else:
            history_f_calls = longer
            history_opt = shorter
    return history_f_calls, history_opt

if __name__ == '__main__':
    results = read_most_recent_snopt_hist()