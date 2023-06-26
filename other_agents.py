import numpy as np
import parameters
import environment


def get_packer_action(machine, job_slot1):
        align_score = 0
        act = len(job_slot1.slot)  # if no action available, hold

        for i in xrange(len(job_slot1.slot)):
            new_job = job_slot1.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot3[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)

                    if tmp_align_score > align_score:
                        align_score = tmp_align_score
                        act = i
        return act
     
        
def get_sjf_action(machine, job_slot1, pa):
        sjf_score1 = 0
        sjf_score2 = 0
        sjf_score3 = 0
        act = len(job_slot1.slot)  # if no action available, hold

        for i in xrange(len(job_slot1.slot)):
            new_job = job_slot1.slot[i]
            
            if new_job is not None:  # there is a pending job
                #print('new job features: ', new_job.len, new_job.res_vec)
                avbl_res1 = machine.avbl_slot1[:new_job.len, :]
                res_left1 = avbl_res1 - new_job.res_vec

                avbl_res2 = machine.avbl_slot2[:new_job.len, :]
                res_left2 = avbl_res2 - new_job.res_vec

                avbl_res3 = machine.avbl_slot3[:new_job.len, :]
                res_left3 = avbl_res3 - new_job.res_vec

                #cloud node for long jobs
                if np.all(res_left1[:] >= 0) and new_job.len >= pa.anomalous_job_len_middle_bound and np.all(new_job.res_vec < pa.anomalous_job_resources_lower):  # enough resource to allocate
                    print('resleft1', res_left1)
                    tmp_sjf_score1 = 1 / float(new_job.len)

                    if tmp_sjf_score1 > sjf_score1:
                        sjf_score1 = tmp_sjf_score1
                        act = i
                        print('action returned cloud node for long jobs', act, 'and job info: ', new_job.id, new_job.len, new_job.res_vec)
                        return act
                #normal edge node for normal jobs
                elif np.all(res_left2[:] >= 0) and new_job.len < pa.anomalous_job_len_middle_bound and np.all(new_job.res_vec < pa.anomalous_job_resources_lower):  # enough resource to allocate
                    print('resleft2', res_left2)
                    tmp_sjf_score1 = 1 / float(new_job.len)

                    if tmp_sjf_score1 > sjf_score1:
                        sjf_score1 = tmp_sjf_score1
                        act = len(job_slot1.slot) + i + 1
                        print('act return normal edge node for normal jobs', act,'and job info: ', new_job.id, new_job.len, new_job.res_vec)
                        return act
                
                #cloud node for expensive jobs
                elif np.all(res_left3[:] >= 0) and np.any(new_job.res_vec >= pa.anomalous_job_resources_lower):  # enough resource to allocate
                    print('resleft3', res_left3)
                    tmp_sjf_score1 = 1 / float(new_job.len)

                    if tmp_sjf_score1 > sjf_score1:
                        sjf_score1 = tmp_sjf_score1
                        act = i
                        act = len(job_slot1.slot) * 2 + i + 1
                        print('act return cloud node for expensive jobs', act, 'and job info: ', new_job.id, new_job.len, new_job.res_vec)
                        return act
        return act       

# add either a if statement to put a threshold on the length of the job. Or add threshold in action to decide to that if too high of a score then moveOn. 


def get_packer_sjf_action(machine, job_slot1, knob):  # knob controls which to favor, 1 to packer, 0 to sjf

        combined_score = 0
        act = len(job_slot1.slot)  # if no action available, hold

        for i in xrange(len(job_slot1.slot)):
            new_job = job_slot1.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot3[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)
                    tmp_sjf_score = 1 / float(new_job.len)

                    tmp_combined_score = knob * tmp_align_score + (1 - knob) * tmp_sjf_score

                    if tmp_combined_score > combined_score:
                        combined_score = tmp_combined_score
                        act = i
        return act


def get_random_action(job_slot1):
    num_act = len(job_slot1.slot * 2) + 1  # if no action available,
    act = np.random.randint(num_act)
    return act
