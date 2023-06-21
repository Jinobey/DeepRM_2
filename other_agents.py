import numpy as np
import parameters
import environment


def get_packer_action(machine, job_slot):
        align_score = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in xrange(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)

                    if tmp_align_score > align_score:
                        align_score = tmp_align_score
                        act = i
        return act

def is_anomalous(job_length, job_lengths, percentile_threshold=95):
    if not job_lengths:
         return False
    percentile_value = np.percentile(job_lengths,percentile_threshold)
    return job_length > percentile_value


def get_sjf_action(machine, job_slot1,job_slot2,job_slot3):
        sjf_score = 0
        act = len(job_slot1.slot)  # if no action available, hold

        for i in xrange(len(job_slot1.slot)):
            new_job = job_slot1.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_sjf_score = 1 / float(new_job.len)

                    if tmp_sjf_score > sjf_score:
                        sjf_score = tmp_sjf_score
                        act = i
        else: 
             for j in xrange(len)                
        
def get_sjf_action(machine, job_slot):
        sjf_score1 = 0
        sjf_score2 = 0
        sjf_score3 = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in xrange(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res1 = machine.avbl_slot1[:new_job.len, :]
                res_left1 = avbl_res1 - new_job.res_vec

                avbl_res2 = machine.avbl_slot2[:new_job.len, :]
                res_left2 = avbl_res2 - new_job.res_vec

                avbl_res3 = machine.avbl_slot3[:new_job.len, :]
                res_left3 = avbl_res3 - new_job.res_vec

                if np.all(res_left1[:] >= 0) and new_job.len >= parameters.Parameters.anomalous_job_len_lower_bound and np.all(new_job.res_vec < parameters.Parameters.anomalous_job_resources_lower):  # enough resource to allocate

                    tmp_sjf_score1 = 1 / float(new_job.len)

                    if tmp_sjf_score1 > sjf_score1:
                        sjf_score1 = tmp_sjf_score1
                        act = i
                        return act
                
                elif np.all(res_left2[:] >= 0) and new_job.len < parameters.Parameters.anomalous_job_len_lower_bound and np.all(new_job.res_vec < parameters.Parameters.anomalous_job_resources_lower):  # enough resource to allocate

                    tmp_sjf_score2 = 1 / float(new_job.len)

                    if tmp_sjf_score2 > sjf_score2:
                        sjf_score2 = tmp_sjf_score2
                        act = len(job_slot.slot) * 2 + i + 1
                        return act
                
                elif np.all(res_left3[:] >= 0) and np.any(new_job.res_vec >= parameters.Parameters.anomalous_job_resources_lower):  # enough resource to allocate

                    tmp_sjf_score3 = 1 / float(new_job.len)

                    if tmp_sjf_score3 > sjf_score3:
                        sjf_score3 = tmp_sjf_score3
                        act = i
                        act = len(job_slot.slot) * 3 + i + 1
                        return act
        return act       
        


# choose the shortest job first in terms of length, if there is enough resource to accomodate it.
def get_sjf_action(machine, job_slot1,job_slot2,job_slot3,removed_slots, pa):
        anomalous_jobs = []
        destroyed_job = []
        sjf_score = 0
        #percentile_threshold = 95
        
        act = len(job_slot1.slot)  # if no action available, hold # changed slot to remove.
        print('a class act :")', act)
        for i in xrange(len(job_slot1.slot)): #loop through the slots to see the jobs (used to be slot not remove slot)
            new_job = job_slot1.slot[i]
            if new_job is not None:  # there is a pending job
                #allocate job if there is a pending job
                avbl_res = machine.avbl_slot1[:new_job.len, :]

                res_left = avbl_res - new_job.res_vec
                #print('resource left', res_left[:], res_left)
                if np.all(res_left[:] >= 0) and np.all(new_job.res_vec < pa.anomalous_job_resources_lower):  # enough resource to allocate
                    if new_job.len < parameters.Parameters.anomalous_job_len_lower_bound or new_job in machine.re_allocated_jobs:
                        #check if there are enough resource to allocate the job
                        tmp_sjf_score = 1 / float(new_job.len)
                        if tmp_sjf_score > sjf_score: #if the score is greater than the current score update the current score to the one we just had and set act to current idx.
                            sjf_score = tmp_sjf_score
                            act = i
                            #print('action decided: ', act)
                        elif new_job in machine.re_allocated_jobs and tmp_sjf_score > sjf_score:
                             sjf_score = tmp_sjf_score
                             print('re allocated jobs list: ', machine.re_allocated_jobs)
                             machine.re_allocated_jobs.remove(new_job)
                             print('re allocated jobs list after removal: ', machine.re_allocated_jobs)
                             act = i
                    elif new_job.len >= parameters.Parameters.anomalous_job_len_middle_bound:
                         destroyed_job.append((new_job, new_job.id, new_job.len, new_job.res_vec))
                         print('value of action and i', act, i)
                         act = len(job_slot.slot) * 2 + i + 1 # +2 else if act is 0 then it becomes a remove action.
                         print('action returned in destroy or ', i, act)
                         return act
                    elif new_job.len >= parameters.Parameters.anomalous_job_len_lower_bound and new_job.len < parameters.Parameters.anomalous_job_len_middle_bound and new_job not in machine.re_allocated_jobs: 
                        #print('the act returned', act, 'and job targeted: ', new_job.id, 'print the I', i)
                        anomalous_jobs.append((new_job, new_job.id, new_job.len))
                        machine.re_allocated_jobs.append(new_job)
                        print('the re allocate list: ', machine.re_allocated_jobs, 'and the new job: ', new_job)
                        #print('value of action and i', act, i)
                        act = len(job_slot.slot) + i + 1#job_slot.remove_slot
                        print('action returned in remove or 6', i, act, 'and the remove queue: ', removed_slots.slot)
                        return act
                elif np.any(new_job.res_vec >= pa.anomalous_job_resources_lower):
                    destroyed_job.append((new_job, new_job.id, new_job.len, new_job.res_vec))
                    print('value of action and i', act, i)
                    act = len(job_slot.slot) * 2 + i + 1 # +2 else if act is 0 then it becomes a remove action.
                    print('action returned in destroy because of too much asked resources. ', i, act)
                    return act
                     
        else:
            if new_job is None and any(slot is not None for slot in removed_slots.slot): #removed_slots.slot.count(None) == 1:
                for j in xrange(len(removed_slots.slot)):
                     if removed_slots.slot[j] is not None:
                          #machine.re_allocated_jobs.append
                          act = len(job_slot.slot) * 3 + j + 1
                          print('the action of re inject: ', act)
                          return act
            elif act == len(job_slot.slot):
                 return act
        print('action taken: ',act)
        return act #return act which is the index of the selected job in job slot

# add either a if statement to put a threshold on the length of the job. Or add threshold in action to decide to that if too high of a score then moveOn. 


def get_packer_sjf_action(machine, job_slot, knob):  # knob controls which to favor, 1 to packer, 0 to sjf

        combined_score = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in xrange(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)
                    tmp_sjf_score = 1 / float(new_job.len)

                    tmp_combined_score = knob * tmp_align_score + (1 - knob) * tmp_sjf_score

                    if tmp_combined_score > combined_score:
                        combined_score = tmp_combined_score
                        act = i
        return act


def get_random_action(job_slot):
    num_act = len(job_slot.slot * 4) + 1  # if no action available,
    act = np.random.randint(num_act)
    return act
