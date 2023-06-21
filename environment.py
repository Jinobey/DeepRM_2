from __future__ import unicode_literals

import io
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import theano
import random
import parameters
import data_collection
import slow_down_cdf

from io import open
import csv

def append_to_csv(file, data):
    with open(file, 'ab') as file:
        csv_writer = csv.writer(file)
        for row in data:
            csv_writer.writerow(row)

class Env:
    def __init__(self, pa, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image', end='no_new_job'):
       # print('repre = ',repre)
        print('initialize the environment...')
        self.job_information = []
        self.anomalous_jobs = []
        self.job_enter_time_lst = []

        self.iteration_counter = 0
        
        self.anomaly = False

        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0

        self.anomalous = False

        
        self.jobsHeaders = ['iteration','job object', 'job ID', 'job Length', 'job ressource requirement', 'job enter time', 'job start time', 'job finish time', 'job waiting time', 'current time']
        self.destroyedJobsFile = 'destroyedJobs.csv'
        self.injectJobsFile = 'injectedJobs.csv'
        self.removedJobsFile = 'removedJobs.csv'
        self.allocatedJobsFile = 'allocatedJobs.csv'
        self.allocatedJobsHeaders = ['iteration','job object', 'job ID', 'job Length', 'job ressource requirement', 'job enter time', 'job start time', 'job finish time', 'job waiting time', 'current time']


        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)


        if nw_len_seqs is None or nw_size_seqs is None:
            # generate new work
            self.nw_len_seqs, self.nw_size_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)

            self.workload = np.zeros(pa.num_res)
            print('workload', self.workload)
            for i in xrange(pa.num_res):
                self.workload[i] = \
                    np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                    float(pa.res_slot) / \
                    float(len(self.nw_len_seqs))
                print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i])) #change res slot here as well.
              #  print('workload2', self.workload)
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        self.iteration = 0
        # initialize system
        self.machine = Machine(pa)
    #    self.job_slot = JobSlot(pa)
        self.job_slot1 = JobSlot1(pa)
        self.job_slot2 = JobSlot2(pa)
        self.job_slot3 = JobSlot3(pa)        
        self.remove_slot = RemovedJobs(pa)
        self.destroyed_slot = DestroyedJobs(pa)
        #print('job slots:',self.job_slot.slot)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

        self.data_collection_instance = data_collection.Data_collection()

        #print("starting or finishing...........")
    def generate_sequence_work(self, simu_len):

        nw_len_seq = np.zeros(simu_len, dtype=int)
        nw_size_seq = np.zeros((simu_len, self.pa.num_res), dtype=int)
        #print("size seq", nw_size_seq)
        for i in range(simu_len):

            if np.random.rand() < self.pa.new_job_rate:  # a new job comes

                nw_len_seq[i], nw_size_seq[i, :] = self.nw_dist()

        return nw_len_seq, nw_size_seq
    

    def get_new_job_from_seq(self, seq_no, seq_idx):

        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                        job_len=self.nw_len_seqs[seq_no, seq_idx],
                        job_id=len(self.job_record.record),
                        enter_time=self.curr_time)
            
        #print('the new job id: ', new_job.id, 'len', new_job.len, 'enter time: ', new_job.enter_time, 'start time: ', new_job.start_time, 'finish time: ', new_job.finish_time, 'res vec?:', new_job.res_vec)
        return new_job

    def observe(self):
        if self.repre == 'image':
            #print('this is image mode')
            backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
        #    print('width of backlog: ', backlog_width)
            image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))
       #     print('image repre before: ', image_repr.shape)
            ir_pt = 0

            for i in xrange(self.pa.num_res):
                #print("Shape of image_repr: ", image_repr.shape)
               # print("Shape of machine.canvas for resource ", i, ": ", self.machine.canvas[i, :, :].shape)
              #  print("Value of res_slot: ", self.pa.res_slot)
                image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :] #change res slot to anomalous. #explore this in more details TODO
                ir_pt += self.pa.res_slot
             #   print('check the image representation pt: ', ir_pt)

                for j in xrange(self.pa.num_nw):

                    if self.job_slot1.slot[j] is not None:  # fill in a block of work
                        image_repr[: self.job_slot1.slot[j].len, ir_pt: ir_pt + self.job_slot1.slot[j].res_vec[i]] = 1

                    ir_pt += self.pa.anomalous_job_resources_upper
            #        print('check ir_pt value: ', ir_pt)
                    #print(' max job size: ', ir_pt)
                for k in xrange(self.pa.num_nw):
                    if self.job_slot2.slot[k] is not None:
                        image_repr[: self.job_slot2.slot[k].len, ir_pt: ir_pt + self.job_slot2.slot[k].res_vec[i]] = 1
                     #   append_to_csv('./image_repr.csv', image_repr)
                    ir_pt += self.pa.anomalous_job_resources_upper
                for l in xrange(self.pa.num_nw):
                    if self.job_slot3.slot[l] is not None:
                        image_repr[: self.job_slot3.slot[l].len, ir_pt: ir_pt + self.job_slot3.slot[l].res_vec[i]] = 1
                     #   append_to_csv('./image_repr.csv', image_repr)
                    ir_pt += self.pa.anomalous_job_resources_upper    
            image_repr[: self.job_backlog.curr_size / backlog_width,
                       ir_pt: ir_pt + backlog_width] = 1
           # print('image rep backlog ting: ', image_repr)
            if self.job_backlog.curr_size % backlog_width > 0:
                image_repr[self.job_backlog.curr_size / backlog_width,
                           ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1
            ir_pt += backlog_width

            image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
                                              float(self.extra_info.max_tracking_time_since_last_job)
            
          #  print('I need to know that: ', self.extra_info.time_since_last_new_job, self.extra_info.max_tracking_time_since_last_job, self.extra_info.time_since_last_new_job / \
           #                                   float(self.extra_info.max_tracking_time_since_last_job)) #TODO, might need to change this.
            ir_pt += 1
            #print('image represenation: ', image_repr, 'and shape of image: ', image_repr.shape)
            #print('check backlog: ',  )
            assert ir_pt == image_repr.shape[1]

            return image_repr

        elif self.repre == 'compact':
            #print('this is compact mode')
            compact_repr = np.zeros(self.pa.time_horizon * (self.pa.num_res + 1) +  # current work
                                    self.pa.num_nw * (self.pa.num_res + 1) +        # new work
                                    1,                                              # backlog indicator
                                    dtype=theano.config.floatX)

            cr_pt = 0

            # current work reward, after each time step, how many jobs left in the machine
            job_allocated = np.ones(self.pa.time_horizon) * len(self.machine.running_job)
            for j in self.machine.running_job:
                job_allocated[j.finish_time - self.curr_time: ] -= 1

            compact_repr[cr_pt: cr_pt + self.pa.time_horizon] = job_allocated
            cr_pt += self.pa.time_horizon

            # current work available slots
            for i in range(self.pa.num_res):
                compact_repr[cr_pt: cr_pt + self.pa.time_horizon] = self.machine.avbl_slot[:, i]
                cr_pt += self.pa.time_horizon
                #print('wtf is that: ', cr_pt, 'next to this: ', compact_repr[cr_pt: cr_pt + self.pa.time_horizon])

            # new work duration and size
            for i in range(self.pa.num_nw):

                if self.job_slot.slot[i] is None:
                    compact_repr[cr_pt: cr_pt + self.pa.num_res + 1] = 0
                    cr_pt += self.pa.num_res + 1
                else:
                    compact_repr[cr_pt] = self.job_slot.slot[i].len
                    cr_pt += 1

                    for j in range(self.pa.num_res):
                        compact_repr[cr_pt] = self.job_slot.slot[i].res_vec[j]
                        cr_pt += 1

            # backlog queue
            compact_repr[cr_pt] = self.job_backlog.curr_size
            cr_pt += 1

            assert cr_pt == len(compact_repr)  # fill up the compact representation vector

            return compact_repr

    def plot_state(self):
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in xrange(self.pa.num_res):

            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

            for j in xrange(self.pa.num_nw):

                job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))
                if self.job_slot.slot[j] is not None:  # fill in a block of work
                    job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

                plt.subplot(self.pa.num_res,
                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                            1 + i * (self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.pa.num_nw - 1:
                    skip_row += 1

        skip_row -= 1
        backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
        backlog = np.zeros((self.pa.time_horizon, backlog_width))

        backlog[: self.job_backlog.curr_size / backlog_width, : backlog_width] = 1
        backlog[self.job_backlog.curr_size / backlog_width, : self.job_backlog.curr_size % backlog_width] = 1

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_nw + 1 + 1)

        plt.imshow(backlog, interpolation='nearest', vmax=1)

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.pa.time_horizon, 1)) * \
                     self.extra_info.time_since_last_new_job / \
                     float(self.extra_info.max_tracking_time_since_last_job)

        plt.imshow(extra_info, interpolation='nearest', vmax=1)

        plt.show()     # manual
        # plt.pause(0.01)  # automatic

    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
          #  print('this is list of running job: ', self.machine.running_job)
         #   print('this is j', j.id, j.len, j.res_vec)
           # if max(j.res_vec) >= self.pa.anomalous_job_resources_lower or j.len >= self.pa.anomalous_job_len_lower_bound:
           #     reward += self.pa.delay_penalty / (float(j.len)/1.5)
           # else: 
            reward += self.pa.delay_penalty / float(j.len)
            #print('delay penality', reward)

        for j in self.job_slot.slot:
            if j is not None:
                #if max(j.res_vec) >= self.pa.anomalous_job_resources_lower or j.len >= self.pa.anomalous_job_len_lower_bound:
                #    reward += self.pa.hold_penalty / (float(j.len)/1.5)
                #else: 
                reward += self.pa.hold_penalty / float(j.len)

        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.len)
             #   print('dismiss penality', reward)
        for j in self.remove_slot.slot:
            if j is not None:
                #if max(j.res_vec) >= self.pa.anomalous_job_resources_lower or j.len >= self.pa.anomalous_job_len_lower_bound:
                #    reward += self.pa.removed_penality / float(j.len)
                #else: 
                reward += self.pa.removed_penality / (float(j.len) * 1.5)
                #reward += self.pa.removed_penality / float(j.len)
        
        for j in self.machine.destroyed_jobs:
            if max(j.res_vec) >= self.pa.anomalous_job_resources_lower:     
                reward += self.pa.destroy_penality / (-(float(max(j.res_vec)))) #maybe divide by 2.
            elif j.len >= self.pa.anomalous_job_len_middle_bound:
                reward += self.pa.destroy_penality / (-(float(j.len)))
             #   print('reward: ', reward, 'max res: ', max(j.res_vec))
            else:
                reward += self.pa.destroy_penality * 2#/ #((float(j.len)/1.5))

        
        #access remove reward
                #what kind of formula to use for penality?
        #print('reward in the reward function: ', reward)


        return reward

    def step(self, a, s=None, repeat=False):
        #print("Active child processes:- ", multiprocessing.active_children())
        #print("Active threads:", threading.enumerate())
        status = None
        list_of_actions = []
        done = False
        reward = 0
        info = None
        list_of_actions.append(a)
        append_to_csv('./output_re.csv', [[a]])
        print('current timestep: ', self.curr_time, "and actions taken: ", a, "job_slots: ", self.job_slot1.slot, "remove slots: ", self.remove_slot.slot, "backlog size: ", self.job_backlog.curr_size)
        
      #  print('the action a is: ', a , 'at time: ', self.curr_time)

        if a == self.pa.num_nw:  # explicit void action # if action taken is 5 then move on
            status = 'MoveOn'

        elif a > self.pa.num_nw and a <= (self.pa.num_nw*2):
            a = a - self.pa.num_nw - 1
            if self.job_slot1.slot[a] is None:
                status = 'MoveOn'
            else:    
                migrate_1_2 = self.machine.migrate_1_to_2(self.job_slot1, self.curr_time, a, self.job_slot1.slot[a], self.job_slot2, self.pa, self.job_record)
                if not migrate_1_2:
                    status = 'MoveOn'
                else:
                    self.job_info = [self.iteration,self.job_slot.slot[a], self.job_slot.slot[a].id, self.job_slot.slot[a].len, self.job_slot.slot[a].res_vec,self.job_slot.slot[a].enter_time,self.job_slot.slot[a].start_time, self.job_slot.slot[a].finish_time, self.job_slot.slot[a].waiting_time,  self.curr_time]
                    self.data_collection_instance.append_job_to_csv(self.removedJobsFile,self.jobsHeaders,self.job_info)
                    status = 'migrate1_2'

        elif (a >= (self.pa.num_nw * 2) + 1) and (a <= (self.pa.num_nw * 3)): # should be + 1 not 2
            a = a - (self.pa.num_nw * 2) - 1
            if self.job_slot1.slot[a] is None:
                status = 'MoveOn'
            else:
                migrate_1_3 = self.machine.migrate_1_to_3(self.job_slot1, self.curr_time, a, self.job_slot1.slot[a], self.job_slot3, self.pa, self.job_record)
                if not migrate_1_3:
                    status = 'MoveOn'
                else:
                    status = 'migrate1_3'   

        elif (a >= (self.pa.num_nw * 3) + 1) and (a <= (self.pa.num_nw * 4)):
            a = a - (self.pa.num_nw * 3) - 1
            allocated2 = self.machine.allocate_job_machine2(self.job_slot2.slot[a], self.curr_time)
            if not allocated2:  # implicit void action
                status = 'MoveOn'
            
            else:
                self.job_info = [self.iteration,self.job_slot.slot[a], self.job_slot.slot[a].id, self.job_slot.slot[a].len, self.job_slot.slot[a].res_vec,self.job_slot.slot[a].enter_time,self.job_slot.slot[a].start_time, self.job_slot.slot[a].finish_time, self.job_slot.slot[a].waiting_time,  self.curr_time]
                self.data_collection_instance.append_job_to_csv(self.allocatedJobsFile,self.jobsHeaders,self.job_info)
                status = 'Allocate2'

        elif (a >= (self.pa.nunm_nw * 4 + 1) and (a <= (self.pa.num_nw * 5))):  
            a = a - (self.pa.num_nw * 4) - 1
            allocated3 = self.machine.allocate_job_machine3(self.job_slot3.slot[a], self.curr_time)
            if not allocated3:  # implicit void action
                status = 'MoveOn'
            else:
                self.job_info = [self.iteration,self.job_slot.slot[a], self.job_slot.slot[a].id, self.job_slot.slot[a].len, self.job_slot.slot[a].res_vec,self.job_slot.slot[a].enter_time,self.job_slot.slot[a].start_time, self.job_slot.slot[a].finish_time, self.job_slot.slot[a].waiting_time,  self.curr_time]
                self.data_collection_instance.append_job_to_csv(self.allocatedJobsFile,self.jobsHeaders,self.job_info)
                status = 'Allocate3'


        elif (a >= (self.pa.num_nw * 5) + 1) and (a <= (self.pa.num_nw * 6)): # should be + 1 not 2
            a = a - (self.pa.num_nw * 5) - 1
            if self.job_slot3.slot[a] is None:
                status = 'MoveOn'
            else:
                migrate_3_1 = self.machine.migrate_3_to_1(self.job_slot3, self.curr_time, a, self.job_slot3.slot[a], self.job_slot1, self.pa, self.job_record)
                if not migrate_3_1:
                    status = 'MoveOn'
                else:
                    status = 'migrate3_1' 

        elif (a >= (self.pa.num_nw * 6) + 1) and (a <= (self.pa.num_nw * 7)): # should be + 1 not 2
            a = a - (self.pa.num_nw * 6) - 1
            if self.job_slot2.slot[a] is None:
                status = 'MoveOn'
            else:
                migrate_2_1 = self.machine.migrate_2_to_1(self.job_slot2, self.curr_time, a, self.job_slot2.slot[a], self.job_slot1, self.pa, self.job_record)
                if not migrate_2_1:
                    status = 'MoveOn'
                else:
                    status = 'migrate2_1' 

        elif (a >= (self.pa.num_nw * 7) + 1) and (a <= (self.pa.num_nw * 8)): # should be + 1 not 2
            a = a - (self.pa.num_nw * 7) - 1
            if self.job_slot3.slot[a] is None:
                status = 'MoveOn'
            else:
                migrate_2_3 = self.machine.migrate_2_to_3(self.job_slot2, self.curr_time, a, self.job_slot2.slot[a], self.job_slot3, self.pa, self.job_record)
                if not migrate_2_3:
                    status = 'MoveOn'
                else:
                    status = 'migrate2_3' 

        elif (a >= (self.pa.num_nw * 8) + 1) and (a <= (self.pa.num_nw * 9)): # should be + 1 not 2
            a = a - (self.pa.num_nw * 8) - 1
            if self.job_slot2.slot[a] is None:
                status = 'MoveOn'
            else:
                migrate_3_2 = self.machine.migrate_3_to_2(self.job_slot3, self.curr_time, a, self.job_slot3.slot[a], self.job_slot2, self.pa, self.job_record)
                if not migrate_3_2:
                    status = 'MoveOn'
                else:
                    status = 'migrate3_2' 

        elif self.job_slot1.slot[a] is None:  # implicit void action # if no actions then move on        
          #  print("Status : Value of a:", a, 'and job slots: ', self.job_slot.slot[a])
            status = 'MoveOn'
            
        else:
            allocated = self.machine.allocate_job_machine1(self.job_slot1.slot[a], self.curr_time)
            if not allocated:  # implicit void action
                status = 'MoveOn'
            else:
                self.job_info = [self.iteration,self.job_slot.slot[a], self.job_slot.slot[a].id, self.job_slot.slot[a].len, self.job_slot.slot[a].res_vec,self.job_slot.slot[a].enter_time,self.job_slot.slot[a].start_time, self.job_slot.slot[a].finish_time, self.job_slot.slot[a].waiting_time,  self.curr_time]
                self.data_collection_instance.append_job_to_csv(self.allocatedJobsFile,self.jobsHeaders,self.job_info)
                status = 'Allocate1'

        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new jobs
            self.seq_idx += 1


            if self.end == "no_new_job":  # end of new job sequence
                if self.seq_idx >= self.pa.simu_len:
                    #print('idx ting', self.seq_idx)
                    done = True
                    print('no new jobss')
            elif self.end == "all_done":  # everything has to be finished
                if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   len(self.machine.running_job2) == 0 and \
                   len(self.machine.running_job3) == 0 and \
                   all(s is None for s in self.job_slot1.slot) and \
                   all(s is None for s in self.job_backlog.backlog) and \
                    all(s is None for s in self.job_slot2.slot) and \
                        all(s is None for s in self.job_slot3.slot):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    print('It ran for too long: destroyed list: ', self.machine.destroyed_jobs, 'job slots: ', self.job_slot.slot , 'remove list:', self.remove_slot.slot, 'backlog: ', self.job_backlog.curr_size)
                    done = True
                   # print('ran for too long')

            if not done:
                print('back log size',self.job_backlog.curr_size)
                #print('the backlog: ', self.job_backlog.curr_size, 'the job slots: ', self.job_slot.slot, 'the remove slots', self.remove_slot.slot, 'destroy slots: ', self.destroyed_slot.slot[:10])# self.job_backlog.backlog[0:10])
                if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                    new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)
                    if new_job.len > 0:  # a new job comes
                        to_backlog = True
                        for i in xrange(self.pa.num_nw):
                            if self.job_slot1.slot[i] is None:  # put in new visible job slots
                                self.job_slot1.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job #put the newly arrived job in the records. 
                                to_backlog = False
             #                   print('queue content: ', self.job_slot.slot, 'and new job arriving w id,len,resource and slot: ', new_job.id, new_job.len, new_job.res_vec, self.job_slot.slot[i])
                                break
                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
            #                    print('Job sent to the backlog: ', new_job.id, 'and backlog information at the time: ', self.job_backlog.curr_size)
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)
                        
                        self.extra_info.new_job_comes()

            reward = self.get_reward()
            #print('the reward: ', reward)
        
        elif status == 'migrate1_2':
            self.job_slot1.slot[a] = None
            if self.job_backlog.curr_size > 0:
                self.job_slot1.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1   
        
        elif status == 'migrate1_3':
            self.job_slot1.slot[a] = None
            if self.job_backlog.curr_size > 0:
                self.job_slot1.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1  
        
        elif status == 'migrate3_1':
            self.job_slot3.slot[a] = None

        elif status == 'migrate2_1':
            self.job_slot2.slot[a] = None

        elif status == 'migrate3_2':
            self.job_slot3.slot[a] = None    
        
        elif status == 'migrate2_3':
            self.job_slot2.slot[a] = None

        elif status == 'Allocate1':
            self.job_record.record[self.job_slot1.slot[a].id] = self.job_slot1.slot[a]
            self.job_slot1.slot[a] = None
            # dequeue backlog
            if self.job_backlog.curr_size > 0:
                self.job_slot1.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1
        elif status == 'Allocate2':
            self.job_record.record[self.job_slot2.slot[a].id] = self.job_slot2.slot[a]
            self.job_slot2.slot[a] = None

        elif status == 'Allocate3':
            self.job_record.record[self.job_slot3.slot[a].id] = self.job_slot3.slot[a]
            self.job_slot3.slot[a] = None

    
            #print('reward allocate act: ', reward)

        ob = self.observe()
        info = self.job_record
        #print('the info about records: ', self.job_record.record, 'and record id', self.job_record)
        if done:
            print('final reward', reward)
            self.iteration_counter += 1
            #print("job records: ", self.job_record.record)
            print('finished!!!!')
            print('final destroyed list: ', self.machine.destroyed_jobs, 'remove list:', self.remove_slot.slot, 'backlog: ', self.job_backlog.curr_size, 'job slots: ', self.job_slot1.slot )
            final_list = map(lambda x: [self.iteration_counter] + list(x), self.machine.job_information)
            #print('the test list: ', self.machine.test_list)
            if not os.path.isfile('./output_re.csv') or os.path.getsize('./output_re.csv') == 0:
                with open('./output_re.csv', mode='ab') as file:
                    csv_writer = csv.writer(file)
                    #csv_writer.writerow(['Iteration', 'Job ID', 'Current Time', 'Start Time', 'Finish Time', 'Waiting Time', 'Job Length', 'Job Resource Vector', 'Available resource after job allocation'])
                    csv_writer.writerow(['Action Label'])

            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex
                print(' not finished!!!!')
            print('reset the environment...')
            self.reset()
        
        if self.render:
            self.plot_state()
      #  print('the reward 2', reward)
        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
       # self.job_slot = JobSlot(self.pa)
        self.job_slot1 = JobSlot1(self.pa)
        self.job_slot2 = JobSlot2(self.pa)
        self.job_slot3 = JobSlot3(self.pa)
        self.remove_slot = RemovedJobs(self.pa)
        self.destroyed_slot = DestroyedJobs(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)

        self.iteration =+ 1


class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1

        self.waiting_time = -1


class JobSlot1:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw

class JobSlot2:
    def __init__(self,pa):
        self.slot = [None] * pa.num_nw

class JobSlot3:
    def __init__(self,pa):
        self.slot = [None] * pa.num_nw


class RemovedJobs:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw
class DestroyedJobs:
    def __init__(self,pa):
        self.slot = [None] * pa.destroyed_size

class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    
    def __init__(self, pa):

        #self.test_type = slow_down_cdf.launch.

        self.removedJobsFile = 'removedJobs.csv'
        self.removedJobsHeaders = ['iteration', 'job object', 'job ID', 'job Length', 'job ressource requirement', 'job enter time', 'current time']

        self.destroyedJobsFile = 'destroyedJobs.csv'
        self.destroyedJobsHeaders = ['iteration','job object', 'job ID', 'job Length', 'job ressource requirement', 'job enter time', 'current time']

        self.injectJobsFile = 'injectedJobs.csv'
        self.injectJobsHeaders = ['iteration','job object', 'job ID', 'job Length', 'job ressource requirement', 'job enter time', 'current time']

        self.allocatedJobsFile = 'allocatedJobs.csv'
        self.allocatedJobsHeaders = ['iteration','job object', 'job ID', 'job Length', 'job ressource requirement', 'job enter time', 'job start time', 'job finish time', 'job waiting time', 'current time']


        self.anomalous_len = pa.anomalous_job_len_middle_bound
        self.epochs_count = pa.num_epochs

        self.anomalous_res = pa.anomalous_job_resources_lower



        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon

        self.backlog_size = pa.backlog_size

        self.res_slot = pa.res_slot
        self.res_slot1 = pa.res_slot1
        self.res_slot2 = pa.res_slot2
        self.res_slot3 = pa.res_slot3

        self.avbl_slot1 = np.ones((self.time_horizon, self.num_res)) * self.res_slot1 #change res slot to anomalous.
        self.avbl_slot2 = np.ones((self.time_horizon, self.num_res)) * self.res_slot2 #change res slot to anomalous.
        self.avbl_slot3 = np.ones((self.time_horizon, self.num_res)) * self.res_slot3 #change res slot to anomalous.
       
        self.removed_jobs = []
        self.destroyed_jobs = []
        
        self.running_job = []
        self.running_job2 = []
        self.running_job3 = []


        self.job_information = []
        self.test_list = []
        self.resource_utilisation = []
        self.re_allocated_jobs = []

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))
       # print('the canvas: ', self.canvas, 'and shape: ', self.canvas.shape)
    
    # implement two logics one for with s and one without. 
    def destroy_job(self, job_slot, curr_time, job_backlog, a, job, job_record, pa, destroyed_slot):
        job.finish_time = curr_time + job.len
        self.destroyed_jobs.append(job)
        print('destroyed jobs list: ', self.destroyed_jobs)
        

    def migrate_1_to_2(self, job_slot1, curr_time, a, job, job_slot2, pa, job_record):
        migrate_1_2 = False
        for i in xrange(pa.num_nw):    
            if job_slot2.slot[i] is None:
                migrate_1_2 = True    
                job_slot2.slot[i] = job_slot1.slot[a]
                job_record.record[job_slot2.slot[i].id] = job_slot1.slot[a]

                break

        return migrate_1_2

    def migrate_2_to_1(self, job_slot2, curr_time, a, job, job_slot1, pa, job_record):
        migrate_2_1 = False
        for i in xrange(pa.num_nw):    
            if job_slot1.slot[i] is None:
                migrate_2_1 = True    
                job_slot1.slot[i] = job_slot2.slot[a]
                job_record.record[job_slot1.slot[i].id] = job_slot2.slot[a]

                break

        return migrate_2_1

    def migrate_1_to_3(self, job_slot1, curr_time, a, job, job_slot3, pa, job_record):
        migrate_1_3 = False
        for i in xrange(pa.num_nw):    
            if job_slot3.slot[i] is None:
                migrate_1_3 = True    
                job_slot3.slot[i] = job_slot1.slot[a]
                job_record.record[job_slot3.slot[i].id] = job_slot1.slot[a]

                break

        return migrate_1_3
    
    def migrate_3_to_2(self, job_slot3, curr_time, a, job, job_slot2, pa, job_record):
       # print('timestep for removal: ', curr_time)
        migrate_3_2 = False
        for i in xrange(pa.num_nw):    
            if job_slot2.slot[i] is None:
                migrate_3_2 = True    
                job_slot2.slot[i] = job_slot3.slot[a]
                job_record.record[job_slot2.slot[i].id] = job_slot3.slot[a]

                break

        return migrate_3_2

    def migrate_2_to_3(self, job_slot2, curr_time, a, job, job_slot3, pa, job_record):
       # print('timestep for removal: ', curr_time)
        migrate_2_3 = False
        for i in xrange(pa.num_nw):    
            if job_slot3.slot[i] is None:
                migrate_2_3 = True    
                job_slot3.slot[i] = job_slot2.slot[a]
                job_record.record[job_slot3.slot[i].id] = job_slot2.slot[a]

                break

        return migrate_2_3
    

    def migrate_3_to_1(self, job_slot3, curr_time, a, job, job_slot1, pa, job_record):
       # print('timestep for removal: ', curr_time)
        migrate_3_1 = False
        for i in xrange(pa.num_nw):    
            if job_slot1.slot[i] is None:
                migrate_3_1 = True    
                job_slot1.slot[i] = job_slot3.slot[a]
                job_record.record[job_slot1.slot[i].id] = job_slot3.slot[a]

                break

        return migrate_3_1
            
    
    def allocate_job_machine1(self, job, curr_time):
        allocated = False
        #print('allocated job:', job.id, 'at time: ', curr_time, 'with res and len ', job.len, job.res_vec)
        for t in xrange(0, self.time_horizon - job.len):
            new_avbl_res = self.avbl_slot1[t: t + job.len, :] - job.res_vec
            if np.all(new_avbl_res[:] >= 0):
                allocated = True
                self.avbl_slot1[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                job.waiting_time = job.start_time - job.enter_time
                self.running_job.append(job)
                
                self.resource_utilisation.append((curr_time, new_avbl_res))
                # update graphical representation
                print("Job", job.id, "allocated at time", curr_time, 'start at time: ', job.start_time, "and finishes at time", job.finish_time, "job length: ", job.len, 'job resource vector', job.res_vec, 'and new avble ressource: ', new_avbl_res)
               
                self.job_information.append((job.id, curr_time,job.start_time,job.finish_time,job.waiting_time,job.len, job.res_vec, new_avbl_res))
                self.test_list.append((job.id, job.len, job.res_vec))
                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time
                #print('start time: ', canvas_start_time, 'end time: ', canvas_end_time)

                for res in xrange(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0] #TODO probably need to change that
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color
                break
            
        return allocated
    

    def allocate_job_machine2(self, job, curr_time):
        allocated = False
        #print('allocated job:', job.id, 'at time: ', curr_time, 'with res and len ', job.len, job.res_vec)
        for t in xrange(0, self.time_horizon - job.len):
            new_avbl_res = self.avbl_slot2[t: t + job.len, :] - job.res_vec
            if np.all(new_avbl_res[:] >= 0):
                allocated = True
                self.avbl_slot2[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                job.waiting_time = job.start_time - job.enter_time
                self.running_job2.append(job)
                
                self.resource_utilisation.append((curr_time, new_avbl_res))
                # update graphical representation
                print("Job", job.id, "allocated at time", curr_time, 'start at time: ', job.start_time, "and finishes at time", job.finish_time, "job length: ", job.len, 'job resource vector', job.res_vec, 'and new avble ressource: ', new_avbl_res)
               
                self.job_information.append((job.id, curr_time,job.start_time,job.finish_time,job.waiting_time,job.len, job.res_vec, new_avbl_res))
                self.test_list.append((job.id, job.len, job.res_vec))
                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time
                #print('start time: ', canvas_start_time, 'end time: ', canvas_end_time)

                for res in xrange(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0] #TODO probably need to change that
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color
                   # print('and print the start-end: ', canvas_start_time, canvas_end_time, canvas_end_time-canvas_start_time, 'and color', new_color)
                break
            
        return allocated
    

    def allocate_job_machine3(self, job, curr_time):
        allocated = False
        #print('allocated job:', job.id, 'at time: ', curr_time, 'with res and len ', job.len, job.res_vec)
        for t in xrange(0, self.time_horizon - job.len):
            new_avbl_res = self.avbl_slot3[t: t + job.len, :] - job.res_vec
            if np.all(new_avbl_res[:] >= 0):
                allocated = True
                self.avbl_slot3[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                job.waiting_time = job.start_time - job.enter_time
                self.running_job3.append(job)
                
                self.resource_utilisation.append((curr_time, new_avbl_res))
                # update graphical representation
                print("Job", job.id, "allocated at time", curr_time, 'start at time: ', job.start_time, "and finishes at time", job.finish_time, "job length: ", job.len, 'job resource vector', job.res_vec, 'and new avble ressource: ', new_avbl_res)
               
                self.job_information.append((job.id, curr_time,job.start_time,job.finish_time,job.waiting_time,job.len, job.res_vec, new_avbl_res))
                self.test_list.append((job.id, job.len, job.res_vec))
                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time
                #print('start time: ', canvas_start_time, 'end time: ', canvas_end_time)

                for res in xrange(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0] #TODO probably need to change that
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color
                   # print('and print the start-end: ', canvas_start_time, canvas_end_time, canvas_end_time-canvas_start_time, 'and color', new_color)
                break
            
        return allocated
    
    def time_proceed(self, curr_time):

        self.avbl_slot1[:-1, :] = self.avbl_slot[1:, :] #TODO probably copy that stuff for the others.
        self.avbl_slot[-1, :] = self.res_slot

        for job in self.running_job:

            if job.finish_time <= curr_time:
                print('job finish time: ', job.finish_time, 'current time', curr_time, 'job id',job.id )
                self.running_job.remove(job)
        

        for job2 in self.running_job2:

            if job2.finish_time <= curr_time:
                print('job finish time: ', job2.finish_time, 'current time', curr_time, 'job id',job2.id )
                self.running_job.remove2(job2)


        for job3 in self.running_job3:

            if job3.finish_time <= curr_time:
                print('job finish time: ', job3.finish_time, 'current time', curr_time, 'job id',job3.id )
                self.running_job3.remove(job3)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_nw = 5
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print "New job is backlogged."

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print "- Backlog test passed -"


def test_compact_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in xrange(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print "- Elapsed time: ", end_time - start_time, "sec -"


def test_image_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in xrange(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print "- Elapsed time: ", end_time - start_time, "sec -"


if __name__ == '__main__':
    test_backlog()
    test_compact_speed()
    test_image_speed()
    print('test')