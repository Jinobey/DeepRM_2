import numpy as np
import math

import job_distribution


class Parameters:

    anomalous_job_len_upper_bound = 29#19 #16
    anomalous_job_len_lower_bound = 19#11
    anomalous_job_len_middle_bound = 23#15
    def __init__(self):

        self.output_filename = 'data/tmp'

        self.num_epochs = 3001 #10000         # number of training epochs
        self.simu_len = 30             # length of the busy cycle that repeats itself
        self.num_ex = 20                # number of sequences

        self.anomalous_job_rate = 0.5

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10 #5 #10    # number of sequences to compute baseline
        self.episode_max_length = 3000 #200 # enforcing an artificial terminal

        self.num_res = 2               # number of resources in the system
        self.num_nw = 5#5#5                # maximum allowed number of work in the queue

        #These parameters are bugy. Also max_job_len should always be smaller than time horizon which should be % backlogsize = 0
        self.time_horizon = 30#40#20         # number of time steps in the graph
        self.max_job_len = 18#10#25#15          # maximum duration of new jobs
        
        self.res_slot = 15 #used to be 10.             # maximum number of available resource slots
        self.res_slot1 = 10
        self.res_slot2 = 10
        self.res_slot3 = 15

        self.max_job_size = 10         # maximum resource request of new work
        
        #fix this to make it available everywhere.
        self.anomalous_job_len_upper_bound = 29#19#39 #16
        self.anomalous_job_len_lower_bound = 19#11#26
        self.anomalous_job_len_middle_bound = 23#31#15#31

        self.anomalous_job_resources_upper = 15
        self.anomalous_job_resources_lower = 11
        self.normal_job_upper = 10

        self.backlog_size = 120         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40          # maximum number of distinct colors in current work graph

        self.new_job_rate = 1#1#0.7        # lambda in new job arrival Poisson Process

        self.discount = 0.95#1#0.1           # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len, self.anomalous_job_len_upper_bound, self.anomalous_job_len_lower_bound, self.anomalous_job_len_middle_bound, self.anomalous_job_resources_lower, self.anomalous_job_resources_upper)

        
        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot1 + self.res_slot2 + self.res_slot3 +
             self.anomalous_job_resources_upper * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job #maybe add num_nw * 4 for all actions? #self.max_job_size instead of anomalous job resources upper

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw * 3 + 1#self.num_nw + 2  # + 1 for void action +2 for remove action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001#0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example
        

        #the way jobs are created are in the blocks so we can visualize them in the same manner. 

        # supervised learning mimic policy
        self.batch_size = 512
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot1 + self.res_slot2 + self.res_slot3 +
             self.anomalous_job_resources_upper * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job  self.max_job_size #487

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw * 3 + 1 # + 1 for void action

