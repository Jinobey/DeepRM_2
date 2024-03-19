import numpy as np
import parameters

class Dist:

    def __init__(self, num_res, max_nw_size, job_len, anomalous_job_len_upper_bound, anomalous_job_len_lower_bound, anomalous_job_len_middle_bound, anomalous_resource_lower, anomalous_resource_upper):
        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.job_len = job_len
        self.anomalous_job_len_upper = anomalous_job_len_upper_bound
        self.anomalous_job_len_middle = anomalous_job_len_middle_bound
        self.anomalous_job_len_lower = anomalous_job_len_lower_bound

        self.anomalous_resource_lower = anomalous_resource_lower
        self.anomalous_resource_upper = anomalous_resource_upper
        self.anomalous_resource_rate = 0.05 #0.15#0.05 for RL training

        self.job_small_chance = 0.75#0.3
        self.anomalous_job_rate =  0.05 #0.05 for RL training

        self.job_len_big_lower = job_len * 2 / 3 
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5

        self.dominant_res_lower = max_nw_size / 2
        self.dominant_res_upper = max_nw_size

        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5#2

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self):
        random_val = np.random.rand()

        # -- job length --
        if random_val < self.job_small_chance and random_val > self.anomalous_job_rate:  # small job
            print('rand val', random_val)
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
            print('small job', nw_len)
        elif random_val < self.anomalous_job_rate:
            print('rand valx', random_val)
            nw_len = np.random.randint(self.anomalous_job_len_lower, self.anomalous_job_len_upper) 
            #nw_len = max(nw_len, self.time_horizon)
            print('anomalous job', nw_len)
        else:  # big job
            print('rand val', random_val)
            print('test2')
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)
            print('big job', nw_len)

        nw_size = np.zeros(self.num_res)
        
        #print('size of nw', nw_size)

        # -- job resource request --
        dominant_res = np.random.randint(0, self.num_res)
        random_res = np.random.rand()
        print('the random val for res', random_res)
        for i in range(self.num_res):
            if i == dominant_res:
                if random_res > self.anomalous_job_rate:    
                    nw_size[i] = np.random.randint(self.dominant_res_lower,
                                                self.dominant_res_upper + 1)
                else:
                    nw_size[i] = np.random.randint(self.anomalous_resource_lower,
                                                   self.anomalous_resource_upper)
            
            else:
                nw_size[i] = np.random.randint(self.other_res_lower,
                                               self.other_res_upper + 1)
        print('the resource vector req: ', nw_size)
        #print('size of nw', nw_size)
        return nw_len, nw_size

    def anomalous_dist(self):
        nw_len = np.random.randint(self.job_len_big_upper, self.job_len_big_upper * 2)
        nw_size = np.random.randint(self.dominant_res_upper, self.dominant_res_upper * 2, size=self.num_res)
        return nw_len, nw_size


def generate_sequence_work(pa, seed=42):

    np.random.seed(seed)

    simu_len = pa.simu_len * pa.num_ex

    nw_dist = pa.dist.bi_model_dist

    nw_anomalous_dist = pa.dist

    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)

    for i in range(simu_len):

        if np.random.rand() < pa.new_job_rate:  # a new job comes
            
            nw_len_seq[i], nw_size_seq[i, :] = nw_dist()
            #print('new job came ! ')

    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq,
                             [pa.num_ex, pa.simu_len, pa.num_res])

    return nw_len_seq, nw_size_seq
