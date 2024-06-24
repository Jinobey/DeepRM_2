import yaml
import numpy as np
import pandas as pd
import os
import csv
# from memory_profiler import profile

class Data_collection:
        def convert_parameter_to_yaml(self,pa):
                parameters = {                
                        'output_filename': pa.output_filename,
                        'num_epochs':pa.num_epochs,
                        'simu_len': pa.simu_len,
                        'num_ex':pa.num_ex,
                        'anomalous_job_rate':pa.anomalous_job_rate,
                        'output_freq':pa.output_freq,
                        'num_seq_per_batch':pa.num_seq_per_batch,
                        'episode_max_length':pa.episode_max_length,
                        'num_res':pa.num_res,
                        'num_nw':pa.num_nw,
                        'time_horizon':pa.time_horizon,
                        'max_job_len':pa.max_job_len,
                        'res_slot1':pa.res_slot1,
                        'res_slot2':pa.res_slot2,
                        'res_slot3':pa.res_slot3,
                        'max_job_size':pa.max_job_size,
                        'anomalous_job_len_upper_bound':pa.anomalous_job_len_upper_bound,
                        'anomalous_job_len-middle_bound': pa.anomalous_job_len_middle_bound,
                        'anomalous_job_len_lower_bound':pa.anomalous_job_len_lower_bound,
                        'anomalous_job_resources_upper':pa.anomalous_job_resources_upper,
                        'anomalous_job_resources_lower':pa.anomalous_job_resources_lower,
                        'backlog_size':pa.backlog_size,
                        'max_track_since_new':pa.max_track_since_new,
                        'job_num_cap':pa.job_num_cap,
                        'new_job_rate':pa.new_job_rate,
                        'discount':pa.discount,
                        'dist':pa.dist,
                        'backlog_width':pa.backlog_width,
                        'network_input_height':pa.network_input_height,
                        'network_input_width':pa.network_input_width,
                        'network_compact_dim':pa.network_compact_dim,
                        'network_output_dim':pa.network_output_dim,
                        'delay_penalty':pa.delay_penalty,
                        'hold_penalty':pa.hold_penalty,
                        'dismiss_penalty':pa.dismiss_penalty,
                        'num_frames':pa.num_frames,
                        'lr_rate':pa.lr_rate,
                        'rms_rho':pa.rms_rho,
                        'rms_eps':pa.rms_eps,
                        'unseen':pa.unseen,
                        'batch_size':pa.batch_size,
                        'evaluate_policy_name':pa.evaluate_policy_name 
                        }
                
                with open('params.yaml', 'w') as file:
                        yaml.dump(parameters, file)

        def csv_check(self, file, headers):
                if not os.path.isfile(file) or os.path.getsize(file) == 0:
                        with open(file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(headers)
        def append_job_to_csv(self, file, header, job_info):
                        if not os.path.isfile(file) or os.path.getsize(file) == 0:
                                with open(file, 'a') as f:
                                        writer = csv.writer(f)
                                        writer.writerow(header)
                        else:
                                with open(file, 'a') as d:
                                        writer = csv.writer(d)
                                        writer.writerow(job_info)