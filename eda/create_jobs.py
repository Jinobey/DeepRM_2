from parameters import Parameters
from job_distribution import Dist
from data_collection import Data_collection

def generate_and_save_jobs_to_csv(num_jobs=100000, csv_file_path='data/jobs.csv'):
    # Initialize parameters, distribution, and data collection objects
    pa = Parameters()
    pa.compute_dependent_parameters()
    dist = Dist(pa.num_res, pa.max_job_size, pa.max_job_len, 
                pa.anomalous_job_len_upper_bound, pa.anomalous_job_len_lower_bound,
                pa.anomalous_job_len_middle_bound, pa.anomalous_job_resources_lower,
                pa.anomalous_job_resources_upper)
    data_collector = Data_collection()

    # Prepare CSV file and headers
    headers = ['Job Duration'] + [f'Job Resource {i+1}' for i in range(pa.num_res)]
    data_collector.csv_check(csv_file_path, headers)

    # Generate jobs and append to CSV
    for _ in range(num_jobs):
        nw_len, nw_size = dist.bi_model_dist()
        job_info = [nw_len] + list(nw_size)
        data_collector.append_job_to_csv(csv_file_path, headers, job_info)

# Example usage
if __name__ == "__main__":
    generate_and_save_jobs_to_csv()