import numpy as np
import theano
import time
import sys
import cPickle
from multiprocessing import Pool

import environment
import pg_network
import other_agents
import job_distribution

np.set_printoptions(threshold=10000)


def add_sample(X, y, idx, X_to_add, y_to_add):
    X[idx, 0, :, :] = X_to_add
    #print('observation of the state: ', X_to_add)
    y[idx] = y_to_add
    #print('sample print', X, y)
    
    
def add_sample_parallel(args):
    X, y, idx, X_to_add, y_to_add = args
    add_sample(X, y, idx, X_to_add, y_to_add)
    return idx, X[idx, 0, :, :], y[idx]


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job', use_cnn=False):
    print('launching')
    env = environment.Env(pa, render=False, repre=repre, end=end)

    training_list = []

    pg_learner = pg_network.PGLearner(pa, use_cnn)

    if pg_resume is not None:
        net_handle = open(pg_resume, 'r')
        net_params = cPickle.load(net_handle)
        pg_learner.set_net_params(net_params)

    if pa.evaluate_policy_name == "SJF":
        evaluate_policy = other_agents.get_sjf_action
    elif pa.evaluate_policy_name == "PACKER":
        evaluate_policy = other_agents.get_packer_action
    else:
        print("Panic: no policy known to evaluate.")
        exit(1)

    # ----------------------------
    print("Preparing for data...")
    # ----------------------------
    #All this generates the training and testing data. The taining and learning of the policy gradient model are done in the pg_network.py file.
    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)

    # print 'nw_time_seqs=', nw_len_seqs
    # print 'nw_size_seqs=', nw_size_seqs

    mem_alloc = 4

    X = np.zeros([pa.simu_len * pa.num_ex * mem_alloc, 1,
                  pa.network_input_height, pa.network_input_width],
                 dtype=theano.config.floatX)
    print('X, X.shape', X, X.shape)
    y = np.zeros(pa.simu_len * pa.num_ex * mem_alloc,
                 dtype='int32')
    #print('Y', y)

    print 'network_input_height=', pa.network_input_height
    print 'network_input_width=', pa.network_input_width

    counter = 0
    results = []
    
    pool = Pool()
    
    for train_ex in range(pa.num_ex):
        env.reset()

        for _ in range(pa.episode_max_length):
            # ---- get current state ----
            ob = env.observe()
            a = evaluate_policy(env.machine, env.job_slot1,pa)
            if counter < pa.simu_len * pa.num_ex * mem_alloc: # 72
                results.append((X, y, counter, ob, a))
                counter += 1
            ob, rew, done, info = env.step(a,repeat=True)
            #print('counter check 2', counter)
            if done:  # hit void action, exit
                print('exit, void action')
                break
            
        # # roll to next example
        # env.seq_no = (env.seq_no + 1) % env.pa.num_ex

    processed_data = pool.map(add_sample_parallel, results)
    pool.close()
    pool.join()
    
    for idx, X_sample, y_sample in processed_data:
        X[idx, 0, :, :] = X_sample
        y[idx] = y_sample
    
    num_train = int(0.8 * counter)
    # print("shape of training: ", num_train)
    num_test = int(0.2 * counter)

    X_train, X_test = X[:num_train], X[num_train: num_train + num_test]
    y_train, y_test = y[:num_train], y[num_train: num_train + num_test]
    # print('print features: ', X_train[3])
    print('shape of xtrain', X_train.shape)
    print('print labels: ', y_train, 'and test: ', y_test, y_train.shape)
    training_list.append((X_train, y_train))
    # print('the training list: ',training_list)

    # Normalization, make sure nothing becomes NaN

    # X_mean = np.average(X[:num_train + num_test], axis=0)
    # X_std = np.std(X[:num_train + num_test], axis=0)
    #
    # X_train = (X_train - X_mean) / X_std
    # X_test = (X_test - X_mean) / X_std

    # ----------------------------
    print("Start training...")
    # ----------------------------

    for epoch in xrange(pa.num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        #print('accccuracy')
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, pa.batch_size, shuffle=True):
            inputs, targets = batch
            err, prob_act = pg_learner.su_train(inputs, targets)
         #   print('prob action', prob_act)
            pg_act = np.argmax(prob_act, axis=1)
         #   print('pg action', pg_act)
            train_err += err
            train_acc += np.sum(pg_act == targets)
            train_batches += 1

        # # And a full pass over the test data:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, pa.batch_size, shuffle=False):
            inputs, targets = batch
            err, prob_act = pg_learner.su_test(inputs, targets)
            pg_act = np.argmax(prob_act, axis=1)
            test_err += err
            test_acc += np.sum(pg_act == targets)
            test_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, pa.num_epochs, time.time() - start_time))
        print("  training loss:    \t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / float(num_train) * 100))
        print("  test loss:        \t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:    \t\t{:.2f} %".format(
            test_acc / float(num_test) * 100))

        sys.stdout.flush()

        if epoch % pa.output_freq == 0:

            net_file = open(pa.output_filename + '_net_file_' + str(epoch) + '.pkl', 'wb')
            cPickle.dump(pg_learner.return_net_params(), net_file, -1)
            net_file.close()

    print("done")


def main():

    import parameters
    pa = parameters.Parameters()

    pa.simu_len = 1000  # 1000
    pa.num_ex = 100  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    pa.output_freq = 50
    pa.new_job_rate = 0.3
    pa.episode_max_length = 10000  # 2000
    pa.compute_dependent_parameters()

    pg_resume = None
    render = False
    use_cnn = False
    
    launch(pa, pg_resume, render, repre='image', end='all_done', use_cnn=False)


if __name__ == '__main__':
    main()