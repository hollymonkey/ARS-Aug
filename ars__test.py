#Code in this file is adapted from
#the Augmented Random Search method
#https://github.com/modestyachts/ARS 

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)
from transformations import get_transformations
import PIL.Image
import numpy as np
import time
import parser
import time
import os
import numpy as np
import logz
import ray
import utilss
import optimizerss
from policies import *
import socket
import random
from shared_noise import *

def gett_dataset(dataset, reduced):
    if dataset == 'cifar10':
        (Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (Xtr, ytr), (Xts, yts) = datasets.cifar100.load_data()
    else:
        raise Exception('Unknown dataset %s' % dataset)
    if reduced:
        ix = np.random.choice(len(Xtr), 4000, False)
        Xtr = Xtr[ix]
        ytr = ytr[ix]
    ytr = utils.to_categorical(ytr)
    yts = utils.to_categorical(yts)
    return (Xtr, ytr), (Xts, yts)

(Xtr, ytr), (Xts, yts) = gett_dataset('cifar10', True)
transformations = get_transformations(Xtr)

class Operation:
    def __init__(self, types_softmax, probs_softmax, magnitudes_softmax):
        
        
        self.type = types_softmax
        t = transformations[self.type]
        self.prob = probs_softmax
        m = magnitudes_softmax
        self.magnitude = m*(t[2]-t[1]) + t[1]
        self.transformation = t[0]

    def __call__(self, X):
        _X = []
        for x in X:
            if np.random.rand() < self.prob:
                x = PIL.Image.fromarray(x)
                x = self.transformation(x, self.magnitude)
            _X.append(np.array(x))
        return np.array(_X)

    def __str__(self):
        return 'Operation %2d (P=%.3f, M=%.3f)' % (self.type, self.prob, self.magnitude)

class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in self.operations:
            X = op(X)
        return X

    def __str__(self):
        ret = ''
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations)-1:
                ret += '\n'
        return ret

def autoaugment(subpolicies,X, y):
        while True:
            ix = np.arange(len(X))
            np.random.shuffle(ix)
            for i in range(4000):
                _ix = ix[i*128:(i+1)*128]
                _X = X[_ix]
                _y = y[_ix]
                subpolicy = np.random.choice(subpolicies)
                _X = _X.astype(np.float32) / 255
                yield _X, _y

                
@ray.remote
def get_dataset(subpolicies,dataset, reduced):
    if dataset == 'cifar10':
        (Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (Xtr, ytr), (Xts, yts) = datasets.cifar100.load_data()
    else:
        raise Exception('Unknown dataset %s' % dataset)
    if reduced:
        ix = np.random.choice(len(Xtr), 4000, False)
        Xtr = Xtr[ix]
        ytr = ytr[ix]
    ytr = utils.to_categorical(ytr)
    yts = utils.to_categorical(yts)
    x = input_layer = layers.Input(Xtr.shape[1:])
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = models.Model(input_layer, x)
    model.compile(optimizers.SGD(decay=1e-4), 'categorical_crossentropy', ['accuracy'])
    gen = autoaugment(subpolicies,Xtr,ytr)
    model.fit_generator(
           gen, 4000, 120, verbose=0, use_multiprocessing=True)
    accuracy = model.evaluate(Xts, yts, verbose=0)[1]
    return accuracy




@ray.remote
class Worker(object):

    def __init__(self, env_seed,
                 env_name='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
       # print('delatas = ', self.deltas)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            
        else:
            raise NotImplementedError
        
        self.w_policy = self.policy.get_weights()
        self.delta_std = delta_std
        self.rollout_length = rollout_length

        
    def get_weights_plus_stats(self):
        assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()
    
    def get_answer(self):
        cnt = 1 
        type_now = []
        prob = []
        magnitude = []
        for a in self.w_policy:
            a[0] = 1 / (1 + np.exp(-a[0]))
            if cnt % 3 ==1 :
                 type_now.append(int(a[0] * 16))
            elif cnt % 3 ==2:
                 prob.append(a[0])
            else:
                 magnitude.append(a[0])
            cnt = cnt + 1
        cnt = 0
        subpolicies = []
        operations = []
        for i in range(5):
            operations = []
            for j in range(1):
                operations.append(Operation(type_now[cnt],prob[cnt],magnitude[cnt]))
                cnt = cnt + 1
                operations.append(Operation(type_now[cnt],prob[cnt],magnitude[cnt]))
            subpolicies.append(Subpolicy(*operations))
        return subpolicies

    def rollout(self, shift = 0., rollout_length = None):
        
        if rollout_length is None:
            rollout_length = self.rollout_length   
        cnt = 1 
        type_now = []
        prob = []
        magnitude = []
        for a in self.w_policy:
            a[0] = 1 / (1 + np.exp(-a[0]))
            if cnt % 3 ==1 :
                 type_now.append(int(a[0] * 16))
            elif cnt % 3 ==2:
                 prob.append(a[0])
            else:
                 magnitude.append(a[0])
            cnt = cnt + 1
        cnt = 0
        subpolicies = []
        operations = []
        for i in range(5):
            operations = []
            for j in range(1):
                operations.append(Operation(type_now[cnt],prob[cnt],magnitude[cnt]))
                cnt = cnt + 1
                operations.append(Operation(type_now[cnt],prob[cnt],magnitude[cnt]))
            subpolicies.append(Subpolicy(*operations))
        a = get_dataset.remote(subpolicies,'cifar10',True)
        acc = ray.get(a)
        steps = 0
        ob = 0
        action = self.policy_params
        total_reward = acc  

            
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                self.policy.update_filter = False

                reward, r_steps = self.rollout(shift = 0., rollout_length = 10)
                rollout_rewards.append(reward)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)
                self.policy.update_filter = True
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps  = self.rollout(shift = shift)
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift = shift) 
                steps += pos_steps + neg_steps
                rollout_rewards.append([pos_reward, neg_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    
class ARSLearner(object):

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=320, 
                 deltas_used=320,
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123):

        logz.configure_output_dir(logdir)
        logz.save_params(params)
        
        self.timesteps = 0
        self.action_size = 30
        self.ob_size = 1
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]


        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        
        else:
            raise NotImplementedError
            
        self.optimizer = optimizerss.SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
            

        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]
        
       


        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)
        

        rollout_rewards, deltas_idx = [], [] 

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
      

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']


        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]

        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()

        g_hat, count = utilss.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat
        

    def train_step(self):
        
        g_hat = self.aggregate_rollouts()                    
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return

    def train(self, num_iter):

        start = time.time()
        for i in range(num_iter):
            
            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)           
            print('iter ', i,' done')

            if ((i + 1) % 10 == 0):
                
                rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.savez(self.logdir + "/lin_policy_plus", w)
                
                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()
                
            t1 = time.time()

            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            self.policy.observation_filter.clear_buffer()

            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]

            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]

            ray.get(increment_filters_ids)            
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)
                        
        return 

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ob_dim = 1
    ac_dim = 30

    policy_params={'type':'linear',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}
    print('policy_params = ', policy_params)

    ARS = ARSLearner(env_name=params['env_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'])
        
    ARS.train(params['n_iter'])
       
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)

    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')

    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    local_ip = socket.gethostbyname(socket.gethostname())
  
    ray.init(redis_address= local_ip + ':6379')
    
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)

