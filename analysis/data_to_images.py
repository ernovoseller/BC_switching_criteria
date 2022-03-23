"""
Save start/end images and get some stats about saved episodes from running any policy in Gym-Cloth.
"""

import os
import argparse
from PIL import Image
import pickle
import numpy as np

def get_start_end(observation, test_sample, save_folder):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    start = observation[0]
    size = len(observation)
    end = observation[size -1]
    fname = os.path.join(save_folder, '%d_start.png' % (test_sample))
    im = Image.fromarray(start.astype('uint8'))
    im.save(fname)
    fname = os.path.join(save_folder, '%d_end.png' % (test_sample))
    im = Image.fromarray(end.astype('uint8'))
    im.save(fname)
    
def get_coverage_stats(iteration_info):
    last_iter = iteration_info[len(iteration_info)-1]
    improvement_percent = last_iter['actual_coverage']/last_iter['start_coverage']
    print(last_iter['start_coverage'])
    print(last_iter['actual_coverage'])
    final_coverage = last_iter['actual_coverage']
    actions = len(iteration_info)
    return improvement_percent, final_coverage, actions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--file', type=str, default='../gym-cloth/logs/Name_of_file_with_episode_rollouts.pkl')
    parser.add_argument('-s', '--save_folder', type=str, default='Plots/episode_start_end')    

    
    args = parser.parse_args()

    combined = []
    with open(args.file, 'rb') as f:
        combined = pickle.load(f)

    avg_actions = 0.0
    avg_final_coverage = 0.0
    avg_improvement_percent = 0.0
    standard_deviation = 0.0
    coverages = []
    samples = 0

    for i in range(0, len(combined)):
        print("Processing iteration %d" % (i))
        iteration = combined[i]
        observation = iteration['obs']
        iteration_info = iteration['info']
        improvement, cov, actions = get_coverage_stats(iteration_info)
        samples += actions
        coverages.append(cov)
        avg_final_coverage += cov
        avg_improvement_percent += improvement
        avg_actions += len(iteration_info)
        get_start_end(observation, i, args.save_folder)
    avg_actions = avg_actions/float(len(combined))
    avg_final_coverage = avg_final_coverage/float(len(combined))
    avg_improvement_percent = avg_improvement_percent/float(len(combined))
    standard_deviation = np.std(coverages)
    standard_error = standard_deviation/samples
    print("Total Actions: %f" %samples)
    print("Actions: %f" % avg_actions)
    print("Average final coverage: %f" % avg_final_coverage)
    print("Average improvement (final coverage/initial coverage): %f" % avg_improvement_percent)
    print("Standard deviation: %f" %standard_deviation)
    print("Standard error: %f" %standard_error)
