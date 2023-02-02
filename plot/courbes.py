import matplotlib
import matplotlib.pyplot as plt
#longimport seaborn as sn
import sys
sys.path.insert(0, '../')
import numpy as np
from scripts.utils import load_pickle

#data1 = load_pickle('../scripts/results/csv/lstm_tbptt.pk')

def base(name):
    data1 = load_pickle('../scripts/results/csv/'+name+'.pk')
    plt.plot(data1['train_loss'], label=name + ' train_loss')
    #plt.plot(data1['test_acc'], label=name + ' test_acc')
    plt.legend()

def err_curve(name):
    N = 1
    data1 = load_pickle('../scripts/results/csv/'+name+'.pk')
    mean = data1['loss']
    std  = data1['stdloss']

    mean = np.convolve(mean, np.ones(N)/N, mode='valid')
    std = np.convolve(std, np.ones(N)/N, mode='valid')
    x = list(range(std.shape[0]))

    plt.errorbar(x, mean, std, linestyle='None', label=name)





def reconstruction(name, N=10):
    logs = load_pickle('../scripts/results/csv/'+name+'.pk')

    offset = 0
    for level in logs:
        x = np.linspace(offset, offset + len(level['train']), len(level['train']))


        plt.subplot(2, 1, 1)

        curve = np.convolve(level['train'], np.ones(N)/N, mode='valid')
        plt.plot(curve, label=name+' '+str(level['n_classes'])+' classes x '+str(level['n_samples'])+' shots = ' + str(level['n_samples']*level['n_classes']) + ' samples')
        plt.title('meta-train-train (Tiny ImageNet) loss')


def curriculum(name, N=10):
    logs = load_pickle('../scripts/results/csv/'+name+'.pk')

    offset = 0
    for level in logs:
        x = np.linspace(offset, offset + len(level['train']), len(level['train']))


        plt.subplot(3, 1, 1)

        curve = np.convolve(level['train'], np.ones(N)/N, mode='valid')
        plt.plot(curve, label=name+' '+str(level['n_classes'])+' classes x '+str(level['n_samples'])+' shots = ' + str(level['n_samples']*level['n_classes']) + ' samples')
        plt.title('meta-train-train (Tiny ImageNet) loss')
        plt.legend()

        plt.subplot(3, 1, 2)

        curve = np.convolve(level['val'], np.ones(N)/N, mode='valid')
        plt.plot(curve, label=name+' '+str(level['n_classes'])+' classes x '+str(level['n_samples'])+' shots = ' + str(level['n_samples']*level['n_classes']) + ' samples')
        plt.title('meta-train-test (Tiny ImageNet) loss')

        plt.legend()

        plt.subplot(3, 1, 3)
        curve = np.convolve(level['acc'], np.ones(N)/N, mode='valid')
        plt.plot(curve, label=str(level['n_classes'])+' classes x '+str(level['n_samples'])+' shots = ' + str(level['n_samples']*level['n_classes']) + ' samples')
        plt.title('meta_test (CIFAR-100) accuracy at end (%)')
        plt.legend()

        offset += len(level['train'])

def forget(name):
    data1 = load_pickle('../scripts/results/csv/'+name+'.pk')
    print(data1)
    plt.plot(data1, label=name)
    plt.legend()
    plt.xlabel('past sample index')
    plt.ylabel('accuracy on past samples')
    plt.title('forgetting characteristic (avg over 50 runs)')


def cstr_vs_nocstr():
    data1 = load_pickle('../scripts/results/csv/fixed_100_cstr.pk')
    data2 = load_pickle('../scripts/results/csv/fixed_100_cstr2.pk')
    data3 = load_pickle('../scripts/results/csv/fixed_100_nocstr.pk')
    data4 = load_pickle('../scripts/results/csv/fixed_100_nocstr2.pk')
    data5 = load_pickle('../scripts/results/csv/fixed_100_slim.pk')
    plt.plot(data1[:], color='blue')
    plt.plot(data2[:], color='blue')
    plt.plot(data3[:], color='red')
    plt.plot(data4[:], color='red')
    plt.plot(data5[:], color='green')
    plt.legend()

#err_curve('complex_optim_2')
#err_curve('complex_optim_3')
#err_curve('complex_optim_4')
#err_curve('complex_optim_2_1_times')
plt.legend()
"""
base('prop_all_500_2')
base('prop_all_2500')
base('prop_all_1000')
"""
#curriculum('NEM_nobn_large')
curriculum('curriculum_notrain')
#curriculum('curriculum_basic')
#curriculum('curriculum_basic')
#curriculum('curriculum_notrain')
#reconstruction('NEM_mix_unsup_4')
#reconstruction('NEM_mix')
#curriculum('NEM_new_gen')
#curriculum('NEM_norm_4')
#curriculum('NEM_norm_5')
#curriculum('NEM_large')
#curriculum('NEM_large_gen')
"""
forget('test_prop_all_500_2_long')
forget('100_2')
forget('500_2')
forget('2500')
forget('2500_512')
forget('5000_512')
"""
plt.show()
