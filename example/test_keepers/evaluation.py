from scipy import stats
import argparse
from array import array
# This file handles the evalation of a learning algorithm against
# a specified benchmarks and report the statistical significance of the 
# difference found between the two algorithms.

def main():

    stats_path = '/home/student/Desktop/HFO-master_ruben/example/test_keepers/stats.bin'
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', default=stats_path, type=str)
    parser.add_argument('--file1_size', default=0, type=int)
    parser.add_argument('--file2', default=stats_path, type=str)
    parser.add_argument('--file2_size', default=0, type=int)
    args = parser.parse_args()
 
    arr1, arr2 = array('b'), array('b')
    fp1, fp2 = open(args.file1, 'rb'), open(args.file2, 'rb')
    arr1.fromfile(fp1, args.file1_size), arr2.fromfile(fp2, args.file2_size)
    fp1.close()
    fp2.close()
 
    result = stats.ttest_ind(arr1, arr2)
    print "p-value : ", result[1]


if __name__ == '__main__':
    main()
