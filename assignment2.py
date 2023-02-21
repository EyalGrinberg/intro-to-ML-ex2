#################################
# Your Name: Eyal Grinberg
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X_sample_vec_uniform = np.sort(np.random.uniform(0, 1, m)) # sampling m values from U[0,1] & sorting them
        Y_sample_vec = np.zeros(m)
        # sampling m Y values according to the given distribution
        for i in range(m):
            if 0.2 < X_sample_vec_uniform[i] < 0.4 or 0.6 < X_sample_vec_uniform[i] < 0.8:
                y_single_sample = np.random.choice([0, 1] , p = [0.9, 0.1])
            else:
                y_single_sample = np.random.choice([0, 1] , p = [0.2, 0.8])
            Y_sample_vec[i] = y_single_sample
        return np.column_stack((X_sample_vec_uniform, Y_sample_vec))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        sample_size_vec = np.arange(m_first, m_last + 1, step) # n = 10, 15, 20, ... , 100
        avg_experiment_empirical_error_vec = np.zeros(len(sample_size_vec)) # vector length 19, averaged empirical error for each experiment
        avg_experiment_true_error_vec = np.zeros(len(sample_size_vec)) # vector length 19, averaged true error for each experiment
        for j in range(len(sample_size_vec)): # j = 0, 1, ..., 18
            for i in range(T): # experiment 100 times
                sample_X_Y = self.sample_from_D(sample_size_vec[j]) 
                # extract xs, ys for the ERM function
                xs = sample_X_Y[:, 0] 
                ys = sample_X_Y[:, 1]
                intervals_lst_ERM, error_cnt_ERM = intervals.find_best_interval(xs, ys, k)
                avg_experiment_empirical_error_vec[j] += error_cnt_ERM  # sum all errors over T = 100 experiments
                avg_experiment_true_error_vec[j] += self.calc_true_error(intervals_lst_ERM) # sum all true errors
            # need to devide in n also because the ERM implementation just counts the errors and not calculating their mean
            avg_experiment_empirical_error_vec[j] /= sample_size_vec[j] * T 
            avg_experiment_true_error_vec[j] /= T # here just need to devide in the size of the experiment
        # plots:
        plt.xlabel("n")
        plt.plot(sample_size_vec, avg_experiment_empirical_error_vec, label = "Averaged Empirical Error")
        plt.plot(sample_size_vec, avg_experiment_true_error_vec, label = "Averaged True Error")
        plt.legend()
        plt.show()
        return np.column_stack((avg_experiment_empirical_error_vec, avg_experiment_true_error_vec)) # the requested return value    

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        best_empirical_error_k = np.zeros(k_last)
        best_true_error_k = np.zeros(k_last)
        X_Y_sample = self.sample_from_D(m) # n = 1500
        xs = X_Y_sample[:, 0]
        ys = X_Y_sample[:, 1]
        k_vec = np.arange(k_first, k_last + 1, step)
        for k in k_vec:
            intervals_lst_ERM, error_cnt_ERM = intervals.find_best_interval(xs, ys, k)
            best_empirical_error_k[k - 1] = error_cnt_ERM / m
            best_true_error_k[k - 1] = self.calc_true_error(intervals_lst_ERM)
        plt.xlabel("k")
        plt.plot(k_vec, best_empirical_error_k, label = "Empirical Error")
        plt.plot(k_vec, best_true_error_k, label = "True Error")
        plt.legend()
        plt.show()
        return np.argmin(best_empirical_error_k) * step + k_first # k starts from 1 and not from 0 so +k_first (+1) is needed

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        best_empirical_error_k = np.zeros(k_last)
        best_true_error_k = np.zeros(k_last)
        penalty_vec = np.zeros(k_last)
        penalty_and_emp_err_vec = np.zeros(k_last)
        X_Y_sample = self.sample_from_D(m) # n = 1500
        xs = X_Y_sample[:, 0]
        ys = X_Y_sample[:, 1]
        k_vec = np.arange(k_first, k_last + 1, step)
        for k in k_vec:
            intervals_lst_ERM, error_cnt_ERM = intervals.find_best_interval(xs, ys, k)
            best_empirical_error_k[k - 1] = error_cnt_ERM / m
            best_true_error_k[k - 1] = self.calc_true_error(intervals_lst_ERM)
            penalty_vec[k - 1] = 2 * np.sqrt( (2*k + np.log(2 / 0.01)) / m ) # from Q3: VCdim(H_k) = 2k , 0.01 = w_k*0.1
            penalty_and_emp_err_vec[k - 1] = penalty_vec[k - 1] + best_empirical_error_k[k - 1]
        plt.xlabel("k")
        plt.plot(k_vec, best_empirical_error_k, label = "Empirical Error")
        plt.plot(k_vec, best_true_error_k, label = "True Error")
        plt.plot(k_vec, penalty_vec, label = "Penalty")
        plt.plot(k_vec, penalty_and_emp_err_vec, label = "Penalty + Empirical Error")
        plt.legend()
        plt.show()
        return np.argmin(penalty_and_emp_err_vec) * step + k_first # k starts from 1 and not from 0 so +k_first (+1) is needed

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        X_Y_sample = self.sample_from_D(m) # n = 1500
        np.random.shuffle(X_Y_sample) # shuffle before splitting data
        S2 = X_Y_sample[: m // 5] # 20% of data goes to holdout set
        S1 = np.array(sorted(X_Y_sample[m // 5 :], key = lambda x: x[0])) # only the training data should be sorted (by x's)
        xs1 = S1[:, 0]
        ys1 = S1[:, 1]
        best_hypo_lst = [] # list of all hypothesis' returned by ERM (list of intervals' lists)
        empirical_errors_lst = np.zeros(10) # list of error count for each retured hypothesis from ERM
        for k in range(1,11): # k = 1,2,...,10
            curr_hypo, curr_error = intervals.find_best_interval(xs1, ys1, k) # run ERM on S1
            best_hypo_lst.append(curr_hypo) # add best hypothesis to the list of best hypothesis'
            empirical_errors_lst[k - 1] = self.calc_validation_error(curr_hypo, S2) # count errors for the hypothesis returned by ERM
        best_k = np.argmin(empirical_errors_lst) + 1 # the k with the minimum error count on S2
        return best_k
            
    #################################
    # Place for additional methods
    
    def get_complement_intervals(self, intervals_lst):
        """
        input: list of intervals
        returns: list of the complementing intervals (under the total interval (0.0, 1.0))
        """
        if len(intervals_lst) == 0: # empty interval case
            return [(0.0, 1.0)]    
        complement_intervals_lst = []
        if intervals_lst[0][0] != 0.0: # if the first interval doesn't start from 0.0 we need to start from 0.0
            complement_intervals_lst.append( (0.0, intervals_lst[0][0]) )
        for i in range(len(intervals_lst) - 1):
            complement_intervals_lst.append( (intervals_lst[i][1], intervals_lst[i + 1][0]) ) # append the wanted interval
        if intervals_lst[-1][1] != 1.0: # check if we need to append one last interval
            complement_intervals_lst.append( (intervals_lst[-1][1], 1.0) )
        return complement_intervals_lst

    def get_total_len_of_intersection(self, intervals_lst1, intervals_lst2):
        """
        input: 2 lists of intervals
        returns: the total length of the intersections between each list's intervals 
        """
        res = 0.0
        for int1 in intervals_lst1:
            for int2 in intervals_lst2:
                if int1[0] >= int2[1] or int2[0] >= int1[1]: # no intersection between int1, int2
                    continue
                else:
                    res += min(int1[1], int2[1]) - max(int1[0], int2[0]) # add to res the length of the intersection
        return res

    def calc_true_error(self, intervals_lst_hypo_1):
        """ calculate true error for a specific hypothesis using total expectation
        input: list of intervals that the hypothesis returs 1 if x is in one of them
        returns: the requested true error
        """
        intervals_lst_hypo_0 = self.get_complement_intervals(intervals_lst_hypo_1) # list of intervals that the hypothesis returns 0 for them
        prob_8_2 = [(0.0, 0.2) , (0.4, 0.6) , (0.8, 1.0)] # if a given x is in these intervals: Pr[y=0] = 0.8 , Pr[y=1] = 0.2
        prob_1_9 = self.get_complement_intervals(prob_8_2) # if a given x is in these intervals: Pr[y=0] = 0.1 , Pr[y=1] = 0.9
        # total expectation:
        ret = 0.2 * self.get_total_len_of_intersection(intervals_lst_hypo_1, prob_8_2) + \
            0.9 * self.get_total_len_of_intersection(intervals_lst_hypo_1, prob_1_9) + \
                0.8 * self.get_total_len_of_intersection(intervals_lst_hypo_0, prob_8_2) +\
                    0.1 * self.get_total_len_of_intersection(intervals_lst_hypo_0, prob_1_9)
        return ret

    def calc_validation_error(self, hypo_ERM, S2):
        """ 
        input: a hypothesis returned by ERM on S1, and the holdout set S2
        returns: empirical error count (not mean)
        """
        error_cnt = 0 
        for data_point in S2: # 300 (x,y) data points
            x_prediction = 0 # if all intervals don't contain x, the prediction will be 0
            for interval in hypo_ERM: 
                if interval[0] <= data_point[0] <= interval[1]: # x is inside the interval ---> h(x) = 1
                    x_prediction = 1
                    break
            error_cnt += x_prediction != data_point[1] # just need to count errors for this question
        return error_cnt

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
