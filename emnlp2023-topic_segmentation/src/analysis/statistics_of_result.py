
import statistics


def compute_avg_std():
    metrics = ["p", "r", "f1", "pk", "wd", "avg_pred", "avg_true"]
    # longformer on 727k
    res = '''
        81.34 / 73.47 / 77.21 / 13.87 / 14.97 / 4.73 / 5.23,
        81.20 / 73.52 / 77.17 / 13.89 / 15.00 / 4.74 / 5.23,
        81.40 / 73.23 / 77.10 / 13.91 / 15.01 / 4.71 / 5.23
        '''

    avgs = []
    stds = []
    for i, m in enumerate(metrics):
        multiple_time_metric = [float(v_str.split("/")[i]) for v_str in res.split(",")]
        print("%s: " % m, multiple_time_metric)
        print("avg: ", statistics.mean(multiple_time_metric))
        print("mean: %.2f" % (sum(multiple_time_metric) * 1.0 / len(multiple_time_metric)))
        print("std: %.2f" % statistics.stdev(multiple_time_metric))

        avgs.append(sum(multiple_time_metric) * 1.0 / len(multiple_time_metric))
        stds.append(statistics.stdev(multiple_time_metric))

    str_res = ["%.2f" % avg + "(" + "%.2f" % std + ")/ " for avg, std in zip(avgs, stds)]
    print("".join(str_res[:-1]))


def compute_p_value():  
    import numpy as np
    from scipy.stats import ttest_ind

    # F1 of longformer on 727k
    x = [76.28, 76.24, 75.52]
    y = [77.21, 77.17, 77.10]
    res = ttest_ind(x, y).pvalue
    print(res)


if __name__ == "__main__":
    compute_p_value()
    compute_avg_std()
