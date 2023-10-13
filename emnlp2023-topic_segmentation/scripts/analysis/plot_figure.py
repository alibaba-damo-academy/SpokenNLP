
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


def context_f1():
    # The change curve of f1 as the context length changes
    context_length = [512, 1024, 2048, 4096]
    city_baseline_f1 = [78.88, 81.42, 82.18, 82.36]
    city_sota_f1 = [81.88, 82.82, 83.16, 83.12]

    dis_baseline_f1 = [66.75, 71.16, 72.14, 72.61]
    dis_sota_f1 = [71.93, 73.46, 74.22, 74.08]

    plt.plot(context_length, city_baseline_f1, 's-', color='g', label='Baseline Longformer on en_city', linestyle="dashed")
    plt.plot(context_length, city_sota_f1, 'o-', color='g', label='Our Longformer on en_city')
    plt.plot(context_length, dis_baseline_f1, 's-', color='b', label='Baseline Longformer on en_disease', linestyle="dashed")
    plt.plot(context_length, dis_sota_f1, 'o-', color='b', label='Our Longformer on en_disease')

    for index, f1 in zip(context_length, city_baseline_f1):
        plt.text(index, f1, str(f1))
    for index, f1 in zip(context_length, city_sota_f1):
        plt.text(index, f1, str(f1))
    for index, f1 in zip(context_length, dis_baseline_f1):
        plt.text(index, f1, str(f1))
    for index, f1 in zip(context_length, dis_sota_f1):
        plt.text(index, f1, str(f1))
    plt.xticks(context_length)
    plt.xlabel("context length")
    plt.ylabel("F1")

    plt.legend()
    plt.title("lf-ws-city_disease-context_length-f1")
    plt.show()
    plt.savefig("./lf-ws_city_disease-context_f1.pdf")


if __name__ == "__main__":
    context_f1()
