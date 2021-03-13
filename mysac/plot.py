import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    with open(sys.argv[1]) as stats_file:
        rewards = [
            float(line)
            for line in stats_file
        ]

    plt.title(sys.argv[1])
    plt.plot(rewards)
    plt.show()