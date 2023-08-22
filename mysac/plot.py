import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    rewards = []
    steps = []

    with open(sys.argv[1]) as stats_file:
        for line in stats_file:
            reward, step = line.split(';')

            rewards.append(float(reward))
            steps.append(float(step))

    plt.title(sys.argv[1])
    plt.plot(steps)
    plt.show()
