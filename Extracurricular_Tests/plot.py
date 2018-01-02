import matplotlib.pyplot as plt

def plot_2d(X, Y, x_max, y_max):
    plt.plot(X, Y, 'ro')
    plt.axis([0, x_max, 0, y_max])
    plt.show()

def main_plot():
    X = [0, 2, 1, 1, 2, 6, 7, 8, 9, 10]
    Y = [1, 1, 2, 3, 5, 8, 10, 9, 7, 10]
    plot_2d(X, Y, 12, 12)


main_plot()

