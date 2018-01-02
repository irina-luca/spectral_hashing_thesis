import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helpers import normalize_data
def show_pc_variance_plot(eig_vals, number_of_pc):
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    x_bar = [i for i in range(1, number_of_pc + 1)]  # PC 1, 2, 3... etc.
    y_bar = var_exp  # their vals

    x_scatter = [i for i in range(1, number_of_pc + 1)],
    y_scatter = cum_var_exp

    fig = plt.figure(figsize=(16,5))
    print("x_bar", x_bar)
    print("y_bar", y_bar)
    for i, tup in enumerate(y_bar):
        print(tup, type(tup), tup.real)
    y_bar = [tup.real for i,tup in enumerate(y_bar)]
    print("y_bar", y_bar)
    print(len(x_bar))
    print(len(y_bar[:number_of_pc]))
    plt.bar(x_bar, y_bar[:number_of_pc], align='center')
    # plt.xticks(x_bar, y_bar[:number_of_pc])
    plt.scatter(x_scatter, y_scatter[:number_of_pc])
    plt.show()


def main():
    delimiter = ' '
    data_filename_location = '../Data/SIFT/Samples/SIFT__ss-5000__1.train'
    data_train = np.genfromtxt(data_filename_location, delimiter=delimiter, dtype=np.float)
    print("DONE reading training set")

    data_train_norm = normalize_data(data_train)
    print("DONE normalizing training set")

    X_std = data_train_norm

    # -- Step 2.1: Do Covariance Matrix -- #
    mean_vec = np.mean(X_std, axis=0)
    # cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
    # print('Covariance matrix \n%s' % cov_mat)
    # print('np covariance matrix: \n%s' % np.cov(X_std.T))

    # -- Step 2.2: Do eigendecomposition on the covariance matrix -- #
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print('Eigenvectors \n%s' % eig_vecs)
    print('\nEigenvalues \n%s' % eig_vals)


    # -- 4. Calculate Variance and Plot it for #n principal components -- #
    show_pc_variance_plot(eig_vals, 65)

    # A = data_train_norm
    #
    # A = np.asmatrix(A.T) * np.asmatrix(A)
    # U, S, V = np.linalg.svd(A)
    # eigvals = S**2 / np.cumsum(S)[-1]
    #
    # sing_vals = np.arange(num_vars) + 1
    # print("sing_vals", sing_vals)
    # print("eigvals", eigvals)
    # plt.plot(sing_vals[:20], eigvals[:20], color='teal', linestyle='-', linewidth=1)
    # plt.title('Scree Plot')
    # plt.xlabel('Principal Component')
    # plt.ylabel('Eigenvalue')
    #
    # # -- Legend stuff -- #
    # leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
    #                  shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
    #                  markerscale=0.4)
    # leg.get_frame().set_alpha(0.4)
    # leg.draggable(state=True)
    #
    #
    # plt.show()


if __name__ == '__main__':
    main()