import numpy as np
from scipy.fftpack import fft
import matplotlib
from matplotlib import pyplot as plt



def plotAndSaveElbo(args, cost, file_path):
    plt.plot(cost[1:])
    plt.title('ELBO vs iteration')
    plt.xlabel('iteration number')
    plt.ylabel('Evidence Lower Bound (ELBO)')

    if (len(cost) > 200):
        max = cost[50] + 0.5
        min = cost[-1] - 0.2
    else:
        max = cost[0] + 0.5
        min = cost[-1] - 0.2
    plt.ylim(max, min)
    plt.savefig(file_path + '/loss-dim_{}-epochs_{}-batchsize_{}-timesteps'.format(args.epochs, args.batch_size, args.T))
    plt.close()

    with open(file_path + '/elbo.txt', 'x') as f:
        for item in cost:
            f.write("%s\n" % item)


def plotObservations(timesteps, real_data, trained_data):
    plt.plot(trained_data[:timesteps], linestyle='--', linewidth=2, label='reconstructed', alpha=0.6)
    plt.plot(real_data[:timesteps], linewidth=2, label='true', alpha=0.6)
    plt.title('Trained and true observations')
    plt.legend()
    plt.show()


def plotAndSaveObservations(timesteps, model_name, real_data, trained_data, file_path, alternateName = None, labelOne = None, labelTwo = None):
    colors = []
    cmap_temp = np.linspace(0.0, 0.8, 2)
    for number in cmap_temp:
        cmap = matplotlib.cm.get_cmap('viridis')
        colors.append(cmap(number))
    dim_xTrue = trained_data.shape[1]
    if dim_xTrue > 12:
        dim_xTrue = 12
        print("Only printing first 12 dimensions")
    fig, ax = plt.subplots(dim_xTrue, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Trained and true observations')
    fig.set_size_inches(12.5, 15.5)

    if labelOne == None:
        labelOne = 'true'
    if labelTwo == None:
        labelTwo = 'reconstructed'
    for ind in range(0, dim_xTrue):
        trainedData = trained_data.t()[ind][:timesteps]
        trueData = real_data.t()[ind][:timesteps]
        ax[ind].plot(trueData, color=colors[1], linewidth=2, label=labelOne, alpha=0.6)
        ax[ind].plot(trainedData,  color=colors[0], linewidth=2, label=labelTwo, alpha=0.6) #linestyle='-',
        #if ind == 0:
        #    ax[ind].set_title('Trained and true observations')

    plt.xlabel('time in timesteps')
    plt.legend()
    if alternateName is not None:
        plt.savefig(file_path + '/' + alternateName)
    else:
        plt.savefig(file_path + '/trained_and_true_observations_{}-timesteps_{}'.format(timesteps, model_name))
    plt.close()

def plotAndSaveObservationsThreeTimeseries(timesteps, model_name, real_data, inferred_data, trained_data, file_path, alternateName = None, labelOne = None, labelTwo = None, labelThree = None):
    colors = []
    cmap_temp = np.linspace(0.0, 0.8, 3)
    for number in cmap_temp:
        cmap = matplotlib.cm.get_cmap('viridis')
        colors.append(cmap(number))
    dim_xTrue = trained_data.shape[1]
    if dim_xTrue > 12:
        dim_xTrue = 12
        print("Only printing first 12 dimensions")
    fig, ax = plt.subplots(dim_xTrue, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Trained, inferred and true observations')
    fig.set_size_inches(12.5, 15.5)

    if labelOne == None:
        labelOne = 'true'
    if labelTwo == None:
        labelTwo = 'inferred'
    if labelThree == None:
        labelThree = 'reconstructed'
    for ind in range(0, dim_xTrue):
        trainedData = trained_data.t()[ind][:timesteps]
        trueData = real_data.t()[ind][:timesteps]
        inferredData = inferred_data.t()[ind][:timesteps]
        ax[ind].plot(trueData, color=colors[2], linewidth=2, label=labelOne, alpha=0.6)
        ax[ind].plot(trainedData,  color=colors[1], linewidth=2, label=labelTwo, alpha=0.6) #linestyle='-',
        ax[ind].plot(inferredData,  color=colors[0], linewidth=2, label=labelThree, alpha=0.6) #linestyle='-',
        #if ind == 0:
        #    ax[ind].set_title('Trained and true observations')

    plt.xlabel('time in timesteps')
    plt.legend()
    if alternateName is not None:
        plt.savefig(file_path + '/' + alternateName)
    else:
        plt.savefig(file_path + '/trained_and_true_observations_{}-timesteps_{}'.format(timesteps, model_name))
    plt.close()


def plotAndSaveObservationsLessDims(timesteps, model_name, real_data, trained_data, file_path, dims, alternateName = None, label1 = None, label2 = None):
    colors = []
    cmap_temp = np.linspace(0.0, 0.8, 2)
    for number in cmap_temp:
        cmap = matplotlib.cm.get_cmap('viridis')
        colors.append(cmap(number))
    dim_xTrue = dims
    fig, ax = plt.subplots(dim_xTrue, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Trained and true observations')
    fig.set_size_inches(8.5, 6.5)
    if labelOne == None:
        labelOne = 'reconstructed'
    if labelTwo == None:
        labelTwo = 'true'
    for ind in range(0, dim_xTrue):
        trainedData = trained_data.t()[ind][:timesteps]
        trueData = real_data.t()[ind][:timesteps]
        ax[ind].plot(trainedData,  color=colors[0], linewidth=2, label=labelOne, alpha=0.6) #linestyle='-',
        ax[ind].plot(trueData, color=colors[1], linewidth=2, label=labelTwo, alpha=0.6)
        #if ind == 0:
        #    ax[ind].set_title('Trained and true observations')

    plt.xlabel('time in timesteps')
    plt.legend()
    if alternateName is not None:
        plt.savefig(file_path + '/' + alternateName)
    else:
        plt.savefig(file_path + '/trained_and_true_observations_{}-timesteps_{}'.format(timesteps, model_name))
    plt.close()

def plotAndSaveCategories(timesteps, model_name, real_data, trained_data, file_path, alternateName=None):
    colors = []

    cmap_temp = np.linspace(0.0, 0.8, 2)

    for number in cmap_temp:
        cmap = matplotlib.cm.get_cmap('viridis')
        colors.append(cmap(number))
    dim_xTrue = 1
    fig, ax = plt.subplots(dim_xTrue, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Trained and true observations')

    fig.set_size_inches(25.5, 15.5)
    trainedData = trained_data[:timesteps]
    trueData = real_data[:timesteps]
    ax.plot(trainedData, color=colors[0], linewidth=2, label='reconstructed', alpha=0.6)  # linestyle='-',
    ax.plot(trueData, color=colors[1], linewidth=2, label='true', alpha=0.6)
    # if ind == 0:
    # ax[ind].set_title('Trained and true observations')

    plt.xlabel('time in timesteps')
    plt.legend()
    if alternateName is not None:
        plt.savefig(file_path + '/' + alternateName)
    else:
        plt.savefig(file_path + '/categories'.format(timesteps, model_name))
    plt.close()

def plotAndSaveTrueLorenz(timesteps, true_data, file_path):
    fig = plt.figure(figsize=(16, 9))
    # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
    bx = fig.add_subplot(1, 1, 1, projection='3d')
    bx.set_xlabel('$x_1$', size=15)
    bx.set_ylabel('$x_2$', size=15)
    bx.set_zlabel('$x_3$', size=15)
    p = bx.plot(true_data[:timesteps, 0].numpy(), true_data[:timesteps, 1].numpy(), true_data[:timesteps, 2].numpy(),
                antialiased=True,
                linewidth=0.5, label='true')
    plt.show()
    plt.savefig(file_path + '/true_Lorenz_{}_timesteps'.format(timesteps))
    plt.close()


def plotAndSaveReconstructedLorenz(timesteps, model_name, trained_data, file_path):
    fig = plt.figure(figsize=(16, 9))
    # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
    cx = fig.add_subplot(1, 1, 1, projection='3d')
    cx.set_xlabel('$x_1$', size=15)
    cx.set_ylabel('$x_2$', size=15)
    cx.set_zlabel('$x_3$', size=15)
    p = cx.plot(trained_data[:timesteps, 0].numpy(), trained_data[:timesteps, 1].numpy(),
                trained_data[:timesteps, 2].numpy(),
                antialiased=True,
                linewidth=0.5, label='reconstructed')

    plt.savefig(file_path + '/reconstructed_Lorenz_{}_timesteps_{}'.format(timesteps, model_name))
    plt.close()


def plotAndSaveBothLorenz(timesteps, model_name, true_data, trained_data, file_path, alternateName = None):
    fig = plt.figure(figsize=(16, 9))
    # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
    dx = fig.add_subplot(1, 1, 1, projection='3d')
    dx.set_xlabel('$x_1$', size=15)
    dx.set_ylabel('$x_2$', size=15)
    dx.set_zlabel('$x_3$', size=15)
    p = dx.plot(true_data[:timesteps, 0].numpy(), true_data[:timesteps, 1].numpy(), true_data[:timesteps, 2].numpy(),
                antialiased=True,
                linewidth=0.5, label='true')
    p = dx.plot(trained_data[:timesteps, 0].numpy(), trained_data[:timesteps, 1].numpy(),
                trained_data[:timesteps, 2].numpy(), color='red', alpha=0.7,
                antialiased=True,
                linewidth=0.5, label='reconstructed')
    if alternateName is not None:
        plt.savefig(file_path + '/' + alternateName)
    else:
        plt.savefig(file_path + '/true_and_reconstructed_Lorenz_{}_timesteps_{}'.format(timesteps, model_name))
    #plt.savefig(model_name)
    plt.close()


def plotBothLorenz(timesteps, true_data, trained_data):
    fig = plt.figure(figsize=(16, 9))
    # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
    dx = fig.add_subplot(1, 1, 1, projection='3d')
    dx.set_xlabel('$x_1$', size=15)
    dx.set_ylabel('$x_2$', size=15)
    dx.set_zlabel('$x_3$', size=15)
    p = dx.plot(true_data[:timesteps, 0].numpy(), true_data[:timesteps, 1].numpy(), true_data[:timesteps, 2].numpy(),
                antialiased=True,
                linewidth=0.5, label='true')
    p = dx.plot(trained_data[:timesteps, 0].numpy(), trained_data[:timesteps, 1].numpy(),
                trained_data[:timesteps, 2].numpy(), color='red', alpha=0.7,
                antialiased=True,
                linewidth=0.5, label='reconstructed')
    plt.show()


def plotFourierObservations(xRealData, xTrainedData, file_path):
    xRealData = xRealData.numpy()
    xTrainedData = xTrainedData.numpy()
    N = len(xRealData)
    # TODO: In the case of real data analyis, this should be matched with the sampling rate of the experiment,
    #  to actually capture the true frequencies
    T = 1.0 / 250.0
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    dim_xTrue = len(xRealData.T)
    if dim_xTrue > 8:
        dim_xTrue = 8
    fig, ax = plt.subplots(dim_xTrue, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Trained and true observations')
    fig.set_size_inches(14.5, 17.5)
    for ind in range(0, dim_xTrue):
        data = xRealData.T[ind]
        data = fft(data.T)
        trainedData = xTrainedData.T[ind]
        trainedData = fft(trainedData.T)

        # take only positive values and disregard negative ones
        fourierDataTrue = 1 / N * np.abs(data[1:int(N / 2)])
        fourierDataTrained = 1 / N * np.abs(trainedData[1:int(N / 2)])
        ax[ind].plot(xf[:-1], fourierDataTrue, linewidth=2, alpha=0.6, label='true')
        ax[ind].plot(xf[:-1], fourierDataTrained, linewidth=2, alpha=0.6, label='trained')
        ax[ind].grid()
        ax[ind].set_xlim([0,90])
    plt.legend()
    plt.savefig(file_path + '/fourierTestRealDataMulti')


