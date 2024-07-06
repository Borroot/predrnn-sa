import matplotlib.pyplot as plt
import matplotlib as matplt
import jsonpickle
import numpy as np


def load_train_data(variant):
    loss = []
    with open(f'training_results/mnist_predrnn_{variant}_stdout.txt') as f:
        for line in f.readlines():
            if 'training loss' in line:
                loss.append(float(line.split(':')[1]))
    return loss


def load_test_data(variant):
    data = dict()
    for epoch in range(500, 30500, 500):
        try:
            with open(f'test_results/mnist_predrnn_{variant}/{epoch}') as f:
                data[epoch] = jsonpickle.decode(f.read())
        except FileNotFoundError:
            pass
    return data


def plot_loss(data):
    fig, ax = plt.subplots(figsize=(15, 10))

    # plot train loss
    for variant, losses in data['train'].items():
        itrs = list(range(0, len(losses) * 100, 100))
        plt.plot(itrs, np.array(losses) * 64 * 64 * 10, label='train ' + variant)

    # plot test loss
    for variant, loss in data['test'].items():
        losses = [
            data['test'][variant][itr]['mse']['average'] # / (64 * 64 * 10)
            for itr in data['test'][variant]
        ]
        itrs = list(range(500, len(loss) * 500 + 500, 500))
        plt.plot(itrs, losses, label='test ' + variant)

    plt.xlabel('iteration')
    plt.ylabel('loss (MSE)')

    plt.legend()
    plt.savefig('plots/loss.png', dpi=300)
    plt.show()


def plot_framewise_measures(data):
    for measure in ['mse', 'ssim', 'lpips', 'psnr']:
        plt.rc('axes', axisbelow=True)
        plt.grid()

        for variant in data['test']:
            last_itr = list(data['test'][variant].keys())[-1]
            values = data['test'][variant][last_itr][measure]['per frame']

            xs = range(1, len(values) + 1)
            plt.plot(xs, values, label=variant)
            plt.scatter(xs, values)

        plt.xticks(xs)

        plt.ylabel(measure.upper())
        plt.xlabel('time')

        plt.legend()
        plt.tight_layout()

        plt.savefig(f'plots/{measure}.png', dpi=300)
        plt.show()


def plot_framewise_measures_overtime(data, variant):
    for measure, rot in [('mse', 0), ('lpips', 0), ('psnr', 180), ('ssim', 180)]:
        x = np.arange(1, 11)
        y = np.arange(500, 500 + 500 * len(data['test'][variant]), 500)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(y), len(x)))

        for i, itr in enumerate(data['test'][variant]):
            values = data['test'][variant][itr][measure]['per frame']
            Z[i] = values

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(9, 9))

        cmap = matplt.cm.coolwarm if rot == 0 else matplt.cm.coolwarm_r
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)

        ax.xaxis.set_major_locator(matplt.ticker.LinearLocator(10))
        ax.set(xlim=(1, 10), xlabel='time', ylabel='iteration', zlabel=measure.upper())

        ax.view_init(25, 90 + 45 + rot)
        plt.subplots_adjust(left=0.0, right=0.9, top=1.1, bottom=-0.1)

        plt.savefig(f'plots/{measure}_overtime_{variant}.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    variants = ['v2_attention', 'v2_baseline']
    data = {'train': dict(), 'test': dict()}

    for variant in variants:
        data['test'][variant] = load_test_data(variant)
        data['train'][variant] = load_train_data(variant)

    plot_loss(data)
    plot_framewise_measures(data)

    # plot_framewise_measures_overtime(data, 'v2_attention')
    # plot_framewise_measures_overtime(data, 'v2_baseline')