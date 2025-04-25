import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle, json
from tqdm import tqdm


def var_buffer(num_simu):

    with open(f"./output/simulation{num_simu}/var_buffer.pck", "rb") as f:
        vb = pickle.load(f)
        

    nbk = len(vb.keys())

    for i, (k, v) in enumerate(vb.items()):

        if k != 'target' :
            plt.subplot(int(nbk/2+0.6), 2, i+1)
            plt.hist(v, color='r', edgecolor='k', bins=100)
            plt.ylabel(f"{k} \n {np.mean(v):.2f} ~ {np.std(v):.2f}")
            
    plt.show()


def spec(num_simu):

    with open(f"./output/{num_simu}/hparams.json", 'r') as f:
        hparams = json.load(f)

    l = np.arange(hparams["LAMBDA_MIN"], hparams["LAMBDA_MAX"], hparams["LAMBDA_STEP"])

    fold = f"./output/{num_simu}/spectrum"
    files = os.listdir(fold)
    pbar = tqdm(total=len(files))
    img = np.zeros((len(files), hparams["N"]))
    inte = np.zeros(len(files))

    for i, file in enumerate(files):

        spectrum = np.load(f"{fold}/{file}")
        inte[i] = np.sum(spectrum)
        img[i] = spectrum
        pbar.update(1)

    argsort = np.argsort(inte)
    img = img[argsort]

    plt.plot(inte[argsort], c='r')
    plt.savefig(f"./output/{num_simu}/inte_spectrum.png")
    plt.show()

    plt.plot(np.sum(img, axis=0), c='r')
    plt.show()

    print(f"--- : {files[argsort[0]]} with {inte[argsort[0]]} ...")
    print(f"+++ : {files[argsort[-1]]} with {inte[argsort[-1]]} ...")

    fig = plt.figure(frameon=False)
    fig.set_size_inches(len(files), len(l))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.log(img.T+1), aspect='equal', cmap='jet')
    fig.savefig(f"./output/{num_simu}/all_spectrum.png", dpi=1)




def spec_one(num_simu, hd):

    with open(f"./output/{num_simu}/hparams.json", 'r') as f:
        hparams = json.load(f)

    with open(f"./output/{num_simu}/var_buffer.pck", "rb") as f:
        vb = pickle.load(f)

    l = np.arange(hparams["LAMBDA_MIN"], hparams["LAMBDA_MAX"], hparams["LAMBDA_STEP"])

    fold = f"./output/{num_simu}/spectrum"
    files = os.listdir(fold)
    pbar = tqdm(total=len(files))
    img = np.zeros((len(files), hparams["N"]))
    inte = np.zeros(len(files))

    for i, file in enumerate(files):

        if vb['target'][i] == hd:

            spectrum = np.load(f"{fold}/{file}")
            inte[i] = np.sum(spectrum)
            img[i] = spectrum
            pbar.update(1)

            plt.plot(l, spectrum, c='r', alpha=0.1)

    plt.show()

    argsort = np.argsort(inte)
    img = img[argsort]

    plt.plot(inte[argsort], c='r')
    plt.show()

    plt.plot(np.sum(img, axis=0), c='r')
    plt.show()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(len(files), len(l))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.log(img.T+1), aspect='equal', cmap='jet')
    fig.savefig(f"./output/{num_simu}/all_spectrum_{hd}.png", dpi=1)


def view(num_simu, s):

    with open(f"./output/{num_simu}/hparams.json", 'r') as f:
        hparams = json.load(f)

    with open(f"./output/{num_simu}/var_buffer.pck", "rb") as f:
        vb = pickle.load(f)

    path = f"./output/{num_simu}"

    l = np.arange(hparams["LAMBDA_MIN"], hparams["LAMBDA_MAX"], hparams["LAMBDA_STEP"])
    spec = np.load(f"{path}/spectrum/spectrum_{s}.npy")

    for k, v in vb.items():

        print(f"For {k} : {v[int(s)]}")

    plt.plot(l, spec, color='r')
    plt.title(f"SIMU spectrum_{s}.npy")
    plt.show()

    plt.imshow(np.log(np.load(f"{path}/image/image_{s}.npy")+1), cmap='gray')
    plt.title(f"SIMU img_{s}.npy")
    plt.show()




if __name__ == '__main__':

    a2v = dict()

    for arg in sys.argv:

        if "=" in arg : a, v = arg.split("=")
        else : a, v = arg, None

        a2v[a] = v


    if "var_b" in a2v.keys():

        var_buffer(a2v["ns"])

    if "spec" in a2v.keys():

        spec(a2v["ns"])

    if "s" in a2v.keys():

        view(a2v["ns"], a2v["s"])

    if "hd" in a2v.keys():

        spec_one(a2v["ns"], a2v["hd"])