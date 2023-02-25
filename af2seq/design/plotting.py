import numpy as np
import seaborn as sns
import py3Dmol
from matplotlib import pyplot as plt

from af2seq.alphafold.common import protein


def plot_pred(design):
    """
    Plots the predicted pdb structure using one of the models
    Returns:

    """
    p = py3Dmol.view(js="https://3dmol.org/build/3Dmol.js")
    if design.modeltype == "ptm":
        p.addModel(
            protein.to_pdb(
                protein.from_prediction(
                    design.best_result["feat"], design.best_result["result"]
                )
            ),
            "pdb",
        )
    else:
        p.addModel(
            protein.to_pdb(
                protein.from_prediction(
                    design.best_result["feat"],
                    design.best_result["result"],
                    remove_leading_feature_dimension=False,
                )
            ),
            "pdb",
        )
    p.setStyle({"cartoon": {"color": "spectrum"}})
    p.zoomTo()
    return p.show()


def animate(design, output=None, speed=100):
    models = ''

    for i, pdb in enumerate(design.pdbs):
        models += "MODEL " + str(i) + "\n"
        models += ''.join(pdb[12:-11])
        models += "ENDMDL\n"

    view = py3Dmol.view(width=800, height=600)
    view.addModelsAsFrames(models)
    default = {"cartoon": {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 0, 'max': 100}}}
    view.setStyle(default)
    view.addStyle({"stick": {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 0, 'max': 100}}})
    view.zoomTo()
    view.animate({'loop': "forward", 'reps': 0, 'interval': speed}, )

    if output:

        with open(f'{output}.html', 'w') as f:
            f.write(view._make_html())
    else:
        return view.show()


def loss_plot(design):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title("Design Loss of all models")
    x = np.arange(len(design.loss_track["mean"][:]))
    sns.lineplot(x=x, y=design.loss_track["mean"][:])
    ax.fill_between(
        x, design.loss_track["lower"][:], design.loss_track["upper"][:], alpha=0.25
    )
    plt.xlabel("Steps")
    plt.ylabel("Design loss on backbone")
    plt.ylim(0, max(design.loss_track["mean"]))

    fig.show()