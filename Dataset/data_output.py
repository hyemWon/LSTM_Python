from matplotlib import pyplot

class Output:

    def __init__(self):
        pass

    def save(self, data, file_name):
        data.to_csv(file_name)

    def save_plot(self, fig, file_name):
        fig = pyplot.gcf()
        fig.savefig(file_name, dpi=fig.dpi)