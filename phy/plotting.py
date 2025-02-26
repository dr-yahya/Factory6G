# physical_layer/plotting.py
from sionna.utils.plotting import PlotBER, plot_ber
import tensorflow as tf

class Plotter:
    def __init__(self, title="Bit/Block Error Rate for 6G Smart Factory"):
        self.plotter = PlotBER(title=title)

    def plot_results(self, ebno_dbs, ber, bler, legend="6G Simulation"):
        self.plotter.simulate(
            mc_fun=lambda batch_size, ebno_db: self.simulate_step(batch_size, ebno_db),
            ebno_dbs=ebno_dbs,
            batch_size=100,
            max_mc_iter=1000,
            legend=legend,
            add_ber=True,
            add_bler=True,
            show_fig=True,
            verbose=True
        )

    def simulate_step(self, batch_size, ebno_db):
        raise NotImplementedError("This method should be overridden or handled by Simulator")