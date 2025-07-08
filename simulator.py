import numpy as np
import matplotlib.pyplot as plt
from sionna.phy.utils import ebnodb2no, compute_ber
from config import Config6G

class Simulator6G:
    """6G System Simulator"""
    
    def __init__(self, transmitter, receiver, channel, resource_grid):
        self.config = Config6G()
        self.transmitter = transmitter
        self.receiver = receiver
        self.channel = channel
        self.resource_grid = resource_grid
    
    def simulate_ber(self, ebno_db, batch_size=None):
        """Simulate BER for given Eb/No"""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        # Calculate noise variance
        no = ebnodb2no(
            ebno_db,
            self.config.BITS_PER_SYMBOL,
            self.config.CODE_RATE,
            self.resource_grid
        )
        
        # Set channel topology
        self.channel.set_topology(batch_size)
        
        # Transmit
        b, x_rg = self.transmitter.transmit(batch_size)
        
        # Channel
        y = self.channel.apply(x_rg, no)
        
        # Receive
        b_hat = self.receiver.receive(y, no)
        
        # Calculate BER
        ber = compute_ber(b, b_hat).numpy()
        
        return ber
    
    def run_simulation(self, ebno_range, batch_size=None):
        """Run simulation over Eb/No range"""
        ber_results = []
        
        for ebno_db in ebno_range:
            ber = self.simulate_ber(ebno_db, batch_size)
            ber_results.append(ber)
            print(f"Eb/No = {ebno_db:3d} dB, BER = {ber:.6f}")
        
        return ebno_range, ber_results
    
    def plot_results(self, ebno_values, ber_results):
        """Plot BER vs Eb/No"""
        plt.figure(figsize=(10, 8))
        plt.semilogy(ebno_values, ber_results, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Eb/No (dB)', fontsize=12)
        plt.ylabel('Bit Error Rate (BER)', fontsize=12)
        plt.title('6G Multiuser MIMO Performance', fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.xlim([min(ebno_values), max(ebno_values)])
        plt.ylim([1e-5, 1])
        plt.tight_layout()
        plt.show()