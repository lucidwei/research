import matplotlib.pyplot as plt


class PerformanceVisualizer:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.portfolio_value = evaluator.portfolio_value
        self.benchmark_value = evaluator.benchmark_value

    def plot_combined_chart(self):
        plt.figure(figsize=(12, 6))

        # Plot Portfolio Value and Benchmark Value
        ax1 = plt.gca()
        ax1.plot(self.portfolio_value, label='Portfolio Value')
        ax1.plot(self.benchmark_value, label='Benchmark Value')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper left')

        # Calculate excess value
        excess_value = self.portfolio_value - self.benchmark_value

        # Plot Excess Value on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(excess_value, label='Excess Value', color='purple', linestyle='--')
        ax2.set_ylabel('Excess Value')
        ax2.legend(loc='upper right')

        plt.title('Portfolio and Benchmark Value with Excess Value')
        plt.show()

    def visualize(self):
        self.plot_combined_chart()
