# Monte Carlo Simulation Study with a normal distribution
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def sample_normal(mu, sigma, n):
    return np.random.normal(mu, sigma, n)

def sample_mean(x):
    return np.mean(x)

def sample_variance(x):
    return np.var(x)

def mean_ci(x, alpha):
    n = len(x)
    mean = sample_mean(x)
    std_error = np.std(x, ddof=1) / np.sqrt(n) # Standard error
    z = norm.ppf(1 - alpha / 2)
    lower = mean - z * std_error
    upper = mean + z * std_error
    return lower, upper

def run(n, mu, sigma, alpha):
    x = sample_normal(mu, sigma, n)
    mean = sample_mean(x)
    var = sample_variance(x)
    ci_lower, ci_upper = mean_ci(x, alpha)
    return {
        'mean': mean,
        'variance': var,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def monte_carlo(n, mu, sigma, alpha, n_rep): 
    results = []
    for _ in range(n_rep):
        result = run(n, mu, sigma, alpha)
        results.append(result)
    return results

# Visualization 
def plot_results(results, mu):
    means = [res['mean'] for res in results]
    ci_lowers = [res['ci_lower'] for res in results]
    ci_uppers = [res['ci_upper'] for res in results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(means, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label='True Mean')
    plt.title('Histogram of Sample Means')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    # Plot confidence intervals
    plt.figure(figsize=(10, 6))
    for i in range(len(results)):
        plt.plot([i, i], [ci_lowers[i], ci_uppers[i]], color='gray')
        plt.plot(i, means[i], 'bo')
    plt.axhline(mu, color='red', linestyle='dashed', linewidth=2, label='True Mean')
    plt.title('Confidence Intervals for Sample Means')
    plt.xlabel('Simulation Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    n = 30          # Sample size
    mu = 0          # True mean
    sigma = 1       # True standard deviation
    alpha = 0.05    # Significance level
    n_rep = 1000    # Number of Monte Carlo replications
    
    results = monte_carlo(n, mu, sigma, alpha, n_rep)
    plot_results(results, mu)
    x = sample_normal(mu, sigma, n)
    mean = sample_mean(x)
    var = sample_variance(x)
    ci_lower, ci_upper = mean_ci(x, alpha)
    print(f"Sample Mean: {mean}")
    print(f"Sample Variance: {var}")
    print(f"{(1-alpha)*100}% Confidence Interval for Mean: ({ci_lower}, {ci_upper})")