
import seaborn as sns
import matplotlib.pyplot as plt 

def plot_metrics(train_stats):
    sns.set_theme(style="darkgrid", palette="colorblind", font_scale=1.2)
    
    # Calculate metrics for each version
    avg_coverage = {}
    found_percentage = {}
    
    for rfunc, df in train_stats.items():
        avg_coverage[rfunc] = df['coverage'].mean()
        found_percentage[rfunc] = (df['found'].sum() / len(df)) * 100
    
    versions = list(avg_coverage.keys())
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Coverage plot
    coverage_values = list(avg_coverage.values())
    bars1 = ax1.bar(versions, coverage_values, alpha=0.8, edgecolor='black', linewidth=1)
    
    for bar, value in zip(bars1, coverage_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Version')
    ax1.set_ylabel('Average Coverage (%)')
    ax1.set_title('Average Coverage by Version')
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Found plot
    found_values = list(found_percentage.values())
    bars2 = ax2.bar(versions, found_values, alpha=0.8, edgecolor='black', linewidth=1, color='orange')
    
    for bar, value in zip(bars2, found_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Version')
    ax2.set_ylabel('Found Success Rate (%)')
    ax2.set_title('Success Rate (Found=True) by Version')
    ax2.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_stats(train_stats, window_size=3):
    sns.set_theme(style="darkgrid", palette="colorblind", font_scale=1.2)

    for rfunc, df in train_stats.items():
        agents = df[['agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4']]

        smoothed = agents.rolling(window=window_size, min_periods=1).mean()

        df_long = smoothed.reset_index().melt(
            id_vars='index', var_name='Agent', value_name='Value'
        )

        avg_series = smoothed.mean(axis=1)

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_long, x='index', y='Value', hue='Agent', linewidth=1.5,alpha=0.4)

        plt.plot(
            avg_series.index,
            avg_series.values,
            label='Average',
            color='black',
            linestyle='--',
            linewidth=2
        )

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Rewards/Episode - {rfunc}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()



def plot_rewards(train_rewards, window_size=5):
    sns.set_theme(style="darkgrid", palette="colorblind", font_scale=1.2)

    for name, agents in train_rewards.items():
        smoothed = agents.rolling(window=window_size, min_periods=1).mean()

        df_long = smoothed.reset_index().melt(
            id_vars='index', var_name='Agent', value_name='Value'
        )

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_long, x='index', y='Value', hue='Agent', linewidth=1.5)

        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title(f'Rewards/Step - {name}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
