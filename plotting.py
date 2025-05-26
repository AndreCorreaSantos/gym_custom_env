
import seaborn as sns
import matplotlib.pyplot as plt 

def plot_coverage(stats):
    sns.set_theme(style="darkgrid", palette="colorblind", font_scale=1.2)
    series = stats["coverage"]
    print(se)

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

        plt.xlabel('Step')
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
