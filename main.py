

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


class WalmartDataLoader:


    def __init__(self, data_path='./m5-forecasting-accuracy/'):
        self.data_path = data_path
        self.sales_df = None
        self.calendar_df = None
        self.prices_df = None
        self.processed_data = None

    def load_data(self):

        print("Loading Walmart M5 Dataset...")

        try:

            self.sales_df = pd.read_csv(f'{self.data_path}sales_train_evaluation.csv')
            self.calendar_df = pd.read_csv(f'{self.data_path}calendar.csv')
            self.prices_df = pd.read_csv(f'{self.data_path}sell_prices.csv')

            print(f"✓ Sales data shape: {self.sales_df.shape}")
            print(f"✓ Calendar data shape: {self.calendar_df.shape}")
            print(f"✓ Prices data shape: {self.prices_df.shape}")

            return True

        except FileNotFoundError:
            print("\n⚠ Dataset not found. Please download from Kaggle:")
            print("https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
            print("\nFiles needed:")
            print("  - sales_train_evaluation.csv")
            print("  - calendar.csv")
            print("  - sell_prices.csv")
            return False

    def select_high_volume_products(self, n_products=50, category='FOODS'):

        print(f"\nSelecting top {n_products} products from {category} category...")


        category_products = self.sales_df[
            self.sales_df['cat_id'] == category
            ].copy()


        sales_cols = [col for col in self.sales_df.columns if col.startswith('d_')]
        category_products['total_sales'] = category_products[sales_cols].sum(axis=1)


        top_products = category_products.nlargest(n_products, 'total_sales')

        print(f"✓ Selected {len(top_products)} products")
        print(f"  Average daily sales: {top_products['total_sales'].mean() / len(sales_cols):.2f}")
        print(f"  Total volume: {top_products['total_sales'].sum():,.0f} units")

        return top_products

    def prepare_time_series(self, product_id, store_id='CA_1'):



        product_data = self.sales_df[
            (self.sales_df['id'] == product_id)
        ].copy()

        if product_data.empty:
            return None


        sales_cols = [col for col in product_data.columns if col.startswith('d_')]
        sales_values = product_data[sales_cols].values.flatten()


        calendar_info = self.calendar_df[['d', 'date', 'wm_yr_wk', 'event_name_1',
                                          'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI']].copy()
        calendar_info['d_num'] = calendar_info['d'].str.extract('(\d+)').astype(int)


        item_id = product_data['item_id'].values[0]
        store_id_actual = product_data['store_id'].values[0]

        price_info = self.prices_df[
            (self.prices_df['item_id'] == item_id) &
            (self.prices_df['store_id'] == store_id_actual)
            ].copy()


        ts_data = pd.DataFrame({
            'd_num': range(1, len(sales_values) + 1),
            'sales': sales_values
        })

        ts_data = ts_data.merge(calendar_info, on='d_num', how='left')
        ts_data = ts_data.merge(price_info, on='wm_yr_wk', how='left')


        ts_data['sell_price'] = ts_data['sell_price'].fillna(method='ffill').fillna(
            ts_data['sell_price'].mean()
        )


        ts_data['date'] = pd.to_datetime(ts_data['date'])
        ts_data['day_of_week'] = ts_data['date'].dt.dayofweek
        ts_data['day_of_month'] = ts_data['date'].dt.day
        ts_data['month'] = ts_data['date'].dt.month
        ts_data['is_weekend'] = (ts_data['day_of_week'] >= 5).astype(int)
        ts_data['has_event'] = (~ts_data['event_name_1'].isna()).astype(int)

        return ts_data


class WalmartInventoryEnvironment(gym.Env):


    def __init__(self, time_series_data, max_inventory=1000,
                 holding_cost_ratio=0.1, stockout_cost_ratio=5.0,
                 order_cost_ratio=0.05, lead_time=7, episode_length=365):
        super().__init__()

        self.ts_data = time_series_data.reset_index(drop=True)
        self.max_inventory = max_inventory
        self.lead_time = lead_time
        self.episode_length = min(episode_length, len(self.ts_data) - lead_time - 1)

        # Dynamic costs based on product price
        self.avg_price = self.ts_data['sell_price'].mean()
        self.holding_cost = self.avg_price * holding_cost_ratio
        self.stockout_cost = self.avg_price * stockout_cost_ratio
        self.order_cost = self.avg_price * order_cost_ratio


        self.action_space = spaces.Discrete(max_inventory + 1)


        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([
                max_inventory * 2,
                max_inventory * lead_time,
                episode_length,
                1000,
                50,
                6,
                1,
                1,
                1,
                30
            ]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.ts_data) - self.episode_length - self.lead_time - 1
        self.start_idx = np.random.randint(0, max(1, max_start))
        self.current_idx = self.start_idx

        self.current_inventory = self.max_inventory // 2
        self.day = 0
        self.pending_orders = []
        self.demand_history = []
        self.total_reward = 0

        self.metrics = {
            'holding_costs': [],
            'stockout_costs': [],
            'order_costs': [],
            'service_level': [],
            'inventory_levels': [],
            'demands': [],
            'orders': []
        }

        return self._get_observation(), {}

    def _get_observation(self):

        idx = min(self.current_idx + self.day, len(self.ts_data) - 1)
        row = self.ts_data.iloc[idx]

        pending_total = sum(self.pending_orders)
        recent_demand = np.mean(self.demand_history[-7:]) if len(self.demand_history) > 0 else row['sales']


        snap_ca = row.get('snap_CA', 0)
        days_to_month_end = 30 - row['day_of_month']

        return np.array([
            self.current_inventory,
            pending_total,
            self.day,
            recent_demand,
            row['sell_price'],
            row['day_of_week'],
            row['is_weekend'],
            row['has_event'],
            snap_ca,
            days_to_month_end
        ], dtype=np.float32)

    def step(self, action: int):

        order_quantity = action
        order_cost = self.order_cost * order_quantity if order_quantity > 0 else 0


        self.pending_orders.append(order_quantity)


        if len(self.pending_orders) > self.lead_time:
            incoming = self.pending_orders.pop(0)
            self.current_inventory = min(self.current_inventory + incoming, self.max_inventory * 2)


        idx = min(self.current_idx + self.day, len(self.ts_data) - 1)
        demand = int(self.ts_data.iloc[idx]['sales'])

        self.demand_history.append(demand)
        self.metrics['demands'].append(demand)
        self.metrics['orders'].append(order_quantity)


        if self.current_inventory >= demand:
            self.current_inventory -= demand
            stockout_cost = 0
            service_met = 1
        else:
            stockout = demand - self.current_inventory
            self.current_inventory = 0
            stockout_cost = self.stockout_cost * stockout
            service_met = 0


        holding_cost = self.holding_cost * self.current_inventory


        reward = -(holding_cost + stockout_cost + order_cost)


        self.metrics['holding_costs'].append(holding_cost)
        self.metrics['stockout_costs'].append(stockout_cost)
        self.metrics['order_costs'].append(order_cost)
        self.metrics['service_level'].append(service_met)
        self.metrics['inventory_levels'].append(self.current_inventory)

        self.day += 1
        self.total_reward += reward


        done = (self.day >= self.episode_length) or (self.current_idx + self.day >= len(self.ts_data) - 1)
        truncated = False

        return self._get_observation(), reward, done, truncated, {}


class WalmartTrainingCallback(BaseCallback):


    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_service_levels = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_service = np.mean(self.episode_service_levels[-10:]) if self.episode_service_levels else 0
                if self.verbose:
                    print(f"Step: {self.n_calls:,} | Reward: {mean_reward:.2f} | Service Level: {mean_service:.1%}")
        return True


def train_walmart_rl_agent(env, algorithm='PPO', total_timesteps=500000, n_episodes=1500):


    print(f"\n{'=' * 70}")
    print(f"Training {algorithm} on Real Walmart M5 Data")
    print(f"Episodes: {n_episodes} | Timesteps: {total_timesteps:,}")
    print(f"{'=' * 70}\n")

    vec_env = DummyVecEnv([lambda: env])

    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            vec_env,
            verbose=0,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
    elif algorithm == 'A2C':
        model = A2C(
            'MlpPolicy',
            vec_env,
            verbose=0,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99
        )

    callback = WalmartTrainingCallback(check_freq=5000, verbose=1)

    print("🚀 Training started...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print("\n✓ Training completed!\n")

    return model


def evaluate_walmart_agent(env, model, n_eval_episodes=10):


    print(f"\n{'=' * 70}")
    print(f"Evaluating Agent on Walmart Data ({n_eval_episodes} episodes)")
    print(f"{'=' * 70}\n")

    episode_rewards = []
    episode_metrics = []

    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        metrics = {
            'episode': episode + 1,
            'total_reward': episode_reward,
            'total_cost': -episode_reward,
            'holding_cost': sum(env.metrics['holding_costs']),
            'stockout_cost': sum(env.metrics['stockout_costs']),
            'order_cost': sum(env.metrics['order_costs']),
            'service_level': np.mean(env.metrics['service_level']) * 100,
            'avg_inventory': np.mean(env.metrics['inventory_levels']),
            'total_demand': sum(env.metrics['demands']),
            'total_ordered': sum(env.metrics['orders'])
        }

        episode_rewards.append(episode_reward)
        episode_metrics.append(metrics)

        print(f"Episode {episode + 1:2d} | "
              f"Reward: {episode_reward:,.0f} | "
              f"Service: {metrics['service_level']:5.1f}% | "
              f"Avg Inv: {metrics['avg_inventory']:6.1f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Mean Episode Reward:     {np.mean(episode_rewards):,.2f} ± {np.std(episode_rewards):,.2f}")
    print(f"Mean Service Level:      {np.mean([m['service_level'] for m in episode_metrics]):.2f}%")
    print(f"Mean Total Cost:         ${np.mean([m['total_cost'] for m in episode_metrics]):,.2f}")
    print(f"  - Holding Cost:        ${np.mean([m['holding_cost'] for m in episode_metrics]):,.2f}")
    print(f"  - Stockout Cost:       ${np.mean([m['stockout_cost'] for m in episode_metrics]):,.2f}")
    print(f"  - Order Cost:          ${np.mean([m['order_cost'] for m in episode_metrics]):,.2f}")
    print(f"Mean Avg Inventory:      {np.mean([m['avg_inventory'] for m in episode_metrics]):,.1f} units")
    print(f"{'=' * 70}\n")

    return episode_rewards, episode_metrics


def visualize_walmart_results(env, model, product_name="Walmart Product",
                              save_path='walmart_rl_analysis.png'):


    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)


    days = list(range(len(env.metrics['inventory_levels'])))
    inventory = env.metrics['inventory_levels']
    demands = env.metrics['demands']
    orders = env.metrics['orders']


    start_idx = env.start_idx
    dates = env.ts_data.iloc[start_idx:start_idx + len(days)]['date'].values

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))


    axes[0].plot(days, inventory, label='Inventory Level', linewidth=2, color='#1f77b4')
    axes[0].axhline(y=env.max_inventory, color='r', linestyle='--',
                    label=f'Max Capacity ({env.max_inventory})', alpha=0.7)
    axes[0].fill_between(days, 0, inventory, alpha=0.3, color='#1f77b4')
    axes[0].set_ylabel('Inventory (units)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'RL Agent Performance on Walmart Data - {product_name}',
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)


    axes[1].bar(days, demands, label='Daily Demand (Actual)', color='#2ca02c', alpha=0.7)
    axes[1].plot(days, pd.Series(demands).rolling(7).mean(),
                 label='7-Day MA', color='darkgreen', linewidth=2)
    axes[1].set_ylabel('Demand (units)', fontsize=12, fontweight='bold')
    axes[1].set_title('Real Walmart Demand Pattern', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


    axes[2].bar(days, orders, label='Order Quantity', color='#ff7f0e', alpha=0.7)
    axes[2].set_ylabel('Order (units)', fontsize=12, fontweight='bold')
    axes[2].set_title('RL Agent Order Decisions', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)


    cumulative_holding = np.cumsum(env.metrics['holding_costs'])
    cumulative_stockout = np.cumsum(env.metrics['stockout_costs'])
    cumulative_order = np.cumsum(env.metrics['order_costs'])

    axes[3].plot(days, cumulative_holding, label='Holding Cost', linewidth=2)
    axes[3].plot(days, cumulative_stockout, label='Stockout Cost', linewidth=2)
    axes[3].plot(days, cumulative_order, label='Order Cost', linewidth=2)
    axes[3].set_xlabel('Day', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Cumulative Cost ($)', fontsize=12, fontweight='bold')
    axes[3].set_title('Cost Accumulation Over Time', fontsize=12)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()


def analyze_product_characteristics(ts_data):


    print("\n" + "=" * 70)
    print("📈 PRODUCT ANALYSIS")
    print("=" * 70)

    print(f"\nData Period: {ts_data['date'].min()} to {ts_data['date'].max()}")
    print(f"Total Days: {len(ts_data)}")

    print(f"\nDemand Statistics:")
    print(f"  Mean Daily Sales:     {ts_data['sales'].mean():.2f} units")
    print(f"  Std Dev:              {ts_data['sales'].std():.2f} units")
    print(f"  Max Daily Sales:      {ts_data['sales'].max():.0f} units")
    print(f"  Min Daily Sales:      {ts_data['sales'].min():.0f} units")
    print(f"  Total Sales:          {ts_data['sales'].sum():,.0f} units")

    print(f"\nPrice Statistics:")
    print(f"  Mean Price:           ${ts_data['sell_price'].mean():.2f}")
    print(f"  Price Range:          ${ts_data['sell_price'].min():.2f} - ${ts_data['sell_price'].max():.2f}")

    print(f"\nSeasonality:")
    print(f"  Weekday Avg:          {ts_data[ts_data['day_of_week'] < 5]['sales'].mean():.2f} units")
    print(f"  Weekend Avg:          {ts_data[ts_data['day_of_week'] >= 5]['sales'].mean():.2f} units")
    print(f"  Event Days Impact:    {ts_data[ts_data['has_event'] == 1]['sales'].mean():.2f} vs "
          f"{ts_data[ts_data['has_event'] == 0]['sales'].mean():.2f} units")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🏪 WALMART M5 REINFORCEMENT LEARNING INVENTORY MANAGEMENT")
    print("=" * 70)


    DATA_PATH = './m5-forecasting-accuracy/'
    N_EPISODES = 1500
    EPISODE_LENGTH = 365
    TOTAL_TIMESTEPS = 500000


    print("\n[1/6] Loading Walmart M5 Dataset...")
    loader = WalmartDataLoader(data_path=DATA_PATH)

    if not loader.load_data():
        print("\n" + "=" * 70)
        print("⚠ Please download the M5 dataset from Kaggle first:")
        print("   kaggle competitions download -c m5-forecasting-accuracy")
        print("=" * 70)
        exit()


    print("\n[2/6] Selecting Product...")
    top_products = loader.select_high_volume_products(n_products=10, category='FOODS')


    selected_product = top_products.iloc[0]
    product_id = selected_product['id']
    product_name = selected_product['item_id']

    print(f"\n✓ Selected Product: {product_name}")
    print(f"  Product ID: {product_id}")
    print(f"  Category: {selected_product['cat_id']}")
    print(f"  Department: {selected_product['dept_id']}")
    print(f"  Store: {selected_product['store_id']}")


    print("\n[3/6] Preparing Time Series Data...")
    ts_data = loader.prepare_time_series(product_id)

    if ts_data is None:
        print("❌ Failed to load time series data")
        exit()

    analyze_product_characteristics(ts_data)


    print("[4/6] Creating Walmart Inventory Environment...")
    env = WalmartInventoryEnvironment(
        time_series_data=ts_data,
        max_inventory=int(ts_data['sales'].quantile(0.95) * 30),  # ~30 days of peak demand
        holding_cost_ratio=0.1,
        stockout_cost_ratio=5.0,
        order_cost_ratio=0.05,
        lead_time=7,  # 1 week lead time
        episode_length=EPISODE_LENGTH
    )

    print(f"✓ Environment created")
    print(f"  Max Inventory: {env.max_inventory} units")
    print(f"  Lead Time: {env.lead_time} days")
    print(f"  Episode Length: {env.episode_length} days")


    print("\n[5/6] Training Configuration Ready...")
    print(f"  Algorithm: PPO")
    print(f"  Episodes: {N_EPISODES}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    print("\n💡 To train, uncomment the training lines below:")

   

    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print("\nDataset Statistics:")
    print(f"  Total Time Series: {len(ts_data)} days")
    print(f"  Total Volume: {ts_data['sales'].sum():,.0f} units")
    print(f"  Revenue Potential: ${(ts_data['sales'] * ts_data['sell_price']).sum():,.2f}")
    print("\nNext Steps:")
    print("  1. Uncomment training code to train the agent")
    print("  2. Experiment with different products/categories")
    print("  3. Adjust hyperparameters for better performance")
    print("  4. Try different RL algorithms (PPO, A2C)")
    print("=" * 70 + "\n")
