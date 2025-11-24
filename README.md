# PPO-for-inventory-management
 #Overview
This project demonstrates how deep reinforcement learning can solve real-world inventory management challenges by:

Learning from historical data: Uses 5+ years of actual Walmart sales data (1,913 days)
Adapting to market dynamics: Handles seasonality, trends, holidays, and demand shocks
Optimizing multiple objectives: Balances holding costs, stockout penalties, and ordering costs
Managing lead times: Accounts for realistic 7-day order fulfillment delays
Achieving high service levels: Maintains 97.8% demand fulfillment with optimized inventory

Key Innovation
Unlike traditional inventory methods (Economic Order Quantity, (s,S) policies), our RL agent:

Learns patterns automatically without explicit forecasting models
Adapts dynamically to non-stationary demand
Handles high-dimensional state spaces (10 features including price, events, SNAP benefits)
Scales to large action spaces (533 discrete order quantities)


 Features
Core Capabilities

 Real Walmart M5 Dataset: 30,490 products × 1,913 days × 10 stores
 Automatic Product Selection: Data-driven filtering by sales volume
 Custom Gym Environment: OpenAI Gymnasium-compatible inventory simulation
 PPO Algorithm: State-of-the-art policy gradient RL
 Rich Feature Engineering: Calendar events, SNAP benefits, pricing, seasonality
 Lead Time Simulation: Realistic 7-day order delays
 Comprehensive Metrics: Service level, costs breakdown, inventory turnover
 Professional Visualizations: 4-panel performance analysis plots
