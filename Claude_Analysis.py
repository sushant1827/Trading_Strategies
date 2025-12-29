import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import seaborn as sns
from datetime import datetime
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StrategyParameters:
    """Parameters for the investment strategy"""
    initial_drop_trigger: float
    increment_percentage: float
    base_investment: float
    monthly_budget: float
    transaction_cost: float
    min_investment: float
    uptrend_days: int

    def to_dict(self):
        return asdict(self)

class InvestmentStrategy:
    def __init__(self, parameters: StrategyParameters):
        self.params = parameters
        self.reset_state()

    def reset_state(self):
        self.consecutive_drops = 0
        self.current_investment = self.params.base_investment
        self.accumulated_funds = 0
        self.monthly_invested = 0
        self.current_month = None

    def calculate_investment_amount(self, row: pd.Series, prev_close: float) -> float:
        try:
            current_date = pd.to_datetime(row['Date'])
            if self.current_month != current_date.month:
                self.current_month = current_date.month
                self.monthly_invested = 0

            if self.monthly_invested >= self.params.monthly_budget:
                return 0

            if prev_close <= 0:  # Protect against division by zero
                return 0

            price_change = (row['Close'] - prev_close) / prev_close

            if price_change < -self.params.initial_drop_trigger:
                self.consecutive_drops += 1
                investment = self.params.base_investment * (
                    1 + self.params.increment_percentage) ** (self.consecutive_drops - 1)
                investment = min(investment, 
                               self.params.monthly_budget - self.monthly_invested)
                
                if investment < self.params.min_investment:
                    return 0

                self.monthly_invested += investment
                return investment

            self.consecutive_drops = 0
            return 0

        except Exception as e:
            logger.error(f"Error in calculate_investment_amount: {str(e)}")
            return 0

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, strategy: InvestmentStrategy):
        self.data = data.copy()
        self.strategy = strategy
        self.results = None
        
        # Ensure required columns exist
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def run_backtest(self) -> pd.DataFrame:
        try:
            results = []
            self.strategy.reset_state()

            for i in range(1, len(self.data)):
                prev_close = self.data.iloc[i-1]['Close']
                current_row = self.data.iloc[i]

                investment = self.strategy.calculate_investment_amount(current_row, prev_close)

                if investment > 0 and current_row['Close'] > 0:  # Protect against division by zero
                    investment_after_costs = investment * (1 - self.strategy.params.transaction_cost)
                    shares_bought = investment_after_costs / current_row['Close']
                else:
                    shares_bought = 0

                results.append({
                    'Date': current_row['Date'],
                    'Close': current_row['Close'],
                    'Investment': investment,
                    'Shares': shares_bought
                })

            self.results = pd.DataFrame(results)
            return self.results

        except Exception as e:
            logger.error(f"Error in run_backtest: {str(e)}")
            raise

    def calculate_metrics(self) -> Dict:
        try:
            if self.results is None:
                raise ValueError("Must run backtest before calculating metrics")

            total_investment = self.results['Investment'].sum()
            total_shares = self.results['Shares'].sum()
            final_value = total_shares * self.results.iloc[-1]['Close']

            # Handle edge cases
            if total_investment <= 0:
                return {
                    'Total Return (%)': 0,
                    'CAGR (%)': 0,
                    'Sharpe Ratio': 0,
                    'Max Drawdown (%)': 0,
                    'Total Transactions': 0,
                    'Total Investment': 0,
                    'Final Value': 0
                }

            # Calculate date difference in years
            start_date = pd.to_datetime(self.results['Date'].iloc[0])
            end_date = pd.to_datetime(self.results['Date'].iloc[-1])
            years = max((end_date - start_date).days / 365, 0.01)  # Minimum 0.01 years to avoid division by zero

            # Calculate returns
            total_return = ((final_value / total_investment) - 1) * 100 if total_investment > 0 else 0
            cagr = ((final_value / total_investment) ** (1/years) - 1) * 100 if total_investment > 0 and years > 0 else 0

            # Calculate Sharpe Ratio
            daily_returns = self.results['Close'].pct_change().fillna(0)
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0

            # Calculate Maximum Drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() * 100

            return {
                'Total Return (%)': round(total_return, 2),
                'CAGR (%)': round(cagr, 2),
                'Sharpe Ratio': round(sharpe, 2),
                'Max Drawdown (%)': round(max_drawdown, 2),
                'Total Transactions': int((self.results['Investment'] > 0).sum()),
                'Total Investment': round(total_investment, 2),
                'Final Value': round(final_value, 2)
            }

        except Exception as e:
            logger.error(f"Error in calculate_metrics: {str(e)}")
            return None

def optimize_parameters(data: pd.DataFrame, param_ranges: Dict) -> Tuple[Optional[StrategyParameters], Optional[Dict], str]:
    """Optimize strategy parameters using grid search"""
    best_params = None
    best_metrics = None
    best_sharpe = float('-inf')
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        all_results = []
        
        total_iterations = (len(param_ranges['drop_trigger']) * 
                          len(param_ranges['increment']) * 
                          len(param_ranges['base_investment']))
        current_iteration = 0

        for drop_trigger in param_ranges['drop_trigger']:
            for increment in param_ranges['increment']:
                for base_investment in param_ranges['base_investment']:
                    current_iteration += 1
                    logger.info(f"Processing iteration {current_iteration}/{total_iterations}")
                    
                    params = StrategyParameters(
                        initial_drop_trigger=drop_trigger,
                        increment_percentage=increment,
                        base_investment=base_investment,
                        monthly_budget=5000,
                        transaction_cost=0.02,
                        min_investment=100,
                        uptrend_days=10
                    )

                    try:
                        strategy = InvestmentStrategy(params)
                        engine = BacktestEngine(data, strategy)
                        engine.run_backtest()
                        metrics = engine.calculate_metrics()

                        if metrics:
                            all_results.append({
                                'parameters': params.to_dict(),
                                'metrics': metrics
                            })

                            if metrics['Sharpe Ratio'] > best_sharpe:
                                best_sharpe = metrics['Sharpe Ratio']
                                best_params = params
                                best_metrics = metrics
                    
                    except Exception as e:
                        logger.error(f"Error in iteration: {str(e)}")
                        continue

        # Save results
        if best_params and best_metrics:
            save_results(run_id, best_params, best_metrics, param_ranges, all_results)

        return best_params, best_metrics, run_id

    except Exception as e:
        logger.error(f"Error in optimize_parameters: {str(e)}")
        return None, None, run_id

def save_results(run_id: str, params: StrategyParameters, metrics: Dict, 
                param_ranges: Dict, all_results: List[Dict], 
                output_dir: str = 'optimization_results'):
    """Save optimization results to files"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Save best results
        results = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'optimal_parameters': params.to_dict(),
            'performance_metrics': metrics,
            'parameter_ranges_tested': param_ranges
        }

        best_results_file = f'{output_dir}/optimization_results_{run_id}.json'
        with open(best_results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Save all iterations
        iterations_df = pd.DataFrame([
            {**result['parameters'], 
             **{f'metric_{k}': v for k, v in result['metrics'].items()}}
            for result in all_results
        ])
        iterations_file = f'{output_dir}/iterations_{run_id}.csv'
        iterations_df.to_csv(iterations_file, index=False)

        logger.info(f"Results saved to {best_results_file} and {iterations_file}")

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare data for analysis"""
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert Date column to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Remove any rows with negative or zero prices
        data = data[data[['Open', 'High', 'Low', 'Close']].gt(0).all(axis=1)]
        
        return data

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load and prepare data
        # data = load_and_prepare_data('GoldBees/GOLDBEES_EQ_D_Min.csv')
        data = load_and_prepare_data('NiftyBees/NIFTYBEES_EQ_D_Min.csv')

        # Define parameter ranges for optimization
        param_ranges = {
            'drop_trigger': [0.0015, 0.002, 0.0025, 0.005, 0.01],
            'increment': [0.1, 0.2, 0.3, 0.4, 0.5],
            'base_investment': [100, 200, 300, 400, 500]
        }

        # Run optimization
        best_params, best_metrics, run_id = optimize_parameters(data, param_ranges)

        if best_params and best_metrics:
            print("\nOptimization Results:")
            print("\nBest Parameters:")
            for param, value in best_params.to_dict().items():
                print(f"{param}: {value}")
            print("\nPerformance Metrics:")
            for metric, value in best_metrics.items():
                print(f"{metric}: {value}")
        else:
            print("Optimization failed to find valid parameters")

    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")