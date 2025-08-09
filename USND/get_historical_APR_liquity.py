import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Liquity V2 APR Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LiquityV2APRAnalytics:
    def __init__(self, dune_api_key: str):
        self.dune_api_key = dune_api_key
        self.dune_base_url = "https://api.dune.com/api/v1"
        
        # Market configurations for Liquity V2
        self.MARKETS = {
            'WETH': {
                'display_name': 'WETH',
                'token_tag': 'ETH',
                'color': '#627EEA',
                'coingecko_id': 'weth'
            },
            'rETH': {
                'display_name': 'rETH',
                'token_tag': 'LST',
                'color': '#FF6B35',
                'coingecko_id': 'rocket-pool-eth'
            },
            'wstETH': {
                'display_name': 'wstETH',
                'token_tag': 'LST',
                'color': '#00A3FF',
                'coingecko_id': 'wrapped-steth'
            }
        }
    
    # def get_table_name(self, market_symbol: str, table_type: str) -> str:
    #     """Generate the correct Dune table name for Liquity V2"""
    #     market_lower = market_symbol.lower()
        
    #     table_mapping = {
    #         'trove_events': f"liquity_v2_ethereum.trovenft_{market_lower}_evt_transfer",
    #         'stability_pool': f"liquity_v2_ethereum.stabilitypool_{market_lower}_evt_stabilitypoolboldbalanceupdated",
    #         'trove_manager': f"liquity_v2_ethereum.trovemanager_{market_lower}_evt_troveupdated"
    #     }
        
    #     return table_mapping.get(table_type, "")
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def execute_dune_query(_self, query_sql: str, query_name: str, market_symbol: str) -> pd.DataFrame:
        """Execute a query on Dune Analytics with progress tracking"""
        try:
            with st.spinner(f"Fetching APR data for {market_symbol}..."):
                # Create query
                create_url = f"{_self.dune_base_url}/query"
                headers = {
                    "X-Dune-API-Key": _self.dune_api_key,
                    "Content-Type": "application/json"
                }
                
                create_payload = {
                    "query_sql": query_sql,
                    "name": f"Liquity V2 APR Analytics - {market_symbol} - {query_name}"
                }
                
                create_response = requests.post(create_url, json=create_payload, headers=headers, timeout=30)
                create_response.raise_for_status()
                query_id = create_response.json()["query_id"]
                
                # Execute query
                execute_url = f"{_self.dune_base_url}/query/{query_id}/execute"
                execute_response = requests.post(execute_url, headers=headers, timeout=30)
                execute_response.raise_for_status()
                execution_id = execute_response.json()["execution_id"]
                
                # Poll for results
                results_url = f"{_self.dune_base_url}/execution/{execution_id}/results"
                
                for i in range(30):  # Wait up to 5 minutes
                    time.sleep(10)
                    
                    results_response = requests.get(results_url, headers=headers, timeout=30)
                    results_response.raise_for_status()
                    result_data = results_response.json()
                    
                    if result_data["state"] == "QUERY_STATE_COMPLETED":
                        rows = result_data["result"]["rows"]
                        if rows:
                            return pd.DataFrame(rows)
                        else:
                            return pd.DataFrame()
                    elif result_data["state"] == "QUERY_STATE_FAILED":
                        raise Exception(f"Query failed: {result_data.get('error', 'Unknown error')}")
                
                raise TimeoutError("Query execution timed out")
                
        except Exception as e:
            st.error(f"Error executing {query_name} for {market_symbol}: {e}")
            return pd.DataFrame()
    
    # def get_stability_pool_apr_history(self, market_symbol: str) -> pd.DataFrame:
    #     """Get historical APR for a specific market's stability pool in Liquity V2"""
        
    #     # Get table names for Liquity V2
    #     sp_table = self.get_table_name(market_symbol, 'stability_pool')
    #     trove_table = self.get_table_name(market_symbol, 'trove_manager')
        
    #     # Note: This query structure may need adjustment based on actual Liquity V2 table schemas
    #     # The original query is adapted for Liquity V2 structure
    #     query = f"""
    #     WITH date_series AS (
    #         SELECT sequence(
    #             DATE('2024-07-01'),  -- Adjust start date for Liquity V2 launch
    #             CURRENT_DATE,
    #             INTERVAL '1' DAY
    #         ) as date_array
    #     ),
    #     dates AS (
    #         SELECT date_val as date
    #         FROM date_series
    #         CROSS JOIN UNNEST(date_array) AS t(date_val)
    #     ),
    #     -- Get stability pool balance for each day
    #     daily_stability_pool AS (
    #         SELECT
    #             DATE(evt_block_time) as date,
    #             MAX(_newBalance / 1e18) as max_sp_balance_bold
    #         FROM {sp_table}
    #         WHERE evt_block_time >= DATE('2024-07-01')
    #         GROUP BY DATE(evt_block_time)
    #     ),
    #     -- Forward-fill stability pool data for ALL days
    #     filled_sp_data AS (
    #         SELECT
    #             d.date,
    #             COALESCE(
    #                 dsp.max_sp_balance_bold,
    #                 LAG(dsp.max_sp_balance_bold) IGNORE NULLS OVER (ORDER BY d.date),
    #                 0
    #             ) as stability_pool_bold
    #         FROM dates d
    #         LEFT JOIN daily_stability_pool dsp ON d.date = dsp.date
    #     ),
    #     -- Get all troves that have ever existed
    #     all_troves AS (
    #         SELECT DISTINCT _troveId
    #         FROM {trove_table}
    #         WHERE _debt > 0
    #         AND evt_block_time >= DATE('2024-07-01')
    #     ),
    #     -- Get the latest state of each trove for each day (only when there are actual events)
    #     daily_trove_events AS (
    #         SELECT
    #             DATE(evt_block_time) as date,
    #             _troveId,
    #             _debt / 1e18 as debt,
    #             _annualInterestRate / 1e16 as interest_rate_percent,
    #             evt_block_time,
    #             ROW_NUMBER() OVER (
    #                 PARTITION BY DATE(evt_block_time), _troveId 
    #                 ORDER BY evt_block_time DESC
    #             ) as rn
    #         FROM {trove_table}
    #         WHERE _debt > 0
    #         AND evt_block_time >= DATE('2024-07-01')
    #     ),
    #     -- Get only the latest update per trove per day (when events occurred)
    #     latest_daily_events AS (
    #         SELECT
    #             date,
    #             _troveId,
    #             debt,
    #             interest_rate_percent
    #         FROM daily_trove_events
    #         WHERE rn = 1
    #     ),
    #     -- Create ALL date x trove combinations (every day for every trove)
    #     all_date_trove_combinations AS (
    #         SELECT
    #             d.date,
    #             t._troveId
    #         FROM dates d
    #         CROSS JOIN all_troves t
    #         WHERE d.date >= DATE('2024-07-01')
    #     ),
    #     -- Join with actual events and forward-fill missing values
    #     forward_filled_troves AS (
    #         SELECT
    #             adt.date,
    #             adt._troveId,
    #             COALESCE(
    #                 lde.debt,
    #                 LAG(lde.debt) IGNORE NULLS OVER (PARTITION BY adt._troveId ORDER BY adt.date),
    #                 0
    #             ) as debt,
    #             COALESCE(
    #                 lde.interest_rate_percent,
    #                 LAG(lde.interest_rate_percent) IGNORE NULLS OVER (PARTITION BY adt._troveId ORDER BY adt.date),
    #                 0
    #             ) as interest_rate_percent
    #         FROM all_date_trove_combinations adt
    #         LEFT JOIN latest_daily_events lde ON adt.date = lde.date AND adt._troveId = lde._troveId
    #     ),
    #     -- Calculate daily total interest costs for ALL days
    #     daily_interest_totals AS (
    #         SELECT
    #             date,
    #             SUM(debt * (interest_rate_percent / 100)) as total_annual_interest_cost
    #         FROM forward_filled_troves
    #         WHERE debt > 0
    #         GROUP BY date
    #     ),
    #     -- Combine stability pool and interest data for ALL days
    #     complete_apr_calculation AS (
    #         SELECT
    #             d.date,
    #             sp.stability_pool_bold,
    #             COALESCE(dit.total_annual_interest_cost, 0) as total_annual_interest_cost,
    #             -- Calculate APR: (Total interest * percentage to SP) / SP Balance * 100
    #             -- Note: Adjust the percentage (0.75 = 75%) based on Liquity V2 specifications
    #             CASE 
    #                 WHEN sp.stability_pool_bold > 0 AND COALESCE(dit.total_annual_interest_cost, 0) > 0
    #                 THEN (COALESCE(dit.total_annual_interest_cost, 0) * 0.75) / sp.stability_pool_bold * 100
    #                 ELSE 0
    #             END as stability_pool_apr_percent
    #         FROM dates d
    #         LEFT JOIN filled_sp_data sp ON d.date = sp.date
    #         LEFT JOIN daily_interest_totals dit ON d.date = dit.date
    #         WHERE d.date >= DATE('2024-07-01')
    #         AND sp.stability_pool_bold > 0
    #     )
    #     SELECT
    #         date,
    #         ROUND(stability_pool_bold, 2) as stability_pool_bold,
    #         ROUND(total_annual_interest_cost, 2) as total_annual_interest_cost,
    #         ROUND(stability_pool_apr_percent, 2) as stability_pool_apr_percent
    #     FROM complete_apr_calculation
    #     ORDER BY date DESC
    #     """
        
    #     return self.execute_dune_query(query, "Historical APR", market_symbol)
    
    def get_stability_pool_apr_history(self, market_symbol: str) -> pd.DataFrame:
        """Get historical APR for a specific market's stability pool based on actual rewards"""
        
        query = f"""
        WITH 

        interest_rewards AS (
            SELECT 
                CASE 
                    WHEN to = 0x9502b7c397e9aa22fe9db7ef7daf21cd2aebe56b THEN 'wstETH'
                    WHEN to = 0xd442e41019b7f5c4dd78f50dc03726c446148695 THEN 'rETH'
                    WHEN to = 0x5721cbbd64fc7ae3ef44a0a3f9a790a9264cf9bf THEN 'WETH'
                END AS collateral_type,
                date_trunc('day', evt_block_time) AS day, 
                SUM(value/1e18) AS bold_amount 
            FROM 
            liquity_v2_ethereum.boldToken_evt_Transfer
            WHERE to IN (0x5721cbbd64fc7ae3ef44a0a3f9a790a9264cf9bf, 0xd442e41019b7f5c4dd78f50dc03726c446148695, 0x9502b7c397e9aa22fe9db7ef7daf21cd2aebe56b)
            AND "from" = 0x0000000000000000000000000000000000000000
            GROUP BY 1, 2 
        ),

        liquidation_rewards AS (
            SELECT 
                collateral_type,
                date_trunc('day', block_time) AS day,
                SUM(collateral_sent_sp) AS liquidation_rewards
            FROM 
            query_4412204
            GROUP BY 1, 2 
        ),

        time_seq AS (
            SELECT
                sequence(
                CAST('2024-07-01' AS timestamp),
                date_trunc('day', CAST(now() AS timestamp)),
                interval '1' day
                ) AS time 
        ),

        days AS (
            SELECT
                time.time AS day 
            FROM time_seq
            CROSS JOIN unnest(time) AS time(time)
        ),

        collaterals AS (
            SELECT 
                collateral_type 
            FROM (
                VALUES 
                    ('wstETH'), ('WETH'), ('rETH')
            ) AS tmp (collateral_type)
        ),

        get_all_collaterals AS (
            SELECT 
                d.day,
                c.collateral_type
            FROM 
            days d 
            INNER JOIN 
            collaterals c 
                ON 1 = 1 
        ),

        get_all_rewards AS (
            SELECT 
                ga.day,
                ga.collateral_type,
                COALESCE(ir.bold_amount, 0) AS bold_rewards,
                COALESCE(0, 0) AS liquidation_rewards
            FROM 
            get_all_collaterals ga 
            LEFT JOIN 
            interest_rewards ir 
                ON ga.day = ir.day 
                AND ga.collateral_type = ir.collateral_type 
            LEFT JOIN 
            liquidation_rewards lr 
                ON ga.day = lr.day 
                AND ga.collateral_type = lr.collateral_type 
        ),

        get_prices AS (
            SELECT 
                date_trunc('day', minute) AS day,
                symbol, 
                max_by(price, minute) AS price 
            FROM 
            prices.usd 
            WHERE minute >= date '2024-07-01'
            AND symbol IN ('wstETH', 'WETH', 'rETH')
            AND blockchain = 'ethereum'
            GROUP BY 1, 2 
        ),

        get_liquid_usd AS (
            SELECT 
                ga.day,
                ga.collateral_type,
                ga.bold_rewards,
                COALESCE(ga.liquidation_rewards * gp.price, 0) AS liquidation_rewards 
            FROM 
            get_all_rewards ga 
            LEFT JOIN 
            get_prices gp 
                ON ga.day = gp.day 
                AND ga.collateral_type = gp.symbol 
        ),

        balances AS (
            SELECT 
                day,
                CASE 
                    WHEN address = 0x9502b7c397e9aa22fe9db7ef7daf21cd2aebe56b THEN 'wstETH'
                    WHEN address = 0xd442e41019b7f5c4dd78f50dc03726c446148695 THEN 'rETH'
                    WHEN address = 0x5721cbbd64fc7ae3ef44a0a3f9a790a9264cf9bf THEN 'WETH'
                END AS collateral_type,
                token_balance AS bold_supply
            FROM 
            query_5161019
            WHERE address IN (0x5721cbbd64fc7ae3ef44a0a3f9a790a9264cf9bf, 0xd442e41019b7f5c4dd78f50dc03726c446148695, 0x9502b7c397e9aa22fe9db7ef7daf21cd2aebe56b)
        ),

        join_balances AS (
            SELECT 
                gl.day,
                gl.collateral_type,
                gl.bold_rewards AS total_rewards,
                b.bold_supply 
            FROM 
            get_liquid_usd gl 
            INNER JOIN 
            balances b 
                ON gl.collateral_type = b.collateral_type 
                AND gl.day = b.day 
        )

        SELECT 
            day AS date,
            collateral_type,
            total_rewards,
            bold_supply AS stability_pool_bold,
            rewards,
            avg_supply,
            (rewards/avg_supply)/3 * 365 * 100 AS stability_pool_apr_percent
        FROM (
            SELECT 
                day,
                collateral_type,
                total_rewards,
                bold_supply,
                SUM(total_rewards) OVER (PARTITION BY collateral_type ORDER BY day ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rewards,
                AVG(bold_supply) OVER (PARTITION BY collateral_type ORDER BY day ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS avg_supply 
            FROM 
            join_balances
            WHERE day != current_date 
        ) 
        WHERE day >= date '2024-07-01'
        AND collateral_type = '{market_symbol}'
        ORDER BY day DESC
        """
        
        return self.execute_dune_query(query, "Actual APR", market_symbol)

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_historical_prices(_self, token_id: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical prices from CoinGecko"""
        try:
            API_KEY = "CG-1TDGd4M3qJyNN7Ujzyt3T5ZM"  # Consider moving to env variable
            url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": str(days),
                "x_cg_demo_api_key": API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "prices" in data:
                prices = data["prices"]
                
                # Convert to DataFrame
                price_data = []
                for entry in prices:
                    timestamp, price = entry
                    date = datetime.fromtimestamp(timestamp / 1000).date()
                    price_data.append({
                        'date': date,
                        'price': price
                    })
                
                df = pd.DataFrame(price_data)
                df = df.drop_duplicates(subset=['date'])  # Remove duplicate dates
                df = df.sort_values('date')
                return df
            else:
                st.error(f"No price data found for {token_id}")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error fetching {token_id} prices: {e}")
            return pd.DataFrame()

def get_price_for_date(price_df: pd.DataFrame, target_date) -> float:
    """Get price for a specific date with forward-filling"""
    if price_df.empty:
        return 1.0  # Default price
    
    # Convert target_date to date if it's datetime
    if hasattr(target_date, 'date'):
        target_date = target_date.date()
    
    # Find exact match first
    exact_match = price_df[price_df['date'] == target_date]
    if not exact_match.empty:
        return exact_match.iloc[0]['price']
    
    # Forward-fill: find the most recent price before or on the target date
    before_date = price_df[price_df['date'] <= target_date]
    if not before_date.empty:
        return before_date.iloc[-1]['price']
    
    # If no price before target date, use the earliest available price
    return price_df.iloc[0]['price']

def calculate_apr_indices(market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate equal-weighted and TVL-weighted APR indices"""
    
    # Filter out markets with no data
    active_markets = {k: v for k, v in market_data.items() if not v.empty}
    
    if not active_markets:
        return pd.DataFrame()
    
    # Prepare all market dataframes with standardized date column
    processed_markets = {}
    for market_symbol, df in active_markets.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy = df_copy.sort_values('date')
            df_copy['market'] = market_symbol
            processed_markets[market_symbol] = df_copy
    
    if not processed_markets:
        return pd.DataFrame()
    
    # Get all unique dates across all markets
    all_dates = set()
    for df in processed_markets.values():
        all_dates.update(df['date'].dt.date)
    all_dates = sorted(list(all_dates))
    
    index_data = []
    
    for date in all_dates:
        date_data = []
        
        # Collect data for this specific date from all markets
        for market_symbol, df in processed_markets.items():
            market_date_data = df[df['date'].dt.date == date]
            if not market_date_data.empty:
                latest_record = market_date_data.iloc[-1]  # Get latest record for this date
                date_data.append({
                    'market': market_symbol,
                    'apr': latest_record['stability_pool_apr_percent'],
                    'tvl': latest_record['stability_pool_bold']
                })
        
        if date_data:
            # Calculate equal-weighted index (simple average)
            aprs = [item['apr'] for item in date_data]
            equal_weighted_apr = sum(aprs) / len(aprs)
            
            # Calculate TVL-weighted index
            total_tvl = sum(item['tvl'] for item in date_data)
            if total_tvl > 0:
                tvl_weighted_apr = sum(
                    (item['tvl'] / total_tvl) * item['apr'] 
                    for item in date_data
                )
            else:
                tvl_weighted_apr = 0
            
            index_data.append({
                'date': date,
                'equal_weighted_apr': equal_weighted_apr,
                'tvl_weighted_apr': tvl_weighted_apr,
                'total_tvl': total_tvl,
                'markets_count': len(date_data)
            })
    
    return pd.DataFrame(index_data)

def calculate_strategy_indices_fast(market_data: Dict[str, pd.DataFrame], eth_prices_df: pd.DataFrame, bold_prices_df: pd.DataFrame, visible_markets: Dict[str, pd.DataFrame], eth_collateral: float, ltv_percent: float) -> pd.DataFrame:
    """Fast calculation of strategy indices using pre-loaded data for ETH collateral in Liquity V2"""
    
    if not visible_markets:
        return pd.DataFrame()
    
    # Prepare all market dataframes with standardized date column
    processed_markets = {}
    for market_symbol, df in visible_markets.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy = df_copy.sort_values('date')
            df_copy['market'] = market_symbol
            processed_markets[market_symbol] = df_copy
    
    if not processed_markets:
        return pd.DataFrame()
    
    # Get all unique dates across all markets
    all_dates = set()
    for df in processed_markets.values():
        all_dates.update(df['date'].dt.date)
    all_dates = sorted(list(all_dates))
    
    strategy_data = []
    
    for date in all_dates:
        date_data = []
        
        # Get historical prices for this date
        eth_price = get_price_for_date(eth_prices_df, date)
        bold_price = get_price_for_date(bold_prices_df, date) if not bold_prices_df.empty else 1.0
        
        # Calculate collateral value and max debt for this date
        collateral_value_usd = eth_collateral * eth_price
        max_debt_bold = collateral_value_usd * (ltv_percent / 100)
        max_debt_usd = max_debt_bold * bold_price
        
        # Collect data for this specific date from all markets
        for market_symbol, df in processed_markets.items():
            market_date_data = df[df['date'].dt.date == date]
            if not market_date_data.empty:
                latest_record = market_date_data.iloc[-1]  # Get latest record for this date
                date_data.append({
                    'market': market_symbol,
                    'original_apr': latest_record['stability_pool_apr_percent'],
                    'original_pool_size': latest_record['stability_pool_bold'],
                    'total_rewards': latest_record.get('total_rewards', 0)  # CHANGED LINE
                })
        
        if date_data and len(date_data) > 0:
            num_markets = len(date_data)
            total_pool_size = sum(item['original_pool_size'] for item in date_data)
            
            # Strategy 1: Equal allocation across all markets
            equal_allocation_per_market = max_debt_bold / num_markets
            
            # Strategy 2: TVL-weighted allocation (proportional to pool sizes)
            strategy_1_apr = 0  # Equal allocation
            strategy_2_apr = 0  # TVL-weighted allocation
            
            # Calculate adjusted APRs after adding our deposits
            for item in date_data:
                # Strategy 1: Equal allocation
                new_pool_size_equal = item['original_pool_size'] + equal_allocation_per_market
                # Use dilution factor since we have actual APR now
                if item['original_apr'] > 0 and new_pool_size_equal > 0:  # CHANGED
                    dilution_factor = item['original_pool_size'] / new_pool_size_equal
                    adjusted_apr_equal = item['original_apr'] * dilution_factor  # CHANGED
                else:
                    adjusted_apr_equal = 0
                
                # Strategy 2: TVL-weighted allocation
                if total_pool_size > 0:
                    tvl_weight = item['original_pool_size'] / total_pool_size
                    tvl_allocation = max_debt_bold * tvl_weight
                    new_pool_size_tvl = item['original_pool_size'] + tvl_allocation
                    
                    if item['original_apr'] > 0 and new_pool_size_tvl > 0:  # CHANGED
                        dilution_factor_tvl = item['original_pool_size'] / new_pool_size_tvl
                        adjusted_apr_tvl = item['original_apr'] * dilution_factor_tvl  # CHANGED
                    else:
                        adjusted_apr_tvl = 0
                else:
                    adjusted_apr_tvl = 0
                    tvl_allocation = 0
                
                # Weight the APRs by allocation amount
                weight_equal = 1 / num_markets
                strategy_1_apr += weight_equal * adjusted_apr_equal
                
                if total_pool_size > 0:
                    weight_tvl = item['original_pool_size'] / total_pool_size
                    strategy_2_apr += weight_tvl * adjusted_apr_tvl
            
            # Calculate yields in both BOLD and USD (using adjusted APRs)
            annual_yield_equal_bold = max_debt_bold * (strategy_1_apr / 100)
            annual_yield_tvl_bold = max_debt_bold * (strategy_2_apr / 100)
            annual_yield_equal_usd = annual_yield_equal_bold * bold_price
            annual_yield_tvl_usd = annual_yield_tvl_bold * bold_price
            
            # Calculate the original (non-adjusted) APRs for comparison
            original_equal_apr = sum(item['original_apr'] for item in date_data) / num_markets
            original_tvl_apr = sum(
                (item['original_pool_size'] / total_pool_size) * item['original_apr'] 
                for item in date_data
            ) if total_pool_size > 0 else 0
            
            strategy_data.append({
                'date': date,
                'eth_collateral': eth_collateral,
                'eth_price': eth_price,
                'bold_price': bold_price,
                'ltv_percent': ltv_percent,
                'max_debt_bold': max_debt_bold,
                'max_debt_usd': max_debt_usd,
                'collateral_value_usd': collateral_value_usd,
                'equal_allocation_apr': strategy_1_apr,
                'tvl_weighted_allocation_apr': strategy_2_apr,
                'original_equal_apr': original_equal_apr,
                'original_tvl_apr': original_tvl_apr,
                'equal_allocation_per_market': equal_allocation_per_market,
                'total_pool_size': total_pool_size,
                'markets_count': num_markets,
                'annual_yield_equal_bold': annual_yield_equal_bold,
                'annual_yield_tvl_bold': annual_yield_tvl_bold,
                'annual_yield_equal_usd': annual_yield_equal_usd,
                'annual_yield_tvl_usd': annual_yield_tvl_usd,
                'apr_impact_equal': original_equal_apr - strategy_1_apr,
                'apr_impact_tvl': original_tvl_apr - strategy_2_apr
            })
    
    return pd.DataFrame(strategy_data)

def create_markets_comparison_tab(visible_markets: Dict[str, pd.DataFrame], analytics: LiquityV2APRAnalytics):
    """Create markets comparison tab"""
    st.subheader("All Markets APR Comparison")
    
    # Combined chart
    fig = go.Figure()
    
    for market_symbol, df in visible_markets.items():
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            config = analytics.MARKETS[market_symbol]
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['stability_pool_apr_percent'],
                mode='lines+markers',
                name=f"{config['display_name']} ({config['token_tag']})",
                line=dict(color=config.get('color', '#1f77b4'), width=2),
                hovertemplate=f"<b>{config['display_name']}</b><br>" +
                            "Date: %{x}<br>" +
                            "APR: %{y:.2f}%<br>" +
                            "<extra></extra>"
            ))
    
    fig.update_layout(
        title="Historical APR Comparison Across Selected Markets",
        xaxis_title="Date",
        yaxis_title="APR (%)",
        height=600,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 APR Statistics Summary")
    
    stats_data = []
    for market_symbol, df in visible_markets.items():
        if not df.empty:
            config = analytics.MARKETS[market_symbol]
            current_apr = df.iloc[-1]['stability_pool_apr_percent'] if len(df) > 0 else 0
            avg_apr = df['stability_pool_apr_percent'].mean()
            max_apr = df['stability_pool_apr_percent'].max()
            min_apr = df['stability_pool_apr_percent'].min()
            
            stats_data.append({
                'Market': config['display_name'],
                'Type': config['token_tag'],
                'Current APR (%)': f"{current_apr:.2f}%",
                'Average APR (%)': f"{avg_apr:.2f}%",
                'Max APR (%)': f"{max_apr:.2f}%",
                'Min APR (%)': f"{min_apr:.2f}%"
            })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)

def create_apr_indices_tab(visible_markets: Dict[str, pd.DataFrame], analytics: LiquityV2APRAnalytics):
    """Create APR indices tab"""
    st.subheader("📈 Liquity V2 APR Indices")
    st.markdown("Composite indices tracking overall protocol performance")
    
    # Calculate indices
    indices_df = calculate_apr_indices(visible_markets)
    
    if not indices_df.empty:
        indices_df['date'] = pd.to_datetime(indices_df['date'])
        indices_df = indices_df.sort_values('date')
        
        # Index comparison chart
        fig_indices = go.Figure()
        
        # Equal-weighted index
        fig_indices.add_trace(go.Scatter(
            x=indices_df['date'],
            y=indices_df['equal_weighted_apr'],
            mode='lines+markers',
            name='Equal-Weighted APR Index',
            line=dict(color='#FF6B6B', width=3),
            hovertemplate="<b>Equal-Weighted Index</b><br>" +
                        "Date: %{x}<br>" +
                        "APR: %{y:.2f}%<br>" +
                        "<extra></extra>"
        ))
        
        # TVL-weighted index
        fig_indices.add_trace(go.Scatter(
            x=indices_df['date'],
            y=indices_df['tvl_weighted_apr'],
            mode='lines+markers',
            name='TVL-Weighted APR Index',
            line=dict(color='#4ECDC4', width=3),
            hovertemplate="<b>TVL-Weighted Index</b><br>" +
                        "Date: %{x}<br>" +
                        "APR: %{y:.2f}%<br>" +
                        "<extra></extra>"
        ))
        
        fig_indices.update_layout(
            title="Liquity V2 APR Indices Comparison",
            xaxis_title="Date",
            yaxis_title="APR (%)",
            height=600,
            template="plotly_white",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_indices, use_container_width=True)
        
        # Index metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_equal = indices_df.iloc[-1]['equal_weighted_apr']
        current_tvl = indices_df.iloc[-1]['tvl_weighted_apr']
        avg_equal = indices_df['equal_weighted_apr'].mean()
        avg_tvl = indices_df['tvl_weighted_apr'].mean()
        
        with col1:
            st.metric("Current Equal-Weighted APR", f"{current_equal:.2f}%")
        with col2:
            st.metric("Current TVL-Weighted APR", f"{current_tvl:.2f}%")
        with col3:
            st.metric("Avg Equal-Weighted APR", f"{avg_equal:.2f}%")
        with col4:
            st.metric("Avg TVL-Weighted APR", f"{avg_tvl:.2f}%")
        
        # Index methodology explanation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Index Methodology")
            st.markdown("""
            **Equal-Weighted Index:**
            - Simple average of all market APRs
            - Each market has equal influence (1/3 weight)
            - Formula: `Σ(APR_i) / N`
            
            **TVL-Weighted Index:**
            - Weighted average based on stability pool sizes
            - Larger pools have more influence
            - Formula: `Σ(APR_i × TVL_i) / Σ(TVL_i)`
            """)
        
        with col2:
            st.subheader("🔍 Index Analysis")
            
            # Calculate correlation
            correlation = indices_df['equal_weighted_apr'].corr(indices_df['tvl_weighted_apr'])
            
            # Calculate spread
            current_spread = abs(current_equal - current_tvl)
            avg_spread = abs(indices_df['equal_weighted_apr'] - indices_df['tvl_weighted_apr']).mean()
            
            st.metric("Index Correlation", f"{correlation:.3f}")
            st.metric("Current Spread", f"{current_spread:.2f}%")
            st.metric("Average Spread", f"{avg_spread:.2f}%")
            
            # Interpretation
            if current_tvl > current_equal:
                st.info("💡 **TVL-weighted > Equal-weighted**: Larger pools are outperforming smaller ones")
            elif current_equal > current_tvl:
                st.info("💡 **Equal-weighted > TVL-weighted**: Smaller pools are outperforming larger ones")
            else:
                st.info("💡 **Indices aligned**: Performance is balanced across pool sizes")
    else:
        st.warning("Unable to calculate indices - insufficient data")

def create_strategy_simulator_tab(visible_markets: Dict[str, pd.DataFrame], analytics: LiquityV2APRAnalytics, eth_prices_df: pd.DataFrame, bold_prices_df: pd.DataFrame):
    """Create strategy simulator tab with fast recalculation for ETH collateral"""
    st.subheader("🎯 ETH Collateral Strategy Simulator")
    st.markdown("Simulate borrowing against ETH collateral and deploying BOLD across stability pools")
    
    # Strategy parameters
    col1, col2 = st.columns(2)
    
    with col1:
        eth_collateral = st.number_input(
            "ETH Collateral Amount",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=0.1,
            help="Amount of ETH to use as collateral"
        )
    
    with col2:
        ltv_percent = st.slider(
            "Loan-to-Value Ratio (%)",
            min_value=0.0,
            max_value=91.0,  # Liquity V2 max LTV
            value=80.0,
            step=1.0,
            help="Percentage of collateral value to borrow (max ~91% for ETH in Liquity V2)"
        )
    
    # Fast calculation using cached data
    strategy_df = calculate_strategy_indices_fast(
        visible_markets, 
        eth_prices_df, 
        bold_prices_df, 
        visible_markets, 
        eth_collateral, 
        ltv_percent
    )
    
    if not strategy_df.empty:
        strategy_df['date'] = pd.to_datetime(strategy_df['date'])
        strategy_df = strategy_df.sort_values('date')
        
        # Display strategy overview
        latest_strategy = strategy_df.iloc[-1]
        
        st.subheader("💰 Strategy Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "ETH Price", 
                f"${latest_strategy['eth_price']:,.0f}",
                help="Current ETH price from CoinGecko"
            )
        
        with col2:
            st.metric(
                "BOLD Price", 
                f"${latest_strategy['bold_price']:.4f}",
                help="Current BOLD price from CoinGecko"
            )
        
        with col3:
            st.metric(
                "Collateral Value", 
                f"${latest_strategy['collateral_value_usd']:,.0f}",
                help="USD value of ETH collateral"
            )
        
        with col4:
            st.metric(
                "Max BOLD Debt", 
                f"{latest_strategy['max_debt_bold']:,.0f} BOLD",
                help="Maximum BOLD that can be borrowed"
            )
        
        with col5:
            st.metric(
                "Max Debt (USD)", 
                f"${latest_strategy['max_debt_usd']:,.0f}",
                help="USD value of maximum debt"
            )
        
        # Strategy APR comparison chart
        fig_strategy = go.Figure()
        
        # Equal allocation strategy
        fig_strategy.add_trace(go.Scatter(
            x=strategy_df['date'],
            y=strategy_df['equal_allocation_apr'],
            mode='lines+markers',
            name='Equal Allocation Strategy',
            line=dict(color='#FF6B6B', width=3),
            hovertemplate="<b>Equal Allocation</b><br>" +
                        "Date: %{x}<br>" +
                        "APR: %{y:.2f}%<br>" +
                        "<extra></extra>"
        ))
        
        # TVL-weighted allocation strategy
        fig_strategy.add_trace(go.Scatter(
            x=strategy_df['date'],
            y=strategy_df['tvl_weighted_allocation_apr'],
            mode='lines+markers',
            name='TVL-Weighted Allocation Strategy',
            line=dict(color='#4ECDC4', width=3),
            hovertemplate="<b>TVL-Weighted Allocation</b><br>" +
                        "Date: %{x}<br>" +
                        "APR: %{y:.2f}%<br>" +
                        "<extra></extra>"
        ))
        
        fig_strategy.update_layout(
            title=f"Strategy APR Comparison ({eth_collateral} ETH @ {ltv_percent}% LTV)",
            xaxis_title="Date",
            yaxis_title="APR (%)",
            height=600,
            template="plotly_white",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_strategy, use_container_width=True)
        
        # Strategy performance metrics with APR impact analysis
        col1, col2, col3, col4 = st.columns(4)
        
        current_equal_apr = latest_strategy['equal_allocation_apr']
        current_tvl_apr = latest_strategy['tvl_weighted_allocation_apr']
        original_equal_apr = latest_strategy['original_equal_apr']
        original_tvl_apr = latest_strategy['original_tvl_apr']
        
        with col1:
            delta_equal = f"{latest_strategy['apr_impact_equal']:.2f}% impact"
            st.metric(
                "Adjusted Equal APR", 
                f"{current_equal_apr:.2f}%",
                delta=delta_equal,
                delta_color="inverse",
                help="APR after adding your deposits (lower due to increased pool size)"
            )
        
        with col2:
            delta_tvl = f"{latest_strategy['apr_impact_tvl']:.2f}% impact"
            st.metric(
                "Adjusted TVL APR", 
                f"{current_tvl_apr:.2f}%",
                delta=delta_tvl,
                delta_color="inverse",
                help="APR after adding your deposits (lower due to increased pool size)"
            )
        
        with col3:
            st.metric(
                "Original Equal APR", 
                f"{original_equal_apr:.2f}%",
                help="Original APR before your deposits"
            )
        
        with col4:
            st.metric(
                "Original TVL APR", 
                f"{original_tvl_apr:.2f}%",
                help="Original APR before your deposits"
            )
        
        # Annual yield projections
        st.subheader("💵 Annual Yield Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Equal Allocation Strategy**")
            annual_yield_equal_bold = latest_strategy['annual_yield_equal_bold']
            annual_yield_equal_usd = latest_strategy['annual_yield_equal_usd']
            
            st.metric("Annual Yield (BOLD)", f"{annual_yield_equal_bold:,.0f} BOLD")
            st.metric("Annual Yield (USD)", f"${annual_yield_equal_usd:,.0f}")
            
            # ROI calculation
            roi_equal = (annual_yield_equal_usd / latest_strategy['collateral_value_usd']) * 100
            st.metric("ROI on Collateral", f"{roi_equal:.2f}%")
        
        with col2:
            st.markdown("**TVL-Weighted Strategy**")
            annual_yield_tvl_bold = latest_strategy['annual_yield_tvl_bold']
            annual_yield_tvl_usd = latest_strategy['annual_yield_tvl_usd']
            
            st.metric("Annual Yield (BOLD)", f"{annual_yield_tvl_bold:,.0f} BOLD")
            st.metric("Annual Yield (USD)", f"${annual_yield_tvl_usd:,.0f}")
            
            # ROI calculation
            roi_tvl = (annual_yield_tvl_usd / latest_strategy['collateral_value_usd']) * 100
            st.metric("ROI on Collateral", f"{roi_tvl:.2f}%")
        
        # Strategy comparison analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 APR Impact Analysis")
            
            strategy_diff = current_equal_apr - current_tvl_apr
            yield_diff_usd = annual_yield_equal_usd - annual_yield_tvl_usd
            
            # Show the impact of deposits on APRs
            equal_impact = latest_strategy['apr_impact_equal']
            tvl_impact = latest_strategy['apr_impact_tvl']
            
            st.metric("APR Difference", f"{strategy_diff:.2f}%", delta=f"Equal vs TVL-weighted")
            st.metric("Equal APR Impact", f"-{equal_impact:.2f}%", help="How much your deposits reduced the equal allocation APR")
            st.metric("TVL APR Impact", f"-{tvl_impact:.2f}%", help="How much your deposits reduced the TVL-weighted APR")
            st.metric("Yield Diff (USD)", f"${yield_diff_usd:,.0f}")
            
            # Interpretation with APR impact context
            if strategy_diff > 0:
                st.success("✅ **Equal allocation strategy outperforming**")
                st.info("💡 Even after reducing APRs, smaller pools offer better yields")
            elif strategy_diff < 0:
                st.success("✅ **TVL-weighted strategy outperforming**")
                st.info("💡 Even after reducing APRs, larger pools offer better yields")
            else:
                st.info("⚖️ **Strategies performing equally after APR adjustments**")
        
        with col2:
            st.subheader("🎯 Allocation Breakdown")
            
            # Show current allocation for each strategy
            if not visible_markets.items():
                st.warning("No market data available")
            else:
                allocation_data = []
                total_pool_size = latest_strategy['total_pool_size']
                
                for market_symbol, df in visible_markets.items():
                    if not df.empty:
                        latest_market = df.iloc[0]
                        pool_size = latest_market['stability_pool_bold']
                        apr = latest_market['stability_pool_apr_percent']
                        
                        equal_allocation = latest_strategy['max_debt_bold'] / latest_strategy['markets_count']
                        tvl_weight = pool_size / total_pool_size if total_pool_size > 0 else 0
                        tvl_allocation = latest_strategy['max_debt_bold'] * tvl_weight
                        
                        # Calculate adjusted APRs
                        #interest_cost = latest_market['total_annual_interest_cost']
                        # Calculate adjusted APRs - use dilution approach since we have actual APR
                        original_apr = latest_market['stability_pool_apr_percent']  # CHANGED
                        new_pool_equal = pool_size + equal_allocation
                        new_pool_tvl = pool_size + tvl_allocation
                        
                        # adjusted_apr_equal = (interest_cost * 0.75) / new_pool_equal * 100 if new_pool_equal > 0 and interest_cost > 0 else 0
                        # adjusted_apr_tvl = (interest_cost * 0.75) / new_pool_tvl * 100 if new_pool_tvl > 0 and interest_cost > 0 else 0
                        adjusted_apr_equal = original_apr * (pool_size / new_pool_equal) if new_pool_equal > 0 and original_apr > 0 else 0
                        adjusted_apr_tvl = original_apr * (pool_size / new_pool_tvl) if new_pool_tvl > 0 and original_apr > 0 else 0
                        
                        allocation_data.append({
                            'Market': analytics.MARKETS[market_symbol]['display_name'],
                            'Original APR (%)': f"{apr:.2f}%",
                            'Equal APR (%)': f"{adjusted_apr_equal:.2f}%",
                            'TVL APR (%)': f"{adjusted_apr_tvl:.2f}%",
                            'Equal Allocation': f"{equal_allocation:,.0f} BOLD",
                            'TVL Allocation': f"{tvl_allocation:,.0f} BOLD",
                            'Original Pool': f"{pool_size:,.0f} BOLD"
                        })
                
                allocation_df = pd.DataFrame(allocation_data)
                st.dataframe(allocation_df, use_container_width=True)
        
        # Historical strategy data
        st.subheader("📋 Historical Strategy Performance")
        
        display_strategy_df = strategy_df[['date', 'eth_price', 'bold_price', 'equal_allocation_apr', 'tvl_weighted_allocation_apr', 'annual_yield_equal_usd', 'annual_yield_tvl_usd']].copy()
        display_strategy_df.columns = ['Date', 'ETH Price ($)', 'BOLD Price ($)', 'Equal APR (%)', 'TVL-Weighted APR (%)', 'Equal Yield ($)', 'TVL Yield ($)']
        display_strategy_df['Date'] = display_strategy_df['Date'].astype(str)
        display_strategy_df['ETH Price ($)'] = display_strategy_df['ETH Price ($)'].apply(lambda x: f"${x:,.0f}")
        display_strategy_df['BOLD Price ($)'] = display_strategy_df['BOLD Price ($)'].apply(lambda x: f"${x:.4f}")
        display_strategy_df['Equal Yield ($)'] = display_strategy_df['Equal Yield ($)'].apply(lambda x: f"${x:,.0f}")
        display_strategy_df['TVL Yield ($)'] = display_strategy_df['TVL Yield ($)'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_strategy_df.head(30), use_container_width=True)
        
    else:
        st.warning("Unable to calculate strategy performance - insufficient data")

def create_individual_analysis_tab(visible_markets: Dict[str, pd.DataFrame], analytics: LiquityV2APRAnalytics):
    """Create individual market analysis tab"""
    st.subheader("Individual Market Deep Dive")
    
    visible_market_options = list(visible_markets.keys())
    if visible_market_options:
        selected_market = st.selectbox(
            "Select Market for Detailed Analysis",
            options=visible_market_options,
            format_func=lambda x: f"{analytics.MARKETS[x]['display_name']} ({analytics.MARKETS[x]['token_tag']})"
        )
        
        if selected_market and selected_market in visible_markets:
            df = visible_markets[selected_market]
            config = analytics.MARKETS[selected_market]
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_apr = df.iloc[0]['stability_pool_apr_percent']
                avg_apr = df['stability_pool_apr_percent'].mean()
                current_pool_size = df.iloc[0]['stability_pool_bold']
                
                with col1:
                    st.metric("Current APR", f"{current_apr:.2f}%")
                
                with col2:
                    st.metric("Average APR", f"{avg_apr:.2f}%")
                
                with col3:
                    st.metric("Pool Size", f"{current_pool_size:,.0f} BOLD")
                
                with col4:
                    st.metric("Data Points", len(df))
                
                # Detailed charts
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        f"{config['display_name']} APR Over Time",
                        f"{config['display_name']} Stability Pool Size"
                    ),
                    vertical_spacing=0.1
                )
                
                # APR chart
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['stability_pool_apr_percent'],
                        mode='lines+markers',
                        name='APR (%)',
                        line=dict(color=config.get('color', '#1f77b4'), width=3),
                        fill='tonexty'
                    ),
                    row=1, col=1
                )
                
                # Pool size chart
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['stability_pool_bold'],
                        mode='lines+markers',
                        name='Pool Size (BOLD)',
                        line=dict(color='#ff7f0e', width=3),
                        fill='tonexty'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=800,
                    template="plotly_white",
                    showlegend=True
                )
                
                fig.update_yaxes(title_text="APR (%)", row=1, col=1)
                fig.update_yaxes(title_text="Pool Size (BOLD)", row=2, col=1)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.subheader("📋 Historical Data Table")
                #display_df = df[['date', 'stability_pool_apr_percent', 'stability_pool_bold', 'total_annual_interest_cost']].copy()
                display_df = df[['date', 'stability_pool_apr_percent', 'stability_pool_bold', 'total_rewards']].copy()
                #display_df.columns = ['Date', 'APR (%)', 'Pool Size (BOLD)', 'Total Interest (BOLD)']
                display_df.columns = ['Date', 'APR (%)', 'Pool Size (BOLD)', 'Total Rewards (BOLD)']
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df.head(30), use_container_width=True)
    else:
        st.info("No visible markets selected for analysis")

def create_current_apr_tab(visible_markets: Dict[str, pd.DataFrame], analytics: LiquityV2APRAnalytics):
    """Create current APR summary tab"""
    st.subheader("🎯 Current APR Snapshot")
    
    # Current APR comparison
    current_data = []
    for market_symbol, df in visible_markets.items():
        if not df.empty:
            config = analytics.MARKETS[market_symbol]
            current_record = df.iloc[0]
            
            current_data.append({
                'Market': config['display_name'],
                'Type': config['token_tag'],
                'Current APR': current_record['stability_pool_apr_percent'],
                'Pool Size': current_record['stability_pool_bold'],
                'Total Rewards': current_record.get('total_rewards', 0)  # CHANGED
            })
    
    if current_data:
        current_df = pd.DataFrame(current_data)
        current_df = current_df.sort_values('Current APR', ascending=False)
        
        # APR ranking chart
        fig = px.bar(
            current_df,
            x='Market',
            y='Current APR',
            color='Type',
            title="Current APR Ranking Across Selected Markets",
            labels={'Current APR': 'APR (%)'},
            height=500
        )
        
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed current metrics
        st.subheader("📊 Detailed Current Metrics")
        
        display_current_df = current_df.copy()
        display_current_df['Current APR'] = display_current_df['Current APR'].apply(lambda x: f"{x:.2f}%")
        display_current_df['Pool Size'] = display_current_df['Pool Size'].apply(lambda x: f"{x:,.0f} BOLD")
        #display_current_df['Interest Cost'] = display_current_df['Interest Cost'].apply(lambda x: f"{x:,.0f} BOLD")
        display_current_df['Total Rewards'] = display_current_df['Total Rewards'].apply(lambda x: f"{x:,.0f} BOLD")
        
        st.dataframe(display_current_df, use_container_width=True)
    else:
        st.info("No data available for current APR snapshot")

def create_apr_dashboard_with_cache(market_data: Dict[str, pd.DataFrame], analytics: LiquityV2APRAnalytics, eth_prices_df: pd.DataFrame, bold_prices_df: pd.DataFrame):
    """Create the main APR dashboard using cached data for fast recalculation"""
    
    # Filter out markets with no data and create toggles
    active_markets = {k: v for k, v in market_data.items() if not v.empty}
    
    if not active_markets:
        st.error("No data available for any markets")
        return
    
    # Market visibility toggles
    st.subheader("🎯 Market Visibility Controls")
    
    # Create columns for toggle switches (only 3 markets for Liquity V2)
    cols = st.columns(3)
    market_visibility = {}
    
    market_list = list(active_markets.keys())
    for i, market_symbol in enumerate(market_list):
        config = analytics.MARKETS[market_symbol]
        with cols[i % 3]:
            market_visibility[market_symbol] = st.checkbox(
                f"{config['display_name']} ({config['token_tag']})",
                value=True,
                key=f"toggle_{market_symbol}"
            )
    
    # Filter markets based on visibility
    visible_markets = {k: v for k, v in active_markets.items() if market_visibility.get(k, False)}
    
    if not visible_markets:
        st.warning("Please select at least one market to display")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 All Markets Comparison", "📈 APR Indices", "🎯 Strategy Simulator", "🔍 Individual Market Analysis", "📋 Current APR Summary"])
    
    with tab1:
        create_markets_comparison_tab(visible_markets, analytics)
    
    with tab2:
        create_apr_indices_tab(visible_markets, analytics)
    
    with tab3:
        create_strategy_simulator_tab(visible_markets, analytics, eth_prices_df, bold_prices_df)
    
    with tab4:
        create_individual_analysis_tab(visible_markets, analytics)
    
    with tab5:
        create_current_apr_tab(visible_markets, analytics)

def main():
    # Header
    st.title("📈 Liquity V2 Stability Pool APR Dashboard")
    st.markdown("Real-time APR analysis for Liquity V2 stability pools on Ethereum")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    dune_api_key = st.sidebar.text_input(
        "Dune API Key", 
        value="",  # Remove default key for security
        type="password",
        help="Enter your Dune Analytics API key"
    )
    
    if not dune_api_key:
        st.error("Please provide a Dune API key in the sidebar")
        st.info("You can get a free API key from [Dune Analytics](https://dune.com/)")
        return
    
    # Initialize analytics
    analytics = LiquityV2APRAnalytics(dune_api_key)
    
    # Data refresh controls
    st.sidebar.header("🔄 Data Controls")
    if st.sidebar.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        if 'market_data_loaded' in st.session_state:
            st.session_state.market_data_loaded = False
    
    # Check if we have data in session state
    if 'market_data_loaded' not in st.session_state:
        st.session_state.market_data_loaded = False
        st.session_state.market_data = {}
        st.session_state.eth_prices_df = pd.DataFrame()
        st.session_state.bold_prices_df = pd.DataFrame()
    
    # Load data only once or when refresh is requested
    if not st.session_state.market_data_loaded:
        # Load ALL markets (WETH, rETH, wstETH)
        all_markets = list(analytics.MARKETS.keys())
        
        # Fetch data for ALL markets
        st.info("🔄 Loading APR data for all Liquity V2 markets. This may take a few minutes...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch price data first
        status_text.text("Fetching historical price data...")
        progress_bar.progress(0.1)
        
        try:
            st.session_state.eth_prices_df = analytics.get_historical_prices("ethereum")
            st.session_state.bold_prices_df = analytics.get_historical_prices("liquity-bold")  # Correct CoinGecko ID for BOLD
        except Exception as e:
            st.error(f"Failed to fetch price data: {e}")
            st.warning("Continuing without price data - strategy simulator will use default prices")
            # Create default price data for ETH and BOLD
            import datetime
            today = datetime.date.today()
            dates = [today - datetime.timedelta(days=i) for i in range(30)]
            
            # Default ETH price around $3000 and BOLD around $1
            st.session_state.eth_prices_df = pd.DataFrame([
                {'date': date, 'price': 3000.0} for date in dates
            ])
            st.session_state.bold_prices_df = pd.DataFrame([
                {'date': date, 'price': 1.0} for date in dates
            ])
        
        # Fetch market data
        market_data = {}
        for i, market in enumerate(all_markets):
            status_text.text(f"Fetching APR data for {analytics.MARKETS[market]['display_name']}...")
            progress_bar.progress((i + 1) / (len(all_markets) + 1))
            
            try:
                df = analytics.get_stability_pool_apr_history(market)
                market_data[market] = df
                if not df.empty:
                    st.sidebar.success(f"✅ {analytics.MARKETS[market]['display_name']}: {len(df)} records")
                else:
                    st.sidebar.warning(f"⚠️ {analytics.MARKETS[market]['display_name']}: No data")
            except Exception as e:
                st.sidebar.error(f"❌ {analytics.MARKETS[market]['display_name']}: Failed")
                st.sidebar.text(f"Error: {str(e)[:100]}...")
                market_data[market] = pd.DataFrame()
        
        progress_bar.progress(1.0)
        status_text.text("✅ Data loading complete!")
        
        # Store in session state
        st.session_state.market_data = market_data
        st.session_state.market_data_loaded = True
        
        # Remove progress indicators after a short delay
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    # Use cached data
    market_data = st.session_state.market_data
    eth_prices_df = st.session_state.eth_prices_df
    bold_prices_df = st.session_state.bold_prices_df
    
    # Show loading summary
    loaded_markets = len([k for k, v in market_data.items() if not v.empty])
    st.success(f"🎯 Successfully loaded data for {loaded_markets}/{len(analytics.MARKETS)} markets")
    
    # Calculate and display index summary
    if loaded_markets > 0:
        indices_df = calculate_apr_indices(market_data)
        if not indices_df.empty:
            current_equal = indices_df.iloc[-1]['equal_weighted_apr']
            current_tvl = indices_df.iloc[-1]['tvl_weighted_apr']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Current Equal-Weighted APR Index", f"{current_equal:.2f}%")
            with col2:
                st.metric("📈 Current TVL-Weighted APR Index", f"{current_tvl:.2f}%")
    
    # Create dashboard with cached data
    create_apr_dashboard_with_cache(market_data, analytics, eth_prices_df, bold_prices_df)
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Source:** Dune Analytics")
        st.markdown("**APR Calculation:** 75% of trove interest to SP")
    
    with col2:
        st.markdown("**Protocol:** Liquity V2")
        st.markdown("**Network:** Ethereum Mainnet")
    
    with col3:
        markets_loaded = len([k for k, v in market_data.items() if not v.empty])
        st.markdown("**Markets Tracked:** 3 (WETH, rETH, wstETH)")
        st.markdown(f"**Markets Loaded:** {markets_loaded}/3")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page to restart the application.")
        
        # Reset on major errors
        if st.button("🔄 Reset Application"):
            st.cache_data.clear()
            st.rerun()
