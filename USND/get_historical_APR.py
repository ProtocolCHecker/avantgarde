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
    page_title="Nerite Protocol APR Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class NeriteSuperAPRAnalytics:
    def __init__(self, dune_api_key: str):
        self.dune_api_key = dune_api_key
        self.dune_base_url = "https://api.dune.com/api/v1"
        
        # Market configurations
        self.MARKETS = {
            'WETH': {
                'display_name': 'WETH',
                'token_tag': 'ETH',
                'color': '#627EEA',
                'stability_pool_address': '0x9d9ef87a197c1bb3a97b2ddc8716df99079c125e',
                'is_weth_exception': True
            },
            'wstETH': {
                'display_name': 'wstETH',
                'token_tag': 'LST',
                'color': '#00A3FF',
                'stability_pool_address': '0xcd94b16e9a126fe61c944b1de024681fcfe05c4b',
                'is_weth_exception': False
            },
            'rETH': {
                'display_name': 'rETH',
                'token_tag': 'LST',
                'color': '#FF6B35',
                'stability_pool_address': '0x47ae276a1cc751ce7b3034d9cbb8cd422968ac35',
                'is_weth_exception': False
            },
            'rsETH': {
                'display_name': 'rsETH (Kelp)',
                'token_tag': 'LRT',
                'color': '#10B981',
                'stability_pool_address': '0xafb439c47b3f518a7d8ef3b82f70df30d84e51ee',
                'is_weth_exception': False
            },
            'weETH': {
                'display_name': 'weETH (Etherfi)',
                'token_tag': 'LRT',
                'color': '#8B5CF6',
                'stability_pool_address': '0x9c3aef8fb9097bb59821422d47f226e35403019a',
                'is_weth_exception': False
            },
            'ARB': {
                'display_name': 'ARB',
                'token_tag': 'ARB',
                'color': '#2563EB',
                'stability_pool_address': '0xb2c0460466c8d6384f52cd29db54ee49d01ee84a',
                'is_weth_exception': False
            },
            'COMP': {
                'display_name': 'COMP',
                'token_tag': 'COMP',
                'color': '#00D395',
                'stability_pool_address': '0x65b83de0733e237dd3d49a4e9c2868b57ee7d9f0',
                'is_weth_exception': False
            },
            'tBTC': {
                'display_name': 'tBTC (Threshold)',
                'token_tag': 'BTC',
                'color': '#F7931A',
                'stability_pool_address': '0xe1fa1f28a67a8807447717f51bf3305636962126',
                'is_weth_exception': False
            }
        }
    
    # def get_table_name(self, market_symbol: str, table_type: str) -> str:
    #     """Generate the correct Dune table name based on market and table type"""
    #     config = self.MARKETS.get(market_symbol, {})
        
    #     if not config:
    #         raise ValueError(f"Unknown market: {market_symbol}")
        
    #     # Special handling for WETH tables
    #     if config['is_weth_exception']:
    #         if table_type == 'stabilitypool_balance':
    #             return "nerite_arbitrum.stabilitypool_weth_evt_stabilitypoolboldbalanceupdated"
    #         elif table_type == 'trovemanager':
    #             return "nerite_arbitrum.trovemanager_weth_evt_troveupdated"
        
    #     # Standard table naming for all other cases
    #     market_lower = market_symbol.lower()
        
    #     table_mapping = {
    #         'stabilitypool_balance': f"nerite_arbitrum.stabilitypool_{market_lower}_evt_stabilitypoolboldbalanceupdated",
    #         'trovemanager': f"nerite_arbitrum.trovemanager_{market_lower}_evt_troveupdated"
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
                    "name": f"Nerite APR Analytics - {market_symbol} - {query_name}"
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
    #     """Get historical APR for a specific market's stability pool"""
        
    #     # Get table names using the same logic as your original script
    #     sp_table = self.get_table_name(market_symbol, 'stabilitypool_balance')
    #     trove_table = self.get_table_name(market_symbol, 'trovemanager')
        
    #     query = f"""
    #     WITH date_series AS (
    #         SELECT sequence(
    #             DATE('2024-07-12'),
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
    #         WHERE evt_block_time >= DATE('2024-07-12')
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
    #         AND evt_block_time >= DATE('2024-07-12')
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
    #         AND evt_block_time >= DATE('2024-07-12')
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
    #         WHERE d.date >= DATE('2024-07-12')
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
    #             -- Calculate APR: (Total interest * 75% to SP) / SP Balance * 100
    #             CASE 
    #                 WHEN sp.stability_pool_bold > 0 AND COALESCE(dit.total_annual_interest_cost, 0) > 0
    #                 THEN (COALESCE(dit.total_annual_interest_cost, 0) * 0.75) / sp.stability_pool_bold * 100
    #                 ELSE 0
    #             END as stability_pool_apr_percent
    #         FROM dates d
    #         LEFT JOIN filled_sp_data sp ON d.date = sp.date
    #         LEFT JOIN daily_interest_totals dit ON d.date = dit.date
    #         WHERE d.date >= DATE('2024-07-12')
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
        """Get historical APR for a specific market's stability pool using real rewards data"""
        
        query = """
        with 
        interest_rewards as (
            select 
                case 
                    when to = 0x9d9ef87a197c1bb3a97b2ddc8716df99079c125e then 'WETH'
                    when to = 0xcd94b16e9a126fe61c944b1de024681fcfe05c4b then 'wstETH'
                    when to = 0x47ae276a1cc751ce7b3034d9cbb8cd422968ac35 then 'rETH'
                    when to = 0xafb439c47b3f518a7d8ef3b82f70df30d84e51ee then 'rsETH'
                    when to = 0x9c3aef8fb9097bb59821422d47f226e35403019a then 'weETH'
                    when to = 0xb2c0460466c8d6384f52cd29db54ee49d01ee84a then 'ARB'
                    when to = 0x65b83de0733e237dd3d49a4e9c2868b57ee7d9f0 then 'COMP'
                    when to = 0xe1fa1f28a67a8807447717f51bf3305636962126 then 'tBTC'
                end as collateral_type,
                date_trunc('day', evt_block_time) as day, 
                sum(value/1e18) as bold_amount 
            from 
            (select * from nerite_arbitrum.boldtoken_evt_transfer
            where contract_address = 0x4ecf61a6c2fab8a047ceb3b3b263b401763e9d49)
            where to in (0x9d9ef87a197c1bb3a97b2ddc8716df99079c125e, 0xcd94b16e9a126fe61c944b1de024681fcfe05c4b, 0x47ae276a1cc751ce7b3034d9cbb8cd422968ac35, 0xafb439c47b3f518a7d8ef3b82f70df30d84e51ee, 0x9c3aef8fb9097bb59821422d47f226e35403019a, 0xb2c0460466c8d6384f52cd29db54ee49d01ee84a, 0x65b83de0733e237dd3d49a4e9c2868b57ee7d9f0, 0xe1fa1f28a67a8807447717f51bf3305636962126)
            and "from" = 0x0000000000000000000000000000000000000000
            group by 1, 2 
        ),

        liquidation_rewards as (
            select 
                collateral_type,
                date_trunc('day', block_time) as day,
                sum(collateral_sent_sp) as liquidation_rewards
            from 
            query_5482233
            group by 1, 2 
        ),

        time_seq AS (
            select
                sequence(
                CAST('2024-07-12' as timestamp),
                date_trunc('day', cast(now() as timestamp)),
                interval '1' day
                ) as time 
        ),

        days AS (
            select
                time.time as day 
            from time_seq
            cross join unnest(time) as time(time)
        ),

        collaterals as (
            select 
                collateral_type 
            from (
                values 
                    ('WETH'), ('wstETH'), ('rETH'), ('rsETH'), ('weETH'), ('ARB'), ('COMP'), ('tBTC')
            ) as tmp (collateral_type)
        ),

        get_all_collaterals as (
            select 
                d.day,
                c.collateral_type
            from 
            days d 
            inner join 
            collaterals c 
                on 1 = 1 
        ),

        get_all_rewards as (
            select 
                ga.day,
                ga.collateral_type,
                coalesce(ir.bold_amount, 0) as bold_rewards,
                coalesce(0, 0) as liquidation_rewards
            from 
            get_all_collaterals ga 
            left join 
            interest_rewards ir 
                on ga.day = ir.day 
                and ga.collateral_type = ir.collateral_type 
            left join 
            liquidation_rewards lr 
                on ga.day = lr.day 
                and ga.collateral_type = lr.collateral_type 
        ),

        get_prices as (
            select 
                date_trunc('day', minute) as day,
                symbol, 
                max_by(price, minute) as price 
            from 
            prices.usd 
            where minute >= date '2024-07-12'
            and symbol in ('WETH', 'wstETH', 'rETH', 'rsETH', 'weETH', 'ARB', 'COMP', 'tBTC')
            and blockchain = 'arbitrum'
            group by 1, 2 
        ),

        get_liquid_usd as (
            select 
                ga.day,
                ga.collateral_type,
                ga.bold_rewards,
                coalesce(ga.liquidation_rewards * gp.price, 0) as liquidation_rewards 
            from 
            get_all_rewards ga 
            left join 
            get_prices gp 
                on ga.day = gp.day 
                and ga.collateral_type = gp.symbol 
        ),

        balances as (
            select 
                day,
                case 
                    when address = 0x9d9ef87a197c1bb3a97b2ddc8716df99079c125e then 'WETH'
                    when address = 0xcd94b16e9a126fe61c944b1de024681fcfe05c4b then 'wstETH'
                    when address = 0x47ae276a1cc751ce7b3034d9cbb8cd422968ac35 then 'rETH'
                    when address = 0xafb439c47b3f518a7d8ef3b82f70df30d84e51ee then 'rsETH'
                    when address = 0x9c3aef8fb9097bb59821422d47f226e35403019a then 'weETH'
                    when address = 0xb2c0460466c8d6384f52cd29db54ee49d01ee84a then 'ARB'
                    when address = 0x65b83de0733e237dd3d49a4e9c2868b57ee7d9f0 then 'COMP'
                    when address = 0xe1fa1f28a67a8807447717f51bf3305636962126 then 'tBTC'
                end as collateral_type,
                token_balance as bold_supply
            from 
            query_5482281
            where address in (0x9d9ef87a197c1bb3a97b2ddc8716df99079c125e, 0xcd94b16e9a126fe61c944b1de024681fcfe05c4b, 0x47ae276a1cc751ce7b3034d9cbb8cd422968ac35, 0xafb439c47b3f518a7d8ef3b82f70df30d84e51ee, 0x9c3aef8fb9097bb59821422d47f226e35403019a, 0xb2c0460466c8d6384f52cd29db54ee49d01ee84a, 0x65b83de0733e237dd3d49a4e9c2868b57ee7d9f0, 0xe1fa1f28a67a8807447717f51bf3305636962126)
        ),

        join_balances as (
            select 
                gl.day,
                gl.collateral_type,
                gl.bold_rewards as total_rewards,
                b.bold_supply 
            from 
            get_liquid_usd gl 
            inner join 
            balances b 
                on gl.collateral_type = b.collateral_type 
                and gl.day = b.day 
        )

        select 
            day as date,
            collateral_type,
            total_rewards,
            bold_supply as stability_pool_bold,
            rewards,
            avg_supply,
            (rewards/avg_supply)/3 * 365 * 100 as stability_pool_apr_percent
        from (
        select 
            day,
            collateral_type,
            total_rewards,
            bold_supply,
            sum(total_rewards) over (partition by collateral_type order by day rows between 2 preceding and current row) as rewards,
            avg(bold_supply) over (partition by collateral_type order by day rows between 2 preceding and current row) as avg_supply 
        from 
        join_balances
        where day != current_date 
        ) 
        where day >= date '2024-07-14'
        and collateral_type = '{market_symbol}'
        order by day desc
        """
        
        formatted_query = query.format(market_symbol=market_symbol)
        return self.execute_dune_query(formatted_query, "Real APR History", market_symbol)

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_historical_prices(_self, token_id: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical prices from CoinGecko"""
        try:
            API_KEY = "CG-1TDGd4M3qJyNN7Ujzyt3T5ZM"
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

def calculate_strategy_indices_fast(market_data: Dict[str, pd.DataFrame], tbtc_prices_df: pd.DataFrame, usnd_prices_df: pd.DataFrame, visible_markets: Dict[str, pd.DataFrame], tbtc_collateral: float, ltv_percent: float) -> pd.DataFrame:
    """Fast calculation of strategy indices using pre-loaded data"""
    
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
        tbtc_price = get_price_for_date(tbtc_prices_df, date)
        usnd_price = get_price_for_date(usnd_prices_df, date) if not usnd_prices_df.empty else 1.0
        
        # Calculate collateral value and max debt for this date
        collateral_value_usd = tbtc_collateral * tbtc_price
        max_debt_usnd = collateral_value_usd * (ltv_percent / 100)
        max_debt_usd = max_debt_usnd * usnd_price
        
        # Collect data for this specific date from all markets
        for market_symbol, df in processed_markets.items():
            market_date_data = df[df['date'].dt.date == date]
            if not market_date_data.empty:
                latest_record = market_date_data.iloc[-1]  # Get latest record for this date
                # date_data.append({
                #     'market': market_symbol,
                #     'original_apr': latest_record['stability_pool_apr_percent'],
                #     'original_pool_size': latest_record['stability_pool_bold'],
                #     'total_annual_interest_cost': latest_record['total_annual_interest_cost']
                # })
                date_data.append({
                    'market': market_symbol,
                    'original_apr': latest_record['stability_pool_apr_percent'],
                    'original_pool_size': latest_record['stability_pool_bold'],
                    'total_rewards': latest_record.get('total_rewards', 0),
                    'daily_rewards': latest_record.get('rewards', 0)
                })
        
        if date_data and len(date_data) > 0:
            num_markets = len(date_data)
            total_pool_size = sum(item['original_pool_size'] for item in date_data)
            
            # Strategy 1: Equal allocation across all markets
            equal_allocation_per_market = max_debt_usnd / num_markets
            
            # Strategy 2: TVL-weighted allocation (proportional to pool sizes)
            strategy_1_apr = 0  # Equal allocation
            strategy_2_apr = 0  # TVL-weighted allocation
            
            # Calculate adjusted APRs after adding our deposits
            # for item in date_data:
            #     # Strategy 1: Equal allocation
            #     new_pool_size_equal = item['original_pool_size'] + equal_allocation_per_market
            #     if item['total_annual_interest_cost'] > 0 and new_pool_size_equal > 0:
            #         adjusted_apr_equal = (item['total_annual_interest_cost'] * 0.75) / new_pool_size_equal * 100
            #     else:
            #         adjusted_apr_equal = 0
                
            #     # Strategy 2: TVL-weighted allocation
            #     if total_pool_size > 0:
            #         tvl_weight = item['original_pool_size'] / total_pool_size
            #         tvl_allocation = max_debt_usnd * tvl_weight
            #         new_pool_size_tvl = item['original_pool_size'] + tvl_allocation
                    
            #         if item['total_annual_interest_cost'] > 0 and new_pool_size_tvl > 0:
            #             adjusted_apr_tvl = (item['total_annual_interest_cost'] * 0.75) / new_pool_size_tvl * 100
            #         else:
            #             adjusted_apr_tvl = 0
            #     else:
            #         adjusted_apr_tvl = 0
            #         tvl_allocation = 0
            for item in date_data:
                original_pool_size = item['original_pool_size']
                daily_rewards = item.get('daily_rewards', 0)  # Real 3-day rolling rewards
                
                # Strategy 1: Equal allocation
                equal_allocation_per_market = max_debt_usnd / num_markets
                new_pool_size_equal = original_pool_size + equal_allocation_per_market
                
                # Calculate adjusted APR: same rewards spread over larger pool
                if new_pool_size_equal > 0 and daily_rewards > 0:
                    # Using the same calculation as in the query: (rewards/avg_supply)/3 * 365 * 100
                    adjusted_apr_equal = (daily_rewards / new_pool_size_equal) / 3 * 365 * 100
                else:
                    adjusted_apr_equal = item['original_apr']
                
                # Strategy 2: TVL-weighted allocation
                if total_pool_size > 0:
                    tvl_weight = item['original_pool_size'] / total_pool_size
                    tvl_allocation = max_debt_usnd * tvl_weight
                    new_pool_size_tvl = item['original_pool_size'] + tvl_allocation
                    
                    if new_pool_size_tvl > 0 and daily_rewards > 0:
                        adjusted_apr_tvl = (daily_rewards / new_pool_size_tvl) / 3 * 365 * 100
                    else:
                        adjusted_apr_tvl = item['original_apr']
                else:
                    adjusted_apr_tvl = 0
                    tvl_allocation = 0
                
                # Weight the APRs by allocation amount
                weight_equal = 1 / num_markets
                strategy_1_apr += weight_equal * adjusted_apr_equal
                
                if total_pool_size > 0:
                    weight_tvl = item['original_pool_size'] / total_pool_size
                    strategy_2_apr += weight_tvl * adjusted_apr_tvl
            
            # Calculate yields in both USND and USD (using adjusted APRs)
            annual_yield_equal_usnd = max_debt_usnd * (strategy_1_apr / 100)
            annual_yield_tvl_usnd = max_debt_usnd * (strategy_2_apr / 100)
            annual_yield_equal_usd = annual_yield_equal_usnd * usnd_price
            annual_yield_tvl_usd = annual_yield_tvl_usnd * usnd_price
            
            # Calculate the original (non-adjusted) APRs for comparison
            original_equal_apr = sum(item['original_apr'] for item in date_data) / num_markets
            original_tvl_apr = sum(
                (item['original_pool_size'] / total_pool_size) * item['original_apr'] 
                for item in date_data
            ) if total_pool_size > 0 else 0
            
            strategy_data.append({
                'date': date,
                'tbtc_collateral': tbtc_collateral,
                'tbtc_price': tbtc_price,
                'usnd_price': usnd_price,
                'ltv_percent': ltv_percent,
                'max_debt_usnd': max_debt_usnd,
                'max_debt_usd': max_debt_usd,
                'collateral_value_usd': collateral_value_usd,
                'equal_allocation_apr': strategy_1_apr,
                'tvl_weighted_allocation_apr': strategy_2_apr,
                'original_equal_apr': original_equal_apr,
                'original_tvl_apr': original_tvl_apr,
                'equal_allocation_per_market': equal_allocation_per_market,
                'total_pool_size': total_pool_size,
                'markets_count': num_markets,
                'annual_yield_equal_usnd': annual_yield_equal_usnd,
                'annual_yield_tvl_usnd': annual_yield_tvl_usnd,
                'annual_yield_equal_usd': annual_yield_equal_usd,
                'annual_yield_tvl_usd': annual_yield_tvl_usd,
                'apr_impact_equal': original_equal_apr - strategy_1_apr,
                'apr_impact_tvl': original_tvl_apr - strategy_2_apr
            })
    
    return pd.DataFrame(strategy_data)

def create_markets_comparison_tab(visible_markets: Dict[str, pd.DataFrame], analytics: NeriteSuperAPRAnalytics):
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
            current_apr = df.iloc[0]['stability_pool_apr_percent'] if len(df) > 0 else 0
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

def create_apr_indices_tab(visible_markets: Dict[str, pd.DataFrame], analytics: NeriteSuperAPRAnalytics):
    """Create APR indices tab"""
    st.subheader("📈 Nerite Protocol APR Indices")
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
            title="Nerite Protocol APR Indices Comparison",
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
            - Each market has equal influence (1/N weight)
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

def create_strategy_simulator_tab(visible_markets: Dict[str, pd.DataFrame], analytics: NeriteSuperAPRAnalytics, tbtc_prices_df: pd.DataFrame, usnd_prices_df: pd.DataFrame):
    """Create strategy simulator tab with fast recalculation"""
    st.subheader("🎯 tBTC Collateral Strategy Simulator")
    st.markdown("Simulate borrowing against tBTC collateral and deploying USND across stability pools")
    
    # Strategy parameters
    col1, col2 = st.columns(2)
    
    with col1:
        tbtc_collateral = st.number_input(
            "tBTC Collateral Amount",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1,
            help="Amount of tBTC to use as collateral"
        )
    
    with col2:
        ltv_percent = st.slider(
            "Loan-to-Value Ratio (%)",
            min_value=0.0,
            max_value=86.96,
            value=70.0,
            step=1.0,
            help="Percentage of collateral value to borrow (max 86.96% for tBTC)"
        )
    
    # Fast calculation using cached data
    strategy_df = calculate_strategy_indices_fast(
        visible_markets, 
        tbtc_prices_df, 
        usnd_prices_df, 
        visible_markets, 
        tbtc_collateral, 
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
                "tBTC Price", 
                f"${latest_strategy['tbtc_price']:,.0f}",
                help="Current tBTC price from CoinGecko"
            )
        
        with col2:
            st.metric(
                "USND Price", 
                f"${latest_strategy['usnd_price']:.4f}",
                help="Current USND price from CoinGecko"
            )
        
        with col3:
            st.metric(
                "Collateral Value", 
                f"${latest_strategy['collateral_value_usd']:,.0f}",
                help="USD value of tBTC collateral"
            )
        
        with col4:
            st.metric(
                "Max USND Debt", 
                f"{latest_strategy['max_debt_usnd']:,.0f} USND",
                help="Maximum USND that can be borrowed"
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
            title=f"Strategy APR Comparison ({tbtc_collateral} tBTC @ {ltv_percent}% LTV)",
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
            annual_yield_equal_usnd = latest_strategy['annual_yield_equal_usnd']
            annual_yield_equal_usd = latest_strategy['annual_yield_equal_usd']
            
            st.metric("Annual Yield (USND)", f"{annual_yield_equal_usnd:,.0f} USND")
            st.metric("Annual Yield (USD)", f"${annual_yield_equal_usd:,.0f}")
            
            # ROI calculation
            roi_equal = (annual_yield_equal_usd / latest_strategy['collateral_value_usd']) * 100
            st.metric("ROI on Collateral", f"{roi_equal:.2f}%")
        
        with col2:
            st.markdown("**TVL-Weighted Strategy**")
            annual_yield_tvl_usnd = latest_strategy['annual_yield_tvl_usnd']
            annual_yield_tvl_usd = latest_strategy['annual_yield_tvl_usd']
            
            st.metric("Annual Yield (USND)", f"{annual_yield_tvl_usnd:,.0f} USND")
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
                        
                        equal_allocation = latest_strategy['max_debt_usnd'] / latest_strategy['markets_count']
                        tvl_weight = pool_size / total_pool_size if total_pool_size > 0 else 0
                        tvl_allocation = latest_strategy['max_debt_usnd'] * tvl_weight
                        
                        # Calculate adjusted APRs
                        # interest_cost = latest_market['total_annual_interest_cost']
                        # new_pool_equal = pool_size + equal_allocation
                        # new_pool_tvl = pool_size + tvl_allocation
                        
                        # adjusted_apr_equal = (interest_cost * 0.75) / new_pool_equal * 100 if new_pool_equal > 0 and interest_cost > 0 else 0
                        # adjusted_apr_tvl = (interest_cost * 0.75) / new_pool_tvl * 100 if new_pool_tvl > 0 and interest_cost > 0 else 0
                        daily_rewards = latest_market.get('rewards', 0)
                        new_pool_equal = pool_size + equal_allocation
                        new_pool_tvl = pool_size + tvl_allocation

                        adjusted_apr_equal = (daily_rewards / new_pool_equal) / 3 * 365 * 100 if new_pool_equal > 0 and daily_rewards > 0 else apr
                        adjusted_apr_tvl = (daily_rewards / new_pool_tvl) / 3 * 365 * 100 if new_pool_tvl > 0 and daily_rewards > 0 else apr
                        
                        allocation_data.append({
                            'Market': analytics.MARKETS[market_symbol]['display_name'],
                            'Original APR (%)': f"{apr:.2f}%",
                            'Equal APR (%)': f"{adjusted_apr_equal:.2f}%",
                            'TVL APR (%)': f"{adjusted_apr_tvl:.2f}%",
                            'Equal Allocation': f"{equal_allocation:,.0f} USND",
                            'TVL Allocation': f"{tvl_allocation:,.0f} USND",
                            'Original Pool': f"{pool_size:,.0f} USND"
                        })
                
                allocation_df = pd.DataFrame(allocation_data)
                st.dataframe(allocation_df, use_container_width=True)
        
        # Historical strategy data
        st.subheader("📋 Historical Strategy Performance")
        
        #display_strategy_df = strategy_df[['date', 'tbtc_price', 'usnd_price', 'equal_allocation_apr', 'tvl_weighted_allocation_apr', 'annual_yield_equal_usd', 'annual_yield_tvl_usd']].copy()
        display_strategy_df = strategy_df[['date', 'tbtc_price', 'usnd_price', 'equal_allocation_apr', 'tvl_weighted_allocation_apr', 'annual_yield_equal_usd', 'annual_yield_tvl_usd']].copy()
        display_strategy_df.columns = ['Date', 'tBTC Price ($)', 'USND Price ($)', 'Equal APR (%)', 'TVL-Weighted APR (%)', 'Equal Yield ($)', 'TVL Yield ($)']
        display_strategy_df['Date'] = display_strategy_df['Date'].astype(str)
        display_strategy_df['tBTC Price ($)'] = display_strategy_df['tBTC Price ($)'].apply(lambda x: f"${x:,.0f}")
        display_strategy_df['USND Price ($)'] = display_strategy_df['USND Price ($)'].apply(lambda x: f"${x:.4f}")
        display_strategy_df['Equal Yield ($)'] = display_strategy_df['Equal Yield ($)'].apply(lambda x: f"${x:,.0f}")
        display_strategy_df['TVL Yield ($)'] = display_strategy_df['TVL Yield ($)'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_strategy_df.head(30), use_container_width=True)
        
    else:
        st.warning("Unable to calculate strategy performance - insufficient data")

def create_individual_analysis_tab(visible_markets: Dict[str, pd.DataFrame], analytics: NeriteSuperAPRAnalytics):
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
                    st.metric("Pool Size", f"{current_pool_size:,.0f} USND")
                
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
                        name='Pool Size (USND)',
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
                fig.update_yaxes(title_text="Pool Size (USND)", row=2, col=1)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.subheader("📋 Historical Data Table")
                # display_df = df[['date', 'stability_pool_apr_percent', 'stability_pool_bold', 'total_annual_interest_cost']].copy()
                # display_df.columns = ['Date', 'APR (%)', 'Pool Size (USND)', 'Total Interest (USND)']
                display_df = df[['date', 'stability_pool_apr_percent', 'stability_pool_bold', 'total_rewards']].copy()
                display_df.columns = ['Date', 'APR (%)', 'Pool Size (USND)', 'Total Rewards (USND)']
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df.head(30), use_container_width=True)
    else:
        st.info("No visible markets selected for analysis")

def create_current_apr_tab(visible_markets: Dict[str, pd.DataFrame], analytics: NeriteSuperAPRAnalytics):
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
                'Interest Cost': current_record['total_rewards']
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
        display_current_df['Pool Size'] = display_current_df['Pool Size'].apply(lambda x: f"{x:,.0f} USND")
        display_current_df['Interest Cost'] = display_current_df['Interest Cost'].apply(lambda x: f"{x:,.0f} USND")
        
        st.dataframe(display_current_df, use_container_width=True)
    else:
        st.info("No data available for current APR snapshot")

def create_apr_dashboard_with_cache(market_data: Dict[str, pd.DataFrame], analytics: NeriteSuperAPRAnalytics, tbtc_prices_df: pd.DataFrame, usnd_prices_df: pd.DataFrame):
    """Create the main APR dashboard using cached data for fast recalculation"""
    
    # Filter out markets with no data and create toggles
    active_markets = {k: v for k, v in market_data.items() if not v.empty}
    
    if not active_markets:
        st.error("No data available for any markets")
        return
    
    # Market visibility toggles
    st.subheader("🎯 Market Visibility Controls")
    
    # Create columns for toggle switches
    cols = st.columns(4)
    market_visibility = {}
    
    market_list = list(active_markets.keys())
    for i, market_symbol in enumerate(market_list):
        config = analytics.MARKETS[market_symbol]
        with cols[i % 4]:
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
        create_strategy_simulator_tab(visible_markets, analytics, tbtc_prices_df, usnd_prices_df)
    
    with tab4:
        create_individual_analysis_tab(visible_markets, analytics)
    
    with tab5:
        create_current_apr_tab(visible_markets, analytics)

def main():
    # Header
    st.title("📈 Nerite Protocol Stability Pool APR Dashboard")
    st.markdown("Real-time APR analysis for all Nerite Protocol stability pools")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    dune_api_key = st.sidebar.text_input(
        "Dune API Key", 
        value="WwYWQED1j1RSp9iWDXokA8rsmFtskfwI",
        type="password"
    )
    
    if not dune_api_key:
        st.error("Please provide a Dune API key in the sidebar")
        return
    
    # Initialize analytics
    analytics = NeriteSuperAPRAnalytics(dune_api_key)
    
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
        st.session_state.tbtc_prices_df = pd.DataFrame()
        st.session_state.usnd_prices_df = pd.DataFrame()
    
    # Load data only once or when refresh is requested
    if not st.session_state.market_data_loaded:
        # Load ALL markets by default
        all_markets = list(analytics.MARKETS.keys())
        
        # Fetch data for ALL markets
        st.info("🔄 Loading APR data for all markets. This may take a few minutes...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch price data first
        status_text.text("Fetching historical price data...")
        progress_bar.progress(0.1)
        
        try:
            st.session_state.tbtc_prices_df = analytics.get_historical_prices("tbtc")
            st.session_state.usnd_prices_df = analytics.get_historical_prices("us-nerite-dollar")
        except Exception as e:
            st.error(f"Failed to fetch price data: {e}")
            return
        
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
    tbtc_prices_df = st.session_state.tbtc_prices_df
    usnd_prices_df = st.session_state.usnd_prices_df
    
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
    create_apr_dashboard_with_cache(market_data, analytics, tbtc_prices_df, usnd_prices_df)
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Source:** Dune Analytics")
        st.markdown("**APR Calculation:** 75% of trove interest to SP")
    
    with col2:
        st.markdown("**Protocol:** Nerite (Liquity v3 fork)")
        st.markdown("**Network:** Arbitrum")
    
    with col3:
        markets_loaded = len([k for k, v in market_data.items() if not v.empty])
        st.markdown("**Markets**")


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

