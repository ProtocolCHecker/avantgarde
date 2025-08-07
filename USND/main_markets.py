import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp

# Page configuration
st.set_page_config(
    page_title="Nerite Protocol Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

class NeriteSuperAnalytics:
    def __init__(self, dune_api_key: str):
        self.dune_api_key = dune_api_key
        self.dune_base_url = "https://api.dune.com/api/v1"
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        
        # Market configurations
        self.MARKETS = {
            'WETH': {
                'display_name': 'WETH',
                'token_tag': 'ETH',
                'mcr': 110,
                'ccr': 150,
                'scr': 110,
                'initial_debt_limit': 100_000_000,
                'ltv': 90.91,
                'coingecko_id': 'weth',
                'stability_pool_address': '0x9d9ef87a197c1bb3a97b2ddc8716df99079c125e',
                'is_weth_exception': True  # Special table naming
            },
            'wstETH': {
                'display_name': 'wstETH',
                'token_tag': 'LST',
                'mcr': 110,
                'ccr': 160,
                'scr': 110,
                'initial_debt_limit': 25_000_000,
                'ltv': 90.91,
                'coingecko_id': 'wrapped-steth',
                'stability_pool_address': '0xcd94b16e9a126fe61c944b1de024681fcfe05c4b',
                'is_weth_exception': False
            },
            'rETH': {
                'display_name': 'rETH',
                'token_tag': 'LST',
                'mcr': 110,
                'ccr': 160,
                'scr': 110,
                'initial_debt_limit': 25_000_000,
                'ltv': 90.91,
                'coingecko_id': 'rocket-pool-eth',
                'stability_pool_address': '0x47ae276a1cc751ce7b3034d9cbb8cd422968ac35',
                'is_weth_exception': False
            },
            'rsETH': {
                'display_name': 'rsETH (Kelp)',
                'token_tag': 'LRT',
                'mcr': 130,
                'ccr': 160,
                'scr': 115,
                'initial_debt_limit': 5_000_000,
                'ltv': 76.92,
                'coingecko_id': 'kelp-dao-restaked-eth',
                'stability_pool_address': '0xafb439c47b3f518a7d8ef3b82f70df30d84e51ee',
                'is_weth_exception': False
            },
            'weETH': {
                'display_name': 'weETH (Etherfi)',
                'token_tag': 'LRT',
                'mcr': 130,
                'ccr': 160,
                'scr': 115,
                'initial_debt_limit': 2_000_000,
                'ltv': 76.92,
                'coingecko_id': 'wrapped-eeth',
                'stability_pool_address': '0x9c3aef8fb9097bb59821422d47f226e35403019a',
                'is_weth_exception': False
            },
            'ARB': {
                'display_name': 'ARB',
                'token_tag': 'ARB',
                'mcr': 140,
                'ccr': 165,
                'scr': 3,  # Note: Special low SCR for ARB
                'initial_debt_limit': 5_000_000,
                'ltv': 71.43,
                'coingecko_id': 'arbitrum',
                'stability_pool_address': '0xb2c0460466c8d6384f52cd29db54ee49d01ee84a',
                'is_weth_exception': False
            },
            'COMP': {
                'display_name': 'COMP',
                'token_tag': 'COMP',
                'mcr': 140,
                'ccr': 165,
                'scr': 115,
                'initial_debt_limit': 2_000_000,
                'ltv': 71.43,
                'coingecko_id': 'compound-governance-token',
                'stability_pool_address': '0x65b83de0733e237dd3d49a4e9c2868b57ee7d9f0',
                'is_weth_exception': False
            },
            'tBTC': {
                'display_name': 'tBTC (Threshold)',
                'token_tag': 'BTC',
                'mcr': 115,
                'ccr': 160,
                'scr': 110,
                'initial_debt_limit': 5_000_000,
                'ltv': 86.96,
                'coingecko_id': 'tbtc',
                'stability_pool_address': '0xe1fa1f28a67a8807447717f51bf3305636962126',
                'is_weth_exception': False
            }
        }
        
        # USND configuration
        self.USND_COINGECKO_ID = 'us-nerite-dollar'
        
        # Current market state
        self.current_market = None
        self.current_market_config = None
    
    def get_market_config(self, market_symbol: str) -> Dict:
        """Get configuration for a specific market"""
        return self.MARKETS.get(market_symbol, {})
    
    def get_table_name(self, market_symbol: str, table_type: str) -> str:
        """Generate the correct Dune table name based on market and table type"""
        config = self.get_market_config(market_symbol)
        
        if not config:
            raise ValueError(f"Unknown market: {market_symbol}")
        
        # Special handling for WETH activepool tables
        if config['is_weth_exception'] and table_type in ['activepool_evt', 'activepool_call']:
            if table_type == 'activepool_evt':
                return "nerite_arbitrum.activepool_evt_activepoolcollbalanceupdated"
            elif table_type == 'activepool_call':
                return "nerite_arbitrum.activepool_call_getbolddebt"
        
        # Standard table naming for all other cases
        market_lower = market_symbol.lower()
        
        table_mapping = {
            'activepool_evt': f"nerite_arbitrum.{market_lower}_activepool_evt_activepoolcollbalanceupdated",
            'activepool_call': f"nerite_arbitrum.{market_lower}_activepool_call_getbolddebt",
            'stabilitypool_balance': f"nerite_arbitrum.stabilitypool_{market_lower}_evt_stabilitypoolboldbalanceupdated",
            'trovemanager': f"nerite_arbitrum.trovemanager_{market_lower}_evt_troveupdated",
            'stabilitypool_deposit': f"nerite_arbitrum.stabilitypool_{market_lower}_evt_depositoperation",
            'boldtoken_transfer': "nerite_arbitrum.boldtoken_evt_transfer"
        }
        
        return table_mapping.get(table_type, "")
    
    def set_current_market(self, market_symbol: str):
        """Set the current market for analysis"""
        if market_symbol in self.MARKETS:
            self.current_market = market_symbol
            self.current_market_config = self.MARKETS[market_symbol]
            
            # Clear cache when switching markets to prevent stale data
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
        else:
            raise ValueError(f"Unknown market: {market_symbol}")
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_coingecko_prices(_self, market_symbol: str) -> Dict[str, float]:
        """Fetch current prices from CoinGecko for the specified market"""
        try:
            config = _self.get_market_config(market_symbol)
            if not config:
                return {'market_price': 0, 'usnd_price': 1}
            
            url = f"{_self.coingecko_base_url}/simple/price"
            params = {
                'ids': f"{config['coingecko_id']},{_self.USND_COINGECKO_ID}",
                'vs_currencies': 'usd'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'market_price': data.get(config['coingecko_id'], {}).get('usd', 0),
                'usnd_price': data.get(_self.USND_COINGECKO_ID, {}).get('usd', 1)
            }
        except Exception as e:
            st.error(f"Error fetching CoinGecko prices for {market_symbol}: {e}")
            # Fallback prices based on market type
            fallback_prices = {
                'WETH': 3500, 'wstETH': 3700, 'rETH': 3600, 'rsETH': 3500, 'weETH': 3500,
                'ARB': 1.2, 'COMP': 80, 'tBTC': 95000
            }
            return {
                'market_price': fallback_prices.get(market_symbol, 1),
                'usnd_price': 1
            }
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def execute_dune_query(_self, query_sql: str, query_name: str, market_symbol: str) -> pd.DataFrame:
        """Execute a query on Dune Analytics with progress tracking"""
        try:
            with st.spinner(f"Executing {query_name} for {market_symbol}..."):
                # Create query
                create_url = f"{_self.dune_base_url}/query"
                headers = {
                    "X-Dune-API-Key": _self.dune_api_key,
                    "Content-Type": "application/json"
                }
                
                create_payload = {
                    "query_sql": query_sql,
                    "name": f"Nerite Analytics - {market_symbol} - {query_name}"
                }
                
                create_response = requests.post(create_url, json=create_payload, headers=headers, timeout=30)
                create_response.raise_for_status()
                query_id = create_response.json()["query_id"]
                
                # Execute query
                execute_url = f"{_self.dune_base_url}/query/{query_id}/execute"
                execute_response = requests.post(execute_url, headers=headers, timeout=30)
                execute_response.raise_for_status()
                execution_id = execute_response.json()["execution_id"]
                
                # Poll for results with progress bar
                results_url = f"{_self.dune_base_url}/execution/{execution_id}/results"
                progress_bar = st.progress(0)
                
                for i in range(30):  # Wait up to 5 minutes
                    time.sleep(10)
                    progress_bar.progress((i + 1) / 30)
                    
                    results_response = requests.get(results_url, headers=headers, timeout=30)
                    results_response.raise_for_status()
                    result_data = results_response.json()
                    
                    if result_data["state"] == "QUERY_STATE_COMPLETED":
                        progress_bar.progress(100)
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
    
    def get_historical_system_metrics(self, market_symbol: str) -> pd.DataFrame:
        """Get historical total collateral, debt and SP deposit data for specified market"""
        config = self.get_market_config(market_symbol)
        if not config:
            return pd.DataFrame()
        
        # Get table names
        collateral_table = self.get_table_name(market_symbol, 'activepool_evt')
        debt_table = self.get_table_name(market_symbol, 'activepool_call')
        sp_table = self.get_table_name(market_symbol, 'stabilitypool_balance')
        
        query = f"""
        WITH daily_collateral AS (
            SELECT
                date_trunc('day', evt_block_time) as date,
                MAX(_collBalance / 1e18) as max_collateral_{market_symbol.lower()}
            FROM {collateral_table}
            GROUP BY date_trunc('day', evt_block_time)
        ),
        daily_debt AS (
            SELECT
                date_trunc('day', call_block_time) as date,
                MAX(output_0 / 1e18) as max_debt_bold
            FROM {debt_table}
            WHERE call_success = true
            GROUP BY date_trunc('day', call_block_time)
        ),
        daily_stability_pool AS (
            SELECT
                date_trunc('day', evt_block_time) as date,
                MAX(_newBalance / 1e18) as max_sp_balance_bold
            FROM {sp_table}
            GROUP BY date_trunc('day', evt_block_time)
        ),
        all_dates AS (
            SELECT DISTINCT date FROM daily_collateral
            UNION
            SELECT DISTINCT date FROM daily_debt
            UNION
            SELECT DISTINCT date FROM daily_stability_pool
        ),
        filled_data AS (
            SELECT
                ad.date,
                COALESCE(
                    dc.max_collateral_{market_symbol.lower()},
                    LAG(dc.max_collateral_{market_symbol.lower()}) IGNORE NULLS OVER (ORDER BY ad.date),
                    0
                ) as collateral_{market_symbol.lower()},
                COALESCE(
                    dd.max_debt_bold,
                    LAG(dd.max_debt_bold) IGNORE NULLS OVER (ORDER BY ad.date),
                    0
                ) as debt_bold,
                COALESCE(
                    dsp.max_sp_balance_bold,
                    LAG(dsp.max_sp_balance_bold) IGNORE NULLS OVER (ORDER BY ad.date),
                    0
                ) as stability_pool_bold
            FROM all_dates ad
            LEFT JOIN daily_collateral dc ON ad.date = dc.date
            LEFT JOIN daily_debt dd ON ad.date = dd.date
            LEFT JOIN daily_stability_pool dsp ON ad.date = dsp.date
        )
        SELECT
            date,
            ROUND(collateral_{market_symbol.lower()}, 2) as collateral_{market_symbol.lower()},
            ROUND(debt_bold, 2) as debt_bold,
            ROUND(stability_pool_bold, 2) as stability_pool_bold,
            CASE
                WHEN debt_bold > 0 AND collateral_{market_symbol.lower()} > 0
                THEN ROUND(collateral_{market_symbol.lower()} / debt_bold, 2)
                ELSE NULL
            END as collateral_ratio,
            CASE
                WHEN debt_bold > 0 AND stability_pool_bold > 0
                THEN ROUND((stability_pool_bold / debt_bold) * 100, 1)
                ELSE NULL
            END as sp_coverage_percent
        FROM filled_data
        WHERE date >= DATE('2024-07-12')
        ORDER BY date DESC
        LIMIT 30
        """
        return self.execute_dune_query(query, "Historical System Metrics", market_symbol)
    
    def get_trove_analysis(self, market_symbol: str) -> pd.DataFrame:
        """Get comprehensive Trove analysis data for specified market"""
        trove_table = self.get_table_name(market_symbol, 'trovemanager')
        
        query = f"""
        WITH ranked_troves AS (
            SELECT
                _troveId,
                _coll / 1e18 as collateral,
                _debt / 1e18 as debt,
                _annualInterestRate / 1e16 as interest_rate_percent,
                evt_block_time,
                ROW_NUMBER() OVER (PARTITION BY _troveId ORDER BY evt_block_time DESC) as rn
            FROM {trove_table}
            WHERE _coll > 0 AND _debt > 0
        ),
        latest_troves AS (
            SELECT
                _troveId,
                collateral,
                debt,
                interest_rate_percent,
                evt_block_time
            FROM ranked_troves
            WHERE rn = 1
        )
        SELECT
            _troveId as trove_id,
            ROUND(collateral, 2) as collateral_{market_symbol.lower()},
            ROUND(debt, 2) as debt_bold,
            ROUND((collateral / debt) * 100, 2) as collateral_ratio_percent,
            ROUND(interest_rate_percent, 2) as interest_rate_percent,
            RANK() OVER (ORDER BY collateral DESC) as size_rank,
            RANK() OVER (ORDER BY debt DESC) as debt_rank,
            RANK() OVER (ORDER BY interest_rate_percent ASC) as rate_rank,
            ROUND((collateral / SUM(collateral) OVER ()) * 100, 1) as pct_of_total_collateral,
            ROUND((debt / SUM(debt) OVER ()) * 100, 1) as pct_of_total_debt,
            ROUND(debt * (interest_rate_percent / 100), 2) as annual_interest_cost_bold
        FROM latest_troves
        ORDER BY collateral DESC
        LIMIT 100
        """
        return self.execute_dune_query(query, "Trove Analysis", market_symbol)
    
    def get_stability_pool_trends(self, market_symbol: str) -> pd.DataFrame:
        """Get stability pool balance trends with user activity for specified market"""
        sp_balance_table = self.get_table_name(market_symbol, 'stabilitypool_balance')
        sp_deposit_table = self.get_table_name(market_symbol, 'stabilitypool_deposit')
        
        query = f"""
        WITH sp_balance_changes AS (
            SELECT
                _newBalance / 1e18 as sp_balance_bold,
                evt_block_time,
                LAG(_newBalance / 1e18) OVER (ORDER BY evt_block_time) as prev_balance
            FROM {sp_balance_table}
        ),
        daily_sp_summary AS (
            SELECT
                date_trunc('day', evt_block_time) as date,
                LAST_VALUE(sp_balance_bold) OVER (
                    PARTITION BY date_trunc('day', evt_block_time)
                    ORDER BY evt_block_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as eod_sp_balance,
                COUNT(*) as balance_updates
            FROM sp_balance_changes
            GROUP BY date_trunc('day', evt_block_time), sp_balance_bold, evt_block_time
        ),
        sp_user_activity AS (
            SELECT
                date_trunc('day', evt_block_time) as date,
                COUNT(*) as operations_count,
                COUNT(DISTINCT _depositor) as unique_depositors,
                SUM(CASE WHEN _topUpOrWithdrawal > 0 THEN _topUpOrWithdrawal / 1e18 END) as daily_deposits,
                SUM(CASE WHEN _topUpOrWithdrawal < 0 THEN ABS(_topUpOrWithdrawal) / 1e18 END) as daily_withdrawals
            FROM {sp_deposit_table}
            GROUP BY date_trunc('day', evt_block_time)
        )
        SELECT DISTINCT
            ds.date,
            ROUND(ds.eod_sp_balance, 0) as sp_balance_bold,
            ROUND(ds.eod_sp_balance - LAG(ds.eod_sp_balance) OVER (ORDER BY ds.date), 0) as daily_change,
            ds.balance_updates,
            COALESCE(spa.operations_count, 0) as user_operations,
            COALESCE(spa.unique_depositors, 0) as unique_depositors,
            ROUND(COALESCE(spa.daily_deposits, 0), 0) as deposits_bold,
            ROUND(COALESCE(spa.daily_withdrawals, 0), 0) as withdrawals_bold
        FROM daily_sp_summary ds
        LEFT JOIN sp_user_activity spa ON ds.date = spa.date
        ORDER BY ds.date DESC
        LIMIT 30
        """
        return self.execute_dune_query(query, "Stability Pool Trends", market_symbol)
    
    def get_stability_pool_historical_apy(self, market_symbol: str) -> pd.DataFrame:
        """Get Stability Pool Historical APY for specified market"""
        config = self.get_market_config(market_symbol)
        sp_address = config['stability_pool_address']
        
        query = f"""
        -- {market_symbol} Stability Pool Historical APY (Complete Days)
        WITH interest_rewards AS (
            SELECT 
                date_trunc('day', evt_block_time) as day, 
                sum(value/1e18) as bold_amount 
            FROM nerite_arbitrum.boldtoken_evt_transfer
            WHERE contract_address = 0x4ecf61a6c2fab8a047ceb3b3b263b401763e9d49
            AND to = {sp_address}  -- {market_symbol} Stability Pool address
            AND "from" = 0x0000000000000000000000000000000000000000  -- Minted from zero address
            AND evt_block_time >= DATE '2024-07-12'
            GROUP BY 1 
        ),
        time_seq AS (
            SELECT sequence(
                CAST('2024-07-12' as timestamp),
                date_trunc('day', cast(now() as timestamp)),
                interval '1' day
            ) as time 
        ),
        days AS (
            SELECT time.time as day 
            FROM time_seq
            CROSS JOIN unnest(time) as time(time)
        ),
        -- Get {market_symbol} Stability Pool balances
        pool_balances AS (
            SELECT 
                date_trunc('day', evt_block_time) as day,
                AVG(_newBalance / 1e18) as bold_supply
            FROM {self.get_table_name(market_symbol, 'stabilitypool_balance')}
            WHERE evt_block_time >= DATE '2024-07-12'
            GROUP BY 1
        ),
        -- Forward-fill missing balances
        filled_balances AS (
            SELECT 
                d.day,
                COALESCE(
                    pb.bold_supply,
                    LAG(pb.bold_supply, 1) OVER (ORDER BY d.day),
                    LAG(pb.bold_supply, 2) OVER (ORDER BY d.day),
                    LAG(pb.bold_supply, 3) OVER (ORDER BY d.day),
                    LAG(pb.bold_supply, 4) OVER (ORDER BY d.day),
                    LAG(pb.bold_supply, 5) OVER (ORDER BY d.day)
                ) as bold_supply
            FROM days d
            LEFT JOIN pool_balances pb ON d.day = pb.day
        ),
        -- Complete data with all days
        complete_data AS (
            SELECT 
                d.day,
                COALESCE(ir.bold_amount, 0) as daily_rewards,
                fb.bold_supply
            FROM days d
            LEFT JOIN interest_rewards ir ON d.day = ir.day
            LEFT JOIN filled_balances fb ON d.day = fb.day
            WHERE d.day < current_date  -- Exclude today
            AND fb.bold_supply > 0  -- Only include days where we have balance data
        ),
        -- Calculate 3-day rolling averages
        rolling_averages AS (
            SELECT 
                day,
                daily_rewards,
                bold_supply,
                -- 3-day rolling sum of rewards
                SUM(daily_rewards) OVER (
                    ORDER BY day 
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as rewards_3d,
                -- 3-day rolling average of supply
                AVG(bold_supply) OVER (
                    ORDER BY day 
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as avg_supply_3d
            FROM complete_data
        )
        SELECT 
            day,
            daily_rewards,
            bold_supply,
            rewards_3d,
            avg_supply_3d,
            -- Period return (3-day)
            CASE 
                WHEN avg_supply_3d > 0 THEN rewards_3d / avg_supply_3d 
                ELSE 0 
            END as period_return,
            -- Annualized APY (converting 3-day period to annual)
            CASE 
                WHEN avg_supply_3d > 0 THEN 
                    ((rewards_3d / avg_supply_3d) / 3) * 365 * 100
                ELSE 0 
            END as apy_percent
        FROM rolling_averages
        WHERE day >= DATE '2024-07-14'  -- Start after we have 3 days of data
        ORDER BY day DESC
        """
        return self.execute_dune_query(query, "Stability Pool Historical APY", market_symbol)

def create_system_overview_dashboard(df: pd.DataFrame, prices: Dict[str, float], analytics: NeriteSuperAnalytics, market_symbol: str):
    """Create system overview dashboard for specified market"""
    if df.empty:
        st.error(f"No data available for {market_symbol} system overview")
        return
    
    config = analytics.get_market_config(market_symbol)
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate USD values
    market_price = prices['market_price']
    usnd_price = prices['usnd_price']
    
    collateral_col = f'collateral_{market_symbol.lower()}'
    df['collateral_usd'] = df[collateral_col] * market_price
    df['debt_usd'] = df['debt_bold'] * usnd_price
    df['sp_usd'] = df['stability_pool_bold'] * usnd_price
    
    # Recalculate collateral ratio using real USD values
    df['collateral_ratio_real'] = (df['collateral_usd'] / df['debt_usd']) * 100
    
    # Latest metrics
    latest = df.iloc[-1]
    current_tcr = latest['collateral_ratio_real']
    
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Collateral", 
            f"{latest[collateral_col]:.2f} {market_symbol}",
            delta=f"${latest['collateral_usd']:,.0f}"
        )
    
    with col2:
        st.metric(
            "Total Debt", 
            f"{latest['debt_bold']:,.0f} USND",
            delta=f"${latest['debt_usd']:,.0f}"
        )
    
    with col3:
        # TCR with color coding
        if current_tcr < config['scr']:
            tcr_color = "inverse"
        elif current_tcr < config['ccr']:
            tcr_color = "inverse"
        else:
            tcr_color = "normal"
        
        st.metric(
            "Total Collateral Ratio (TCR)", 
            f"{current_tcr:.1f}%",
            delta=None,
            delta_color=tcr_color
        )
    
    with col4:
        coverage_color = "normal" if latest['sp_coverage_percent'] >= 50 else "inverse"
        st.metric(
            "SP Coverage", 
            f"{latest['sp_coverage_percent']:.1f}%",
            delta=None,
            delta_color=coverage_color
        )
    
    with col5:
        st.metric(
            f"{market_symbol} Price", 
            f"${market_price:,.2f}"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # TVL Chart
        fig_tvl = go.Figure()
        
        fig_tvl.add_trace(go.Scatter(
            x=df['date'],
            y=df['collateral_usd'],
            mode='lines+markers',
            name=f'Collateral ({market_symbol} USD)',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig_tvl.add_trace(go.Scatter(
            x=df['date'],
            y=df['debt_usd'],
            mode='lines+markers',
            name='Debt (USND)',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        fig_tvl.add_trace(go.Scatter(
            x=df['date'],
            y=df['sp_usd'],
            mode='lines+markers',
            name='Stability Pool (USND)',
            line=dict(color='#2ca02c', width=3)
        ))
        
        fig_tvl.update_layout(
            title=f"Total Value Locked - {market_symbol} Market (USD)",
            xaxis_title="Date",
            yaxis_title="Value (USD)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_tvl, use_container_width=True)
    
    with col2:
        # Collateral Ratio Chart
        fig_ratio = go.Figure()
        
        fig_ratio.add_trace(go.Scatter(
            x=df['date'],
            y=df['collateral_ratio_real'],
            mode='lines+markers',
            name='Collateral Ratio (%)',
            line=dict(color='#2ca02c', width=3)
        ))
        
        # Add protocol threshold lines
        fig_ratio.add_hline(
            y=config['scr'], 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"SCR - Shutdown ({config['scr']}%)"
        )
        
        fig_ratio.add_hline(
            y=config['ccr'], 
            line_dash="dash", 
            line_color="orange", 
            annotation_text=f"CCR - Critical ({config['ccr']}%)"
        )
        
        fig_ratio.update_layout(
            title=f"System Collateralization Ratio - {market_symbol} (%)",
            xaxis_title="Date",
            yaxis_title="Ratio (%)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_ratio, use_container_width=True)
    
    # Stability Pool Coverage
    fig_sp = go.Figure()
    
    fig_sp.add_trace(go.Scatter(
        x=df['date'],
        y=df['sp_coverage_percent'],
        mode='lines+markers',
        name='SP Coverage %',
        line=dict(color='#9467bd', width=3),
        fill='tonexty'
    ))
    
    fig_sp.add_hline(
        y=50, 
        line_dash="dash", 
        line_color="orange", 
        annotation_text="Good Coverage (50%)"
    )
    
    fig_sp.add_hline(
        y=80, 
        line_dash="dash", 
        line_color="green", 
        annotation_text="Excellent Coverage (80%)"
    )
    
    fig_sp.update_layout(
        title=f"Stability Pool Coverage - {market_symbol} Market",
        xaxis_title="Date",
        yaxis_title="Coverage %",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_sp, use_container_width=True)

def create_trove_analysis_dashboard(df: pd.DataFrame, prices: Dict[str, float], analytics: NeriteSuperAnalytics, market_symbol: str):
    """Create Trove analysis dashboard for specified market"""
    if df.empty:
        st.error(f"No data available for {market_symbol} Trove analysis")
        return
    
    config = analytics.get_market_config(market_symbol)
    market_price = prices['market_price']
    usnd_price = prices['usnd_price']
    
    collateral_col = f'collateral_{market_symbol.lower()}'
    df['collateral_usd'] = df[collateral_col] * market_price
    df['debt_usd'] = df['debt_bold'] * usnd_price
    
    # Recalculate collateral ratio using real USD values and convert to percentage
    df['collateral_ratio_real'] = ((df[collateral_col] * market_price) / (df['debt_bold'] * usnd_price)) * 100
    
    # Calculate risk categories
    at_risk_troves = len(df[df['collateral_ratio_real'] < config['mcr']])
    healthy_troves = len(df[df['collateral_ratio_real'] >= config['ccr']])
    warning_troves = len(df[(df['collateral_ratio_real'] >= config['mcr']) & (df['collateral_ratio_real'] < config['ccr'])])
    
    # Summary metrics
    st.subheader(f"📊 {market_symbol} Trove Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Troves", len(df))
    
    with col2:
        st.metric("Healthy Troves", healthy_troves, delta=f"≥{config['ccr']}%")
    
    with col3:
        st.metric("Warning Troves", warning_troves, delta=f"{config['mcr']}-{config['ccr']}%")
    
    with col4:
        st.metric("At-Risk Troves", at_risk_troves, delta=f"<{config['mcr']}%", delta_color="inverse")
    
    with col5:
        st.metric("Avg ICR", f"{df['collateral_ratio_real'].mean():.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Troves by Collateral
        top_20 = df.head(20)
        fig_top = px.bar(
            top_20,
            x=range(len(top_20)),
            y='collateral_usd',
            title=f"Top 20 {market_symbol} Troves by Collateral (USD)",
            labels={'x': 'Trove Rank', 'y': 'Collateral (USD)'},
            color='collateral_ratio_real',
            color_continuous_scale='RdYlGn'
        )
        fig_top.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # Interest Rate Distribution
        fig_rate = px.histogram(
            df,
            x='interest_rate_percent',
            nbins=20,
            title=f"{market_symbol} Interest Rate Distribution",
            labels={'x': 'Interest Rate %', 'y': 'Number of Troves'}
        )
        fig_rate.add_vline(
            x=df['interest_rate_percent'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {df['interest_rate_percent'].mean():.2f}%"
        )
        fig_rate.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_rate, use_container_width=True)
    
    # Collateral Ratio Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cr = px.histogram(
            df,
            x='collateral_ratio_real',
            nbins=20,
            title=f"{market_symbol} Collateralization Ratio Distribution (%)",
            labels={'x': 'Collateral Ratio (%)', 'y': 'Number of Troves'}
        )
        fig_cr.add_vline(x=config['mcr'], line_dash="dash", line_color="red", annotation_text=f"MCR - Liquidation ({config['mcr']}%)")
        fig_cr.add_vline(x=config['ccr'], line_dash="dash", line_color="orange", annotation_text=f"CCR - Critical ({config['ccr']}%)")
        fig_cr.add_vline(
            x=df['collateral_ratio_real'].mean(),
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Mean: {df['collateral_ratio_real'].mean():.1f}%"
        )
        fig_cr.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_cr, use_container_width=True)
    
    with col2:
        # Debt vs Interest Rate Scatter
        fig_scatter = px.scatter(
            df,
            x='interest_rate_percent',
            y='debt_bold',
            size='collateral_usd',
            color='collateral_ratio_real',
            color_continuous_scale='RdYlGn',
            title=f"{market_symbol} Debt vs Interest Rate (Size = Collateral, Color = Ratio)",
            labels={'x': 'Interest Rate %', 'y': 'Debt (USND)'},
            hover_data=['trove_id', collateral_col]
        )
        fig_scatter.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Detailed Trove Table
    st.subheader(f"🏦 Detailed {market_symbol} Trove Data")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_collateral = st.slider(f"Min Collateral ({market_symbol})", 0.0, float(df[collateral_col].max()) if not df.empty else 25.0, 0.0, key=f"min_collateral_{market_symbol}")
    
    with col2:
        max_interest = st.slider("Max Interest Rate (%)", 0.0, float(df['interest_rate_percent'].max()) if not df.empty else 10.0, float(df['interest_rate_percent'].max()) if not df.empty else 10.0, key=f"max_interest_{market_symbol}")
    
    with col3:
        min_ratio = st.slider("Min Collateral Ratio (%)", 100.0, float(df['collateral_ratio_real'].max()) if not df.empty else 300.0, float(config['mcr']), key=f"min_ratio_{market_symbol}")
    
    # Filter data
    if not df.empty:
        filtered_df = df[
            (df[collateral_col] >= min_collateral) &
            (df['interest_rate_percent'] <= max_interest) &
            (df['collateral_ratio_real'] >= min_ratio)
        ]
        
        st.write(f"Filtered results: {len(filtered_df)} troves")
    else:
        filtered_df = df
    
    # Display table
    display_df = filtered_df[['trove_id', collateral_col, 'debt_bold', 'collateral_ratio_real', 'interest_rate_percent', 'annual_interest_cost_bold']].copy()
    display_df.columns = ['Trove ID', f'Collateral ({market_symbol})', 'Debt (USND)', 'Collateral Ratio (%)', 'Interest Rate (%)', 'Annual Interest (USND)']
    
    # Format the collateral ratio to show as percentage with 1 decimal place
    display_df['Collateral Ratio (%)'] = display_df['Collateral Ratio (%)'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Liquidation Price Analysis
    st.subheader(f"⚠️ {market_symbol} Liquidation Price Analysis")
    
    if not df.empty:
        # Calculate liquidation prices for each trove
        market_price = prices['market_price']
        usnd_price = prices['usnd_price']
        mcr = config['mcr']
        
        # Liquidation price = (debt * usnd_price * MCR) / (collateral * 100)
        df['liquidation_price'] = (df['debt_bold'] * usnd_price * mcr) / (df[collateral_col] * 100)
        
        # Calculate price drop percentage needed for liquidation
        df['price_drop_pct'] = ((market_price - df['liquidation_price']) / market_price) * 100
        
        # Filter out troves that are already at risk (negative price drop)
        safe_troves = df[df['price_drop_pct'] > 0].copy()
        
        if not safe_troves.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Liquidation Price Distribution
                fig_liq_dist = px.histogram(
                    safe_troves,
                    x='liquidation_price',
                    nbins=20,
                    title=f"{market_symbol} Liquidation Price Distribution",
                    labels={'x': f'Liquidation Price (USD)', 'y': 'Number of Troves'},
                    color_discrete_sequence=['#ff6b6b']
                )
                
                # Add current price line
                fig_liq_dist.add_vline(
                    x=market_price,
                    line_dash="solid",
                    line_color="green",
                    line_width=3,
                    annotation_text=f"Current Price: ${market_price:,.2f}"
                )
                
                # Add MCR threshold info
                fig_liq_dist.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=f"MCR Threshold: {mcr}%<br>Troves at Risk: {len(df[df['price_drop_pct'] <= 0])}",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1
                )
                
                fig_liq_dist.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_liq_dist, use_container_width=True)
            
            with col2:
                # Price Drop Analysis
                safe_troves_sorted = safe_troves.sort_values('price_drop_pct')
                
                # Create cumulative liquidations chart
                safe_troves_sorted['cumulative_debt'] = safe_troves_sorted['debt_bold'].cumsum()
                safe_troves_sorted['cumulative_collateral_usd'] = (safe_troves_sorted[collateral_col] * market_price).cumsum()
                
                fig_cascade = go.Figure()
                
                # Add cumulative debt line
                fig_cascade.add_trace(go.Scatter(
                    x=safe_troves_sorted['price_drop_pct'],
                    y=safe_troves_sorted['cumulative_debt'],
                    mode='lines',
                    name='Cumulative Debt at Risk (USND)',
                    line=dict(color='#ff7f0e', width=3),
                    yaxis='y1'
                ))
                
                # Add cumulative collateral line
                fig_cascade.add_trace(go.Scatter(
                    x=safe_troves_sorted['price_drop_pct'],
                    y=safe_troves_sorted['cumulative_collateral_usd'],
                    mode='lines',
                    name='Cumulative Collateral at Risk (USD)',
                    line=dict(color='#1f77b4', width=3),
                    yaxis='y2'
                ))
                
                fig_cascade.update_layout(
                    title=f"{market_symbol} Liquidation Cascade Analysis",
                    xaxis_title="Price Drop Required (%)",
                    yaxis=dict(title="Cumulative Debt (USND)", side="left", color="#ff7f0e"),
                    yaxis2=dict(title="Cumulative Collateral (USD)", side="right", overlaying="y", color="#1f77b4"),
                    height=400,
                    template="plotly_white",
                    legend=dict(x=0.02, y=0.98)
                )
                
                st.plotly_chart(fig_cascade, use_container_width=True)
            
            # Liquidation Risk Summary
            st.markdown("### 📊 Liquidation Risk Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate risk buckets
            immediate_risk = len(df[df['price_drop_pct'] <= 0])  # Already at risk
            high_risk = len(df[(df['price_drop_pct'] > 0) & (df['price_drop_pct'] <= 10)])  # <10% drop
            medium_risk = len(df[(df['price_drop_pct'] > 10) & (df['price_drop_pct'] <= 25)])  # 10-25% drop
            low_risk = len(df[df['price_drop_pct'] > 25])  # >25% drop
            
            with col1:
                st.metric("🔴 Immediate Risk", immediate_risk, delta="Already at MCR")
            
            with col2:
                st.metric("🟠 High Risk", high_risk, delta="<10% price drop")
            
            with col3:
                st.metric("🟡 Medium Risk", medium_risk, delta="10-25% price drop")
            
            with col4:
                st.metric("🟢 Low Risk", low_risk, delta=">25% price drop")
            
            # Top 10 most at-risk troves
            st.markdown("### ⚠️ Most At-Risk Troves")
            
            most_at_risk = df.nsmallest(10, 'price_drop_pct')[
                ['trove_id', collateral_col, 'debt_bold', 'collateral_ratio_real', 'liquidation_price', 'price_drop_pct']
            ].copy()
            
            # Format display
            most_at_risk.columns = [
                'Trove ID', f'Collateral ({market_symbol})', 'Debt (USND)', 
                'Current Ratio (%)', 'Liquidation Price ($)', 'Price Drop Needed (%)'
            ]
            
            # Format columns
            most_at_risk['Current Ratio (%)'] = most_at_risk['Current Ratio (%)'].apply(lambda x: f"{x:.1f}%")
            most_at_risk['Liquidation Price ($)'] = most_at_risk['Liquidation Price ($)'].apply(lambda x: f"${x:,.2f}")
            most_at_risk['Price Drop Needed (%)'] = most_at_risk['Price Drop Needed (%)'].apply(
                lambda x: f"{x:.1f}%" if x > 0 else "⚠️ AT RISK"
            )
            
            st.dataframe(most_at_risk, use_container_width=True)
            
        else:
            st.warning("All troves are currently at liquidation risk or no safe troves found.")
    else:
        st.info("No trove data available for liquidation analysis.")

def create_stability_pool_dashboard(df: pd.DataFrame, apy_df: pd.DataFrame, market_symbol: str):
    """Create Stability Pool dashboard for specified market"""
    if df.empty:
        st.error(f"No data available for {market_symbol} Stability Pool analysis")
        return
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Latest metrics
    latest = df.iloc[-1]
    
    # Calculate current APY if available
    current_apy = None
    if not apy_df.empty:
        apy_df['day'] = pd.to_datetime(apy_df['day'])
        apy_df = apy_df.sort_values('day')
        current_apy = apy_df.iloc[-1]['apy_percent'] if len(apy_df) > 0 else None
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Balance", f"{latest['sp_balance_bold']:,.0f} USND")
    
    with col2:
        st.metric("Daily Change", f"{latest['daily_change']:,.0f} USND")
    
    with col3:
        st.metric("Active Depositors", latest['unique_depositors'])
    
    with col4:
        if current_apy is not None:
            st.metric("Current APY", f"{current_apy:.2f}%")
        else:
            st.metric("Daily Operations", latest['user_operations'])
    
    # APY Chart (full width if available)
    if not apy_df.empty:
        st.subheader(f"📈 {market_symbol} Historical APY")
        
        fig_apy = go.Figure()
        
        fig_apy.add_trace(go.Scatter(
            x=apy_df['day'],
            y=apy_df['apy_percent'],
            mode='lines+markers',
            name='APY %',
            line=dict(color='#ff6b6b', width=3),
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        # Add average line
        avg_apy = apy_df['apy_percent'].mean()
        fig_apy.add_hline(
            y=avg_apy,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Average: {avg_apy:.2f}%"
        )
        
        fig_apy.update_layout(
            title=f"{market_symbol} Stability Pool Historical APY (3-day rolling average)",
            xaxis_title="Date",
            yaxis_title="APY (%)",
            height=400,
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig_apy, use_container_width=True)
        
        # APY Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average APY", f"{avg_apy:.2f}%")
        
        with col2:
            st.metric("Max APY", f"{apy_df['apy_percent'].max():.2f}%")
        
        with col3:
            st.metric("Min APY", f"{apy_df['apy_percent'].min():.2f}%")
        
        with col4:
            st.metric("APY Volatility", f"{apy_df['apy_percent'].std():.2f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Balance Over Time
        fig_balance = go.Figure()
        
        fig_balance.add_trace(go.Scatter(
            x=df['date'],
            y=df['sp_balance_bold'],
            mode='lines+markers',
            name='SP Balance',
            line=dict(color='#1f77b4', width=3),
            fill='tonexty'
        ))
        
        fig_balance.update_layout(
            title=f"{market_symbol} Stability Pool Balance Over Time",
            xaxis_title="Date",
            yaxis_title="Balance (USND)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_balance, use_container_width=True)
    
    with col2:
        # Daily Changes
        colors = ['green' if x >= 0 else 'red' for x in df['daily_change']]
        
        fig_changes = go.Figure()
        
        fig_changes.add_trace(go.Bar(
            x=df['date'],
            y=df['daily_change'],
            marker_color=colors,
            name='Daily Change'
        ))
        
        fig_changes.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        fig_changes.update_layout(
            title=f"{market_symbol} Daily Balance Changes",
            xaxis_title="Date",
            yaxis_title="Change (USND)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_changes, use_container_width=True)
    
    # User Activity and Deposits/Withdrawals
    col1, col2 = st.columns(2)
    
    with col1:
        # User Activity
        fig_users = go.Figure()
        
        fig_users.add_trace(go.Scatter(
            x=df['date'],
            y=df['unique_depositors'],
            mode='lines+markers',
            name='Unique Depositors',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        fig_users.add_trace(go.Scatter(
            x=df['date'],
            y=df['user_operations'],
            mode='lines+markers',
            name='Daily Operations',
            line=dict(color='#2ca02c', width=3),
            yaxis='y2'
        ))
        
        fig_users.update_layout(
            title=f"{market_symbol} User Activity",
            xaxis_title="Date",
            yaxis_title="Unique Depositors",
            yaxis2=dict(title="Daily Operations", overlaying='y', side='right'),
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_users, use_container_width=True)
    
    with col2:
        # Deposits vs Withdrawals
        fig_flows = go.Figure()
        
        fig_flows.add_trace(go.Bar(
            x=df['date'],
            y=df['deposits_bold'],
            name='Deposits',
            marker_color='green',
            opacity=0.7
        ))
        
        fig_flows.add_trace(go.Bar(
            x=df['date'],
            y=df['withdrawals_bold'],
            name='Withdrawals',
            marker_color='red',
            opacity=0.7
        ))
        
        fig_flows.update_layout(
            title=f"{market_symbol} Daily Deposits vs Withdrawals",
            xaxis_title="Date",
            yaxis_title="Amount (USND)",
            barmode='group',
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_flows, use_container_width=True)
    
    # Show detailed APY data table if available
    if not apy_df.empty:
        st.subheader(f"📊 {market_symbol} Detailed APY Data")
        
        # Show last 15 days
        recent_apy = apy_df.head(15).copy()
        recent_apy['day'] = recent_apy['day'].dt.strftime('%Y-%m-%d')
        
        display_cols = ['day', 'daily_rewards', 'bold_supply', 'rewards_3d', 'period_return', 'apy_percent']
        display_names = ['Date', 'Daily Rewards (USND)', 'Pool Balance (USND)', '3-Day Rewards (USND)', '3-Day Return', 'APY (%)']
        
        display_apy_df = recent_apy[display_cols].copy()
        display_apy_df.columns = display_names
        
        # Format numbers
        display_apy_df['Daily Rewards (USND)'] = display_apy_df['Daily Rewards (USND)'].apply(lambda x: f"{x:,.2f}")
        display_apy_df['Pool Balance (USND)'] = display_apy_df['Pool Balance (USND)'].apply(lambda x: f"{x:,.0f}")
        display_apy_df['3-Day Rewards (USND)'] = display_apy_df['3-Day Rewards (USND)'].apply(lambda x: f"{x:,.2f}")
        display_apy_df['3-Day Return'] = display_apy_df['3-Day Return'].apply(lambda x: f"{x:.6f}")
        display_apy_df['APY (%)'] = display_apy_df['APY (%)'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_apy_df, use_container_width=True, height=400)

def render_market_parameters_info(config: Dict, market_symbol: str):
    """Render protocol parameters information for the selected market"""
    st.divider()
    
    st.subheader(f"📋 {config['display_name']} Market Parameters & Risk Levels")
    
    st.markdown("### **Individual Trove Parameters**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **MCR (Minimum Collateral Ratio): {config['mcr']}%**
        - Individual trove liquidation threshold for {market_symbol} market
        - If a trove's ICR (Individual Collateral Ratio) falls below MCR, it becomes eligible for liquidation
        - Protects the protocol from under-collateralized debt
        """)
    
    with col2:
        st.markdown(f"""
        **LTV (Loan-to-Value): {config['ltv']}%**
        - Maximum borrowing capacity relative to collateral value
        - Determines the maximum debt a user can take against their {market_symbol} collateral
        """)
    
    st.markdown("### **System-Wide Market Parameters**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **CCR (Critical Collateral Ratio): {config['ccr']}%**
        - System-wide risk threshold for the entire {market_symbol} market
        - When the branch's TCR (Total Collateral Ratio) falls below CCR:
          - ⛔ New borrowing is restricted unless it brings TCR back above CCR
          - ⚠️ Collateral withdrawals only allowed with equal value debt repayment
          - 🚫 Troves can't be adjusted in ways that would reduce the TCR
        """)
    
    with col2:
        st.markdown(f"""
        **SCR (Shutdown Collateral Ratio): {config['scr']}%**
        - Emergency shutdown threshold for the entire {market_symbol} market
        - If the branch's TCR falls below SCR:
          - 🛑 Protocol triggers shutdown of the borrow market
          - ❌ All borrowing operations are permanently disabled
          - ✅ Only closing troves is allowed
        """)
    
    st.markdown("### **Market Limits & Information**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Initial Debt Limit: ${config['initial_debt_limit']:,}**
        - Maximum total debt that can be issued in the {market_symbol} market
        - Prevents unlimited expansion and maintains system stability
        
        **Token Type: {config['token_tag']}**
        - Market classification and risk category
        """)
    
    with col2:
        st.markdown(f"""
        **Risk Level Interpretation for {market_symbol}:**
        - 🟢 **Healthy**: TCR ≥ {config['ccr']}% - Normal operations
        - 🟡 **Critical**: {config['scr']}% ≤ TCR < {config['ccr']}% - Limited operations
        - 🔴 **Shutdown**: TCR < {config['scr']}% - Emergency mode
        
        **Stability Pool Address:**
        - `{config['stability_pool_address']}`
        """)

def initialize_session_state():
    """Initialize session state variables to prevent app crashes when switching markets"""
    if 'current_market' not in st.session_state:
        st.session_state.current_market = None  # No default market - wait for user selection
    
    if 'market_selected' not in st.session_state:
        st.session_state.market_selected = False  # Track if user has made a selection
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'market_data_cache' not in st.session_state:
        st.session_state.market_data_cache = {}
    
    if 'analytics_instance' not in st.session_state:
        st.session_state.analytics_instance = None

def clear_market_cache(market_symbol: str):
    """Clear cached data for a specific market"""
    if 'market_data_cache' in st.session_state:
        if market_symbol in st.session_state.market_data_cache:
            del st.session_state.market_data_cache[market_symbol]

def safe_market_switch(new_market: str, analytics: NeriteSuperAnalytics):
    """Safely switch markets with proper state management"""
    try:
        if st.session_state.current_market != new_market:
            # Clear cache for better data integrity
            st.cache_data.clear()
            
            # Update session state
            st.session_state.current_market = new_market
            st.session_state.market_selected = True  # Mark that user has selected a market
            
            # Set the analytics instance market
            analytics.set_current_market(new_market)
            
            # Force a rerun to update the interface
            st.rerun()
            
    except Exception as e:
        st.error(f"Error switching to {new_market} market: {e}")
        st.info("Please try refreshing the page if the issue persists.")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🏦 Nerite Protocol Analytics Dashboard")
    st.markdown("Comprehensive multi-market analytics for the Nerite Protocol (Liquity v3 fork) on Arbitrum")
    
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
    
    # Initialize or get analytics instance
    if st.session_state.analytics_instance is None or st.session_state.analytics_instance.dune_api_key != dune_api_key:
        st.session_state.analytics_instance = NeriteSuperAnalytics(dune_api_key)
    
    analytics = st.session_state.analytics_instance
    
    # Market Selection
    st.sidebar.header("🎯 Market Selection")
    
    # Create market options with display names
    market_options = {config['display_name']: symbol for symbol, config in analytics.MARKETS.items()}
    
    # Add a placeholder option
    market_display_options = ["-- Select a Market --"] + list(market_options.keys())
    
    # Determine the current index
    if st.session_state.current_market and st.session_state.market_selected:
        current_config = analytics.get_market_config(st.session_state.current_market)
        current_index = market_display_options.index(current_config['display_name'])
    else:
        current_index = 0  # Default to placeholder
    
    selected_market_display = st.sidebar.selectbox(
        "Select Market to Analyze",
        options=market_display_options,
        index=current_index,
        key="market_selector"
    )
    
    # Handle market selection
    if selected_market_display == "-- Select a Market --":
        # Show welcome message and instructions
        st.markdown("""
        ## 🏦 Welcome to Nerite Protocol Analytics
        
        **Comprehensive multi-market analytics for the Nerite Protocol (Liquity v3 fork) on Arbitrum**
        
        ### 📊 Available Markets:
        """)
        
        # Display available markets in a nice format
        cols = st.columns(3)
        market_info = [
            ("WETH", "🔷", "Ethereum", "ETH"),
            ("wstETH", "🔵", "Lido Staked ETH", "LST"),
            ("rETH", "🟠", "Rocket Pool ETH", "LST"),
            ("rsETH", "🟡", "Kelp Restaked ETH", "LRT"),
            ("weETH", "🟢", "Ether.fi Wrapped ETH", "LRT"),
            ("ARB", "🔴", "Arbitrum Token", "ARB"),
            ("COMP", "🟣", "Compound Token", "COMP"),
            ("tBTC", "🟨", "Threshold Bitcoin", "BTC")
        ]
        
        for i, (symbol, emoji, name, token_type) in enumerate(market_info):
            config = analytics.MARKETS[symbol]
            with cols[i % 3]:
                st.markdown(f"""
                **{emoji} {config['display_name']}**
                - {name} ({token_type})
                - MCR: {config['mcr']}% | CCR: {config['ccr']}%
                - Max Debt: ${config['initial_debt_limit']:,}
                """)
        
        st.markdown("""
        ### 🚀 Getting Started:
        1. **Select a market** from the dropdown in the sidebar
        2. **Explore the data** across three main sections:
           - 📈 **System Overview**: Total collateral, debt, and system health
           - 🏦 **Trove Analysis**: Individual borrowing positions and risk assessment  
           - 🛡️ **Stability Pool**: Depositor activity and yield analysis
        3. **Switch markets** anytime to compare different assets
        
        ### ⚙️ Configuration:
        - Set your **Dune API key** in the sidebar (pre-filled with demo key)
        - Enable **auto-refresh** for real-time data updates
        - Use **refresh controls** to manually update data
        
        **👈 Choose a market from the sidebar to begin analysis!**
        """)
        
        return  # Exit early, don't load any data
    
    # User has selected a market
    selected_market = market_options[selected_market_display]
    
    # Handle market switch
    if selected_market != st.session_state.current_market or not st.session_state.market_selected:
        safe_market_switch(selected_market, analytics)
        return  # Exit early to allow rerun
    
    # Only proceed if a market is selected
    if not st.session_state.market_selected or not st.session_state.current_market:
        return
    
    # Set current market in analytics
    analytics.set_current_market(st.session_state.current_market)
    current_config = analytics.get_market_config(st.session_state.current_market)
    
    # Display current market info
    st.sidebar.success(f"📊 Analyzing: **{current_config['display_name']}** Market")
    st.sidebar.info(f"Token Type: {current_config['token_tag']}")
    
    # Auto-refresh controls
    st.sidebar.header("🔄 Data Refresh")
    auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False, key="sidebar_auto_refresh")
    refresh_interval = st.sidebar.selectbox("Refresh interval (minutes)", [1, 5, 10, 15], index=1, key="sidebar_refresh_interval")
    
    if st.sidebar.button("🔄 Refresh Data") or auto_refresh:
        st.cache_data.clear()
        clear_market_cache(st.session_state.current_market)
    
    # Fetch prices
    with st.spinner(f"Fetching current prices for {st.session_state.current_market}..."):
        prices = analytics.get_coingecko_prices(st.session_state.current_market)
    
    # Price display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{st.session_state.current_market} Price", f"${prices['market_price']:,.2f}")
    with col2:
        st.metric("USND Price", f"${prices['usnd_price']:.4f}")
    with col3:
        st.metric("Market Cap Limit", f"${current_config['initial_debt_limit']:,}")
    with col4:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    st.divider()
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📈 System Overview", "🏦 Trove Analysis", "🛡️ Stability Pool"])
    
    with tab1:
        st.header(f"{current_config['display_name']} System Overview")
        try:
            system_df = analytics.get_historical_system_metrics(st.session_state.current_market)
            if not system_df.empty:
                create_system_overview_dashboard(system_df, prices, analytics, st.session_state.current_market)
            else:
                st.warning(f"No system data available for {st.session_state.current_market} market. This might be a new market or there could be data issues.")
        except Exception as e:
            st.error(f"Error loading system overview for {st.session_state.current_market}: {e}")
    
    with tab2:
        st.header(f"{current_config['display_name']} Trove Analysis")
        try:
            trove_df = analytics.get_trove_analysis(st.session_state.current_market)
            if not trove_df.empty:
                create_trove_analysis_dashboard(trove_df, prices, analytics, st.session_state.current_market)
            else:
                st.warning(f"No Trove data available for {st.session_state.current_market} market. This might be a new market or there could be no active troves.")
        except Exception as e:
            st.error(f"Error loading trove analysis for {st.session_state.current_market}: {e}")
    
    with tab3:
        st.header(f"{current_config['display_name']} Stability Pool Analysis")
        try:
            sp_df = analytics.get_stability_pool_trends(st.session_state.current_market)
            apy_df = analytics.get_stability_pool_historical_apy(st.session_state.current_market)
            if not sp_df.empty:
                create_stability_pool_dashboard(sp_df, apy_df, st.session_state.current_market)
            else:
                st.warning(f"No Stability Pool data available for {st.session_state.current_market} market. This might be a new market or there could be no stability pool activity.")
        except Exception as e:
            st.error(f"Error loading stability pool analysis for {st.session_state.current_market}: {e}")
    
    # Market Parameters Information
    render_market_parameters_info(current_config, st.session_state.current_market)
    
    # Footer
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Sources:**")
        st.markdown("- Dune Analytics")
        st.markdown("- CoinGecko API")
    
    with col2:
        st.markdown("**Protocol Info:**")
        st.markdown("- Network: Arbitrum")
        st.markdown("- Protocol: Nerite (Liquity v3)")
    
    with col3:
        st.markdown("**Supported Markets:**")
        market_list = ", ".join([config['display_name'] for config in analytics.MARKETS.values()])
        st.markdown(f"- {market_list}")

def handle_auto_refresh():
    """Handle auto-refresh functionality"""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Check if auto-refresh is enabled and enough time has passed
    auto_refresh_main = st.sidebar.checkbox("Auto-refresh data", value=False, key="main_auto_refresh")
    if auto_refresh_main:
        refresh_interval_main = st.sidebar.selectbox("Refresh interval (minutes)", [1, 5, 10, 15], index=1, key="main_refresh_interval")
        
        if datetime.now() - st.session_state.last_refresh > timedelta(minutes=refresh_interval_main):
            st.cache_data.clear()
            if 'market_data_cache' in st.session_state:
                st.session_state.market_data_cache.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()

if __name__ == "__main__":
    try:
        # Handle auto-refresh
        handle_auto_refresh()
        
        # Run main application
        main()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page to restart the application.")
        
        # Reset session state on major errors
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()