import os
import time
import hmac
import hashlib
import random
import socket
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
from datetime import datetime, timedelta
import json

# Disable SSL warnings for bypass connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Load API credentials
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
RECV_WINDOW = 5000

# Multiple fallback methods for accessing Bybit API
class BybitAPIWorkaround:
    def __init__(self):
        # Known Bybit IP addresses (may change over time)
        self.bybit_ips = [
            "104.18.10.26",
            "104.18.11.26", 
            "172.67.74.226",
            "104.18.8.26",
            "104.18.9.26"
        ]
        
        # Alternative API endpoints
        self.alternative_endpoints = [
            "https://api.bybit.com",
            "https://api.bytick.com",  # Alternative domain
        ]
        
        self.working_endpoint = None
        self.session = None
        
    def create_dns_bypass_session(self, target_ip):
        """Create a session that bypasses DNS by using direct IP"""
        session = requests.Session()
        
        # Custom adapter for DNS bypass
        class DNSBypassAdapter(HTTPAdapter):
            def __init__(self, target_ip, target_host="api.bybit.com"):
                self.target_ip = target_ip
                self.target_host = target_host
                super().__init__()
            
            def send(self, request, **kwargs):
                # Replace hostname with IP in URL
                if self.target_host in request.url:
                    request.url = request.url.replace(self.target_host, self.target_ip)
                    # Critical: Set Host header for proper routing
                    request.headers['Host'] = self.target_host
                
                # Disable SSL verification for IP connections
                kwargs['verify'] = False
                
                return super().send(request, **kwargs)
        
        # Mount the custom adapter
        adapter = DNSBypassAdapter(target_ip)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        
        # Set reasonable timeouts and retries
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        
        return session
    
    def test_endpoint(self, endpoint_or_ip, is_ip=False):
        """Test if an endpoint is accessible"""
        try:
            if is_ip:
                session = self.create_dns_bypass_session(endpoint_or_ip)
                test_url = f"https://{endpoint_or_ip}/v5/market/time"
            else:
                session = requests.Session()
                test_url = f"{endpoint_or_ip}/v5/market/time"
            
            response = session.get(test_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0:
                    return True, session
        except Exception as e:
            st.write(f"Failed to connect to {endpoint_or_ip}: {str(e)[:100]}...")
        
        return False, None
    
    def find_working_connection(self):
        """Find a working connection method"""
        st.info("🔍 Testing connection methods...")
        
        # Method 1: Try standard endpoints first
        for endpoint in self.alternative_endpoints:
            st.write(f"Testing endpoint: {endpoint}")
            success, session = self.test_endpoint(endpoint)
            if success:
                st.success(f"✅ Connected via: {endpoint}")
                self.working_endpoint = endpoint
                self.session = session
                return True
        
        # Method 2: Try direct IP connections
        st.write("Standard endpoints failed. Trying direct IP connections...")
        for ip in self.bybit_ips:
            st.write(f"Testing IP: {ip}")
            success, session = self.test_endpoint(ip, is_ip=True)
            if success:
                st.success(f"✅ Connected via IP: {ip}")
                self.working_endpoint = f"https://{ip}"
                self.session = session
                return True
        
        return False
    
    def make_request(self, endpoint, params=None, headers=None):
        """Make an API request using the working connection"""
        if not self.session or not self.working_endpoint:
            if not self.find_working_connection():
                raise Exception("No working connection found")
        
        if headers is None:
            headers = {}
        
        # Construct full URL
        if self.working_endpoint.startswith("https://"):
            url = f"{self.working_endpoint}{endpoint}"
        else:
            url = f"https://{self.working_endpoint}{endpoint}"
        
        # Make request
        response = self.session.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()

# Global API client
api_client = BybitAPIWorkaround()

def sign_request(endpoint: str, params: dict = None):
    """Sign and execute API request with DNS workaround"""
    if params is None:
        params = {}
    
    timestamp = str(int(time.time() * 1000))
    query = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    sign_payload = f"{timestamp}{API_KEY}{RECV_WINDOW}{query}"
    signature = hmac.new(API_SECRET.encode(), sign_payload.encode(), hashlib.sha256).hexdigest()

    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": str(RECV_WINDOW),
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }

    try:
        data = api_client.make_request(endpoint, params=params, headers=headers)
        
        # Check API response status
        if data.get("retCode") != 0:
            error_msg = data.get("retMsg", "Unknown API error")
            st.error(f"❌ API Error: {error_msg}")
            return {}
            
        return data
    except Exception as e:
        st.error(f"❌ API call failed: {endpoint} - {str(e)}")
        return {}

@st.cache_data(ttl=60)
def fetch_tickers():
    """Fetch ticker data with all metrics"""
    data = sign_request("/v5/market/tickers", {"category": "linear"})
    items = data.get("result", {}).get("list", [])
    if not items:
        return pd.DataFrame()
    
    df = pd.DataFrame(items)
    
    # Convert numeric columns
    numeric_cols = [
        "price24hPcnt", "volume24h", "turnover24h", "fundingRate", 
        "openInterest", "openInterestValue", "lastPrice", "markPrice",
        "indexPrice", "basis", "basisRate"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def fetch_historical_funding_rates(symbols, days_back=30):
    """Fetch historical funding rates for averaging"""
    all_funding_data = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate timestamps for different periods
    now = int(time.time() * 1000)
    one_day_ago = now - (24 * 60 * 60 * 1000)
    three_days_ago = now - (3 * 24 * 60 * 60 * 1000)
    seven_days_ago = now - (7 * 24 * 60 * 60 * 1000)
    thirty_days_ago = now - (30 * 24 * 60 * 60 * 1000)
    
    for i, symbol in enumerate(symbols):
        status_text.text(f"Fetching funding history: {symbol} ({i+1}/{len(symbols)})")
        progress_bar.progress((i + 1) / len(symbols))
        
        try:
            # Fetch funding rate history (last 200 records)
            data = sign_request("/v5/market/funding/history", {
                "category": "linear",
                "symbol": symbol,
                "limit": 200
            })
            
            funding_list = data.get("result", {}).get("list", [])
            
            if funding_list:
                # Convert to DataFrame for easier calculation
                funding_df = pd.DataFrame(funding_list)
                funding_df['fundingRateTimestamp'] = pd.to_numeric(funding_df['fundingRateTimestamp'])
                funding_df['fundingRate'] = pd.to_numeric(funding_df['fundingRate'])
                
                # Calculate averages for different periods
                funding_24h = funding_df[funding_df['fundingRateTimestamp'] >= one_day_ago]['fundingRate'].mean()
                funding_3d = funding_df[funding_df['fundingRateTimestamp'] >= three_days_ago]['fundingRate'].mean()
                funding_7d = funding_df[funding_df['fundingRateTimestamp'] >= seven_days_ago]['fundingRate'].mean()
                funding_30d = funding_df[funding_df['fundingRateTimestamp'] >= thirty_days_ago]['fundingRate'].mean()
                
                all_funding_data[symbol] = {
                    'funding_24h_avg': funding_24h if not pd.isna(funding_24h) else 0,
                    'funding_3d_avg': funding_3d if not pd.isna(funding_3d) else 0,
                    'funding_7d_avg': funding_7d if not pd.isna(funding_7d) else 0,
                    'funding_30d_avg': funding_30d if not pd.isna(funding_30d) else 0,
                }
            else:
                all_funding_data[symbol] = {
                    'funding_24h_avg': 0,
                    'funding_3d_avg': 0,
                    'funding_7d_avg': 0,
                    'funding_30d_avg': 0,
                }
        except Exception as e:
            all_funding_data[symbol] = {
                'funding_24h_avg': 0,
                'funding_3d_avg': 0,
                'funding_7d_avg': 0,
                'funding_30d_avg': 0,
            }
        
        # Rate limiting
        time.sleep(0.5 + random.uniform(0.1, 0.3))
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame.from_dict(all_funding_data, orient='index').reset_index().rename(columns={'index': 'symbol'})

@st.cache_data(ttl=300)
def fetch_instruments():
    """Fetch instruments data with improved leverage extraction"""
    data = sign_request("/v5/market/instruments-info", {"category": "linear"})
    items = data.get("result", {}).get("list", [])
    if not items:
        return pd.DataFrame()
    
    df = pd.DataFrame(items)
    
    # Debug: Show what columns are available
    st.write(f"🔍 Instruments columns: {list(df.columns)}")
    
    # Show sample data structure
    if len(df) > 0:
        st.write("📋 Sample instrument data:")
        st.json(items[0])
    
    return df

def extract_leverage_info(instruments_df):
    """Extract leverage information from instruments data"""
    leverage_data = []
    
    if instruments_df.empty:
        return pd.DataFrame(columns=['symbol', 'maxLeverage'])
    
    for _, row in instruments_df.iterrows():
        symbol = row.get('symbol', '')
        max_leverage = None
        
        # Extract from leverageFilter (which is already a dict)
        if 'leverageFilter' in row and pd.notnull(row['leverageFilter']):
            try:
                leverage_filter = row['leverageFilter']
                
                # leverageFilter is already a dictionary
                if isinstance(leverage_filter, dict):
                    max_leverage_str = leverage_filter.get('maxLeverage', '0')
                    max_leverage = float(max_leverage_str)
                elif isinstance(leverage_filter, str):
                    # Fallback: try to parse as JSON string
                    leverage_data_parsed = json.loads(leverage_filter)
                    max_leverage = float(leverage_data_parsed.get('maxLeverage', 0))
                    
            except Exception as e:
                if symbol == '1000000BABYDOGEUSDT':  # Debug first symbol only
                    st.write(f"Error parsing leverageFilter for {symbol}: {e}")
                    st.write(f"leverageFilter type: {type(row['leverageFilter'])}")
                    st.write(f"leverageFilter content: {row['leverageFilter']}")
        
        leverage_data.append({
            'symbol': symbol,
            'maxLeverage': max_leverage if max_leverage and max_leverage > 0 else None
        })
    
    return pd.DataFrame(leverage_data)

def calculate_enhanced_metrics(df, funding_history_df):
    """Calculate all the enhanced metrics shown in the image"""
    
    # Basic calculations
    df["priceChange24h"] = df["price24hPcnt"] * 100
    df["currentFundingAPR"] = df["fundingRate"] * 3 * 365 * 100
    
    # Merge with funding history
    if not funding_history_df.empty:
        df = pd.merge(df, funding_history_df, on="symbol", how="left")
        
        # Calculate APR for different periods
        df["funding24hAPR"] = df["funding_24h_avg"] * 3 * 365 * 100
        df["funding3dAPR"] = df["funding_3d_avg"] * 3 * 365 * 100
        df["funding7dAPR"] = df["funding_7d_avg"] * 3 * 365 * 100
        df["funding30dAPR"] = df["funding_30d_avg"] * 3 * 365 * 100
    else:
        df["funding24hAPR"] = df["currentFundingAPR"]
        df["funding3dAPR"] = df["currentFundingAPR"]
        df["funding7dAPR"] = df["currentFundingAPR"]
        df["funding30dAPR"] = df["currentFundingAPR"]
    
    # Calculate basis - difference between mark price and index price
    if "markPrice" in df.columns and "indexPrice" in df.columns:
        df["basis"] = ((pd.to_numeric(df["markPrice"], errors='coerce') - 
                       pd.to_numeric(df["indexPrice"], errors='coerce')) / 
                       pd.to_numeric(df["indexPrice"], errors='coerce')) * 100
    else:
        df["basis"] = 0
    
    # Calculate spread - multiple methods to get bid-ask spread or premium
    df["spread"] = 0.0  # Default value as float
    
    # Method 1: Use basisRate if available and not null
    if "basisRate" in df.columns:
        basis_rate_numeric = pd.to_numeric(df["basisRate"], errors='coerce')
        valid_basis = basis_rate_numeric.notna() & (basis_rate_numeric != 0)
        df.loc[valid_basis, "spread"] = (basis_rate_numeric[valid_basis] * 100).astype(float)
    
    # Method 2: Calculate from bid1Price and ask1Price if basisRate is not available
    if "bid1Price" in df.columns and "ask1Price" in df.columns:
        bid_price = pd.to_numeric(df["bid1Price"], errors='coerce')
        ask_price = pd.to_numeric(df["ask1Price"], errors='coerce')
        
        # Only calculate where we don't already have spread from basisRate
        mask = (df["spread"] == 0) & bid_price.notna() & ask_price.notna() & (bid_price > 0) & (ask_price > 0)
        
        if mask.any():
            mid_price = (bid_price + ask_price) / 2
            calculated_spread = ((ask_price - bid_price) / mid_price) * 100
            df.loc[mask, "spread"] = calculated_spread[mask].astype(float)
    
    # Method 3: Use basis if no other spread is available
    if "basis" in df.columns:
        basis_numeric = pd.to_numeric(df["basis"], errors='coerce')
        mask = (df["spread"] == 0) & basis_numeric.notna() & (basis_numeric != 0)
        df.loc[mask, "spread"] = abs(basis_numeric[mask]).astype(float)  # Use absolute value of basis
    
    # Format volume in millions
    df["volume24hFormatted"] = df["volume24h"] / 1e6
    
    return df

def build_enhanced_dashboard():
    """Enhanced dashboard with all features from the image"""
    st.set_page_config(
        page_title="Enhanced Bybit Dashboard", 
        layout="wide"
    )
    
    st.title("📊 Enhanced Bybit Dashboard")
    st.markdown("*Complete trading analysis with historical funding rates*")
    
    # Check credentials
    if not API_KEY or not API_SECRET:
        st.error("🚫 Missing API credentials.")
        st.stop()
    
    # Connection status
    with st.expander("🔧 Connection Status", expanded=True):
        if st.button("🔄 Test Connection"):
            if api_client.find_working_connection():
                st.success("✅ Successfully connected to Bybit API!")
                st.info(f"Using endpoint: {api_client.working_endpoint}")
            else:
                st.error("❌ Could not establish connection")
                st.stop()
    
    # Sidebar controls
    st.sidebar.header("🔎 Filters & Settings")
    
    # Data options
    include_funding_history = st.sidebar.checkbox("📈 Include Funding History", value=True,
                                                help="Fetch historical funding rates (slower but more complete)")
    
    test_mode = st.sidebar.checkbox("🧪 Test Mode (50 symbols)", value=False,
                                  help="Limit symbols for faster loading")
    
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False,
                                   help="Show additional debugging information")
    
    if st.sidebar.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        api_client.working_endpoint = None
        api_client.session = None
        st.rerun()
    
    # Main data fetching
    with st.spinner("Fetching comprehensive market data..."):
        try:
            # Ensure connection
            if not api_client.working_endpoint:
                if not api_client.find_working_connection():
                    st.error("❌ Cannot establish connection")
                    st.stop()
            
            # Fetch tickers (includes current funding rates)
            tickers = fetch_tickers()
            if tickers.empty:
                st.error("🚫 No ticker data received")
                st.stop()
            
            st.success(f"✅ Loaded {len(tickers)} symbols")
            
            # Get symbols list
            symbols = tickers["symbol"].unique().tolist()
            if test_mode:
                symbols = symbols[:50]
                st.info(f"🧪 Test mode: Using first {len(symbols)} symbols")
            
            # Filter tickers to match symbols
            tickers = tickers[tickers["symbol"].isin(symbols)]
            
            # Fetch instruments for leverage info
            try:
                st.info("📊 Fetching instruments data for leverage information...")
                instruments = fetch_instruments()
                
                if not instruments.empty:
                    # Extract leverage information
                    leverage_df = extract_leverage_info(instruments)
                    
                    if debug_mode:
                        st.write("🔍 Leverage extraction results:")
                        st.write(f"Total instruments: {len(instruments)}")
                        st.write(f"Leverage data extracted: {len(leverage_df)}")
                        st.write(f"Non-null leverage values: {leverage_df['maxLeverage'].notna().sum()}")
                        
                        # Show sample leverage data
                        sample_with_leverage = leverage_df[leverage_df['maxLeverage'].notna()].head()
                        if not sample_with_leverage.empty:
                            st.dataframe(sample_with_leverage)
                        else:
                            st.warning("No leverage data found!")
                else:
                    leverage_df = pd.DataFrame(columns=['symbol', 'maxLeverage'])
                    st.warning("⚠️ No instruments data received")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not fetch instruments data: {str(e)}")
                leverage_df = pd.DataFrame(columns=['symbol', 'maxLeverage'])
            
            # Fetch historical funding rates if requested
            funding_history_df = pd.DataFrame()
            if include_funding_history:
                st.info("📊 Fetching historical funding rates...")
                funding_history_df = fetch_historical_funding_rates(symbols)
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            return
    
    # Process and enhance the data
    with st.spinner("Calculating enhanced metrics..."):
        try:
            # Calculate all metrics
            df = calculate_enhanced_metrics(tickers, funding_history_df)
            
            # Add leverage info
            if not leverage_df.empty:
                df = pd.merge(df, leverage_df, on="symbol", how="left")
            else:
                df["maxLeverage"] = None
            
            # Create the enhanced display dataframe
            display_columns = [
                "symbol", "priceChange24h", "fundingRate", "currentFundingAPR",
                "funding24hAPR", "funding3dAPR", "funding7dAPR", "funding30dAPR", 
                "basis", "spread", "volume24hFormatted", "maxLeverage", "openInterest"
            ]
            
            # Check which columns actually exist
            available_columns = [col for col in display_columns if col in df.columns]
            missing_columns = [col for col in display_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Missing columns: {missing_columns}")
            
            display_df = df[available_columns].copy()
            
            # Rename columns to match your image
            column_mapping = {
                "symbol": "Symbol",
                "priceChange24h": "Price Change 24h", 
                "fundingRate": "Funding", 
                "currentFundingAPR": "Current Funding Annualised",
                "funding24hAPR": "Funding 24hr avg", 
                "funding3dAPR": "Funding 3d avg", 
                "funding7dAPR": "Funding 7d avg", 
                "funding30dAPR": "Funding 30d avg",
                "basis": "Basis", 
                "spread": "Spread", 
                "volume24hFormatted": "Volume 24h", 
                "maxLeverage": "Max Leverage", 
                "openInterest": "Open Interest"
            }
            
            # Only rename columns that exist
            existing_mapping = {k: v for k, v in column_mapping.items() if k in display_df.columns}
            display_df = display_df.rename(columns=existing_mapping)
            
        except Exception as e:
            st.error(f"❌ Error calculating metrics: {str(e)}")
            st.write(f"Available columns: {list(df.columns) if 'df' in locals() else 'None'}")
            return
    
    # Filters
    st.sidebar.subheader("📊 Data Filters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_apr = st.number_input("Min APR (%)", value=0.0, step=5.0)
    with col2:
        max_apr = st.number_input("Max APR (%)", value=1000.0, step=10.0)
    
    min_volume = st.sidebar.number_input("Min Volume (M)", value=0.0, step=1.0)
    symbol_filter = st.sidebar.text_input("Symbol contains", "").upper()
    
    # Apply filters
    filtered = display_df[
        (display_df["Current Funding Annualised"].fillna(0) >= min_apr) & 
        (display_df["Current Funding Annualised"].fillna(0) <= max_apr) &
        (display_df["Volume 24h"].fillna(0) >= min_volume)
    ].copy()
    
    if symbol_filter:
        filtered = filtered[filtered["Symbol"].str.contains(symbol_filter, na=False)]
    
    # Sort by current funding APR
    filtered = filtered.sort_values("Current Funding Annualised", ascending=False, na_position='last')
    
    # Display results
    if not filtered.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbols Found", len(filtered))
        with col2:
            avg_apr = filtered["Current Funding Annualised"].mean()
            st.metric("Avg Current APR", f"{avg_apr:.2f}%" if not pd.isna(avg_apr) else "N/A")
        with col3:
            total_vol = filtered["Volume 24h"].sum()
            st.metric("Total Volume", f"${total_vol:.0f}M" if not pd.isna(total_vol) else "N/A")
        with col4:
            leverage_available = filtered["Max Leverage"].notna().sum()
            st.metric("Leverage Data", f"{leverage_available}/{len(filtered)}")
        
        st.subheader(f"📊 Enhanced Market Data ({len(filtered)} symbols)")
        
        # Show leverage statistics if debug mode is on
        if debug_mode:
            st.write("🔍 Data Analysis:")
            st.write(f"Symbols with leverage data: {filtered['Max Leverage'].notna().sum()}")
            st.write(f"Average max leverage: {filtered['Max Leverage'].mean():.1f}" if filtered['Max Leverage'].notna().any() else "No leverage data")
            st.write(f"Max leverage range: {filtered['Max Leverage'].min():.0f} - {filtered['Max Leverage'].max():.0f}" if filtered['Max Leverage'].notna().any() else "No leverage data")
            
            # Check spread data
            spread_non_zero = (filtered['Spread'] != 0).sum()
            spread_total = len(filtered)
            st.write(f"Symbols with non-zero spread: {spread_non_zero}/{spread_total}")
            
            if spread_non_zero > 0:
                spread_values = filtered[filtered['Spread'] != 0]['Spread']
                st.write(f"Spread range: {spread_values.min():.4f}% - {spread_values.max():.4f}%")
                st.write(f"Average spread: {spread_values.mean():.4f}%")
            
            # Show some sample spread values
            st.write("📋 Sample spread values:")
            sample_spreads = filtered[['Symbol', 'Spread']].head(10)
            st.dataframe(sample_spreads)
            
            # Debug the data pipeline
            st.write("🔍 Data Pipeline Debug:")
            st.write(f"Original tickers shape: {tickers.shape}")
            st.write(f"Enhanced df shape: {df.shape}")
            st.write(f"Display df shape: {display_df.shape}")
            
            # Check if spread data exists in different stages
            if 'basisRate' in tickers.columns:
                basis_rate_non_null = tickers['basisRate'].notna().sum()
                basis_rate_non_zero = pd.to_numeric(tickers['basisRate'], errors='coerce').fillna(0).ne(0).sum()
                st.write(f"Non-null basisRate in tickers: {basis_rate_non_null}")
                st.write(f"Non-zero basisRate in tickers: {basis_rate_non_zero}")
                
            if 'spread' in df.columns:
                spread_non_zero_original = (pd.to_numeric(df['spread'], errors='coerce').fillna(0) != 0).sum()
                st.write(f"Non-zero spread in enhanced df: {spread_non_zero_original}")
                
                # Show spread calculation breakdown
                if 'bid1Price' in df.columns and 'ask1Price' in df.columns:
                    bid_ask_available = (pd.to_numeric(df['bid1Price'], errors='coerce').notna() & 
                                       pd.to_numeric(df['ask1Price'], errors='coerce').notna()).sum()
                    st.write(f"Symbols with bid/ask data: {bid_ask_available}")
                
            # Show sample values at each stage
            st.write("📋 Sample data at each stage:")
            sample_ticker = tickers[['symbol', 'basisRate', 'bid1Price', 'ask1Price']].head(3)
            st.write("Tickers stage:")
            st.dataframe(sample_ticker)
            
            sample_enhanced = df[['symbol', 'spread', 'basis']].head(3) if 'spread' in df.columns else pd.DataFrame()
            st.write("Enhanced stage:")
            st.dataframe(sample_enhanced)
            
            # Check available columns in original data
            st.write("📋 Available ticker columns:")
            st.write(list(tickers.columns))
        
        # Format the display
        formatted_df = filtered.copy()
        
        # Format percentage columns
        pct_cols = ["Price Change 24h", "Current Funding Annualised", "Funding 24hr avg", 
                   "Funding 3d avg", "Funding 7d avg", "Funding 30d avg", "Basis", "Spread"]
        
        for col in pct_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.4f}%" if not pd.isna(x) and x != 0 else "0.0000%" if x == 0 else "N/A"
                )
        
        # Format funding rate (smaller numbers)
        if "Funding" in formatted_df.columns:
            formatted_df["Funding"] = formatted_df["Funding"].apply(
                lambda x: f"{x:.6f}" if not pd.isna(x) else "N/A"
            )
        
        # Format volume
        if "Volume 24h" in formatted_df.columns:
            formatted_df["Volume 24h"] = formatted_df["Volume 24h"].apply(
                lambda x: f"{x:.1f}M" if not pd.isna(x) else "N/A"
            )
        
        # Format open interest
        if "Open Interest" in formatted_df.columns:
            formatted_df["Open Interest"] = formatted_df["Open Interest"].apply(
                lambda x: f"{x/1e6:.1f}M" if not pd.isna(x) and x >= 1e6 else f"{x/1e3:.0f}K" if not pd.isna(x) else "N/A"
            )
        
        # Format max leverage
        if "Max Leverage" in formatted_df.columns:
            formatted_df["Max Leverage"] = formatted_df["Max Leverage"].apply(
                lambda x: f"{x:.0f}x" if not pd.isna(x) else "N/A"
            )
        
        # Display the enhanced dataframe
        st.dataframe(formatted_df, use_container_width=True, height=500)
        
        # Export functionality
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Enhanced Data to CSV",
            csv,
            f"bybit_enhanced_data_{int(time.time())}.csv",
            "text/csv",
            help="Download complete dataset with all metrics"
        )
        
    else:
        st.warning("No data matches your filters. Try adjusting the criteria.")
    
    # Footer
    st.markdown("---")
    st.markdown("*📊 Enhanced dashboard with historical funding analysis. Data updates every 1-5 minutes.*")

if __name__ == "__main__":
    build_enhanced_dashboard()
    