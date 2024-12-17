import tkinter as tk
from tkinter import ttk, scrolledtext
import yfinance as yf
from tkinter import messagebox
import threading
import logging
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime, timedelta
from functools import lru_cache
import json
from pathlib import Path
import time as time_module
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('10k_viewer.log')
    ]
)
logger = logging.getLogger(__name__)

# Add these constants after imports
SEC_REQUEST_DELAY = 0.1  # 100ms delay between requests to stay under rate limit
SEC_CACHE_FILE = 'sec_cik_cache.json'
SEC_CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds
SEC_API_DELAY = 0.1  # 100ms between requests
SEC_MAX_RETRIES = 3  # Maximum number of retries for failed requests
SEC_BACKOFF_FACTOR = 2  # Exponential backoff factor
SEC_API_HEADERS = {
    'User-Agent': 'Placeholder User Agent',  # Will be updated from config
    'Accept': 'application/json',
}
CONFIG_FILE = 'sec_config.json'
SEC_FORM_TYPES = {
    '10-K': 'Annual Report',
    '10-Q': 'Quarterly Report',
    '8-K': 'Current Report',
    '4': 'Statement of Changes in Beneficial Ownership',
    '13F': 'Institutional Investment Manager Holdings',
    'S-1': 'Initial Registration Statement',
    'DEF 14A': 'Definitive Proxy Statement',
    '144': 'Notice of Proposed Sale of Securities',
    'SC 13G': 'Beneficial Ownership Report',
    'SC 13D': 'Beneficial Ownership Report - Activist',
    'All': 'All Filing Types'
}
SEC_10K_SECTIONS = {
    'All': 'Full Document',
    'Business': 'Item 1. Business',
    'Risk Factors': 'Item 1A. Risk Factors',
    'Properties': 'Item 2. Properties',
    'Legal Proceedings': 'Item 3. Legal Proceedings',
    'Financial Data': 'Item 6. Selected Financial Data',
    'MD&A': 'Item 7. Management\'s Discussion and Analysis',
    'Results': 'Results of Operations',
    'Financial Statements': 'Item 8. Financial Statements',
    'Controls': 'Item 9. Controls and Procedures',
    'Executive Officers': 'Item 10. Directors, Executive Officers',
    'Compensation': 'Item 11. Executive Compensation'
}
FILING_TYPES = {
    '10-K': {
        'start_markers': ['ITEM 1.', 'PART I', 'BUSINESS'],
        'sections': [
            ('Item 1', 'ITEM 1.', 'ITEM 1A.'),
            ('Item 1A', 'ITEM 1A.', 'ITEM 2.'),
            ('Item 2', 'ITEM 2.', 'ITEM 3.'),
            ('Item 3', 'ITEM 3.', 'ITEM 4.'),
            ('Item 7', 'ITEM 7.', 'ITEM 7A.'),
            ('Item 8', 'ITEM 8.', 'ITEM 9.')
        ]
    }
}

def get_sp500_tickers():
    """Fetch S&P 500 components using Wikipedia and verify with yfinance"""
    try:
        logger.info("Fetching S&P 500 components from Wikipedia...")
        # Get S&P 500 table from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        tickers = []
        for row in table.findAll('tr')[1:]:  # Skip header row
            ticker = row.findAll('td')[0].text.strip()
            ticker = ticker.replace(".", "-")  # Handle BRK.B -> BRK-B
            tickers.append(ticker)
        
        logger.info(f"Found {len(tickers)} tickers from Wikipedia")
        return tickers

    except Exception as e:
        logger.error(f"Error fetching S&P 500 components: {e}")
        # Fallback to top companies if Wikipedia fails
        logger.warning("Falling back to top companies list")
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "XOM", "UNH", "JNJ",
            "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "LLY", "AVGO", "PEP"
        ]

def load_cached_data():
    """Load company data from cache file"""
    try:
        if os.path.exists('sp500_cache.csv'):
            df = pd.read_csv('sp500_cache.csv')
            cache_time = datetime.fromtimestamp(os.path.getmtime('sp500_cache.csv'))
            
            # Check if cache is older than 24 hours
            if datetime.now() - cache_time > timedelta(hours=24):
                return None
                
            # Add CIK column if it doesn't exist (for backwards compatibility)
            if 'CIK' not in df.columns:
                df['CIK'] = 'N/A'
                
            return df
        return None
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None

def save_cached_data(data):
    """Save company data to cache file"""
    try:
        processed_data = []
        for company in data:
            info = company['Info']
            processed_company = {
                'Ticker': company['Ticker'],
                'Company Name': company['Company Name'],
                'Market Cap': company['Market Cap'],
                'CIK': info.get('cik', 'N/A'),  # Add CIK to cached data
                # General Information
                'Industry': info.get('industry', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Country': info.get('country', 'N/A'),
                'State': info.get('state', 'N/A'),
                'City': info.get('city', 'N/A'),
                'Website': info.get('website', 'N/A'),
                'Employees': info.get('fullTimeEmployees', 'N/A'),
                'Business Summary': info.get('longBusinessSummary', 'N/A'),
                # Financial Information
                'Total Revenue': info.get('totalRevenue', 'N/A'),
                'Operating Margin': info.get('operatingMargins', 'N/A'),
                'Profit Margin': info.get('profitMargins', 'N/A'),
                'EBITDA': info.get('ebitda', 'N/A'),
                'Total Cash': info.get('totalCash', 'N/A'),
                'Total Debt': info.get('totalDebt', 'N/A'),
                'Debt To Equity': info.get('debtToEquity', 'N/A'),
                # Market Data
                'Current Price': info.get('currentPrice', 'N/A'),
                'Target High': info.get('targetHighPrice', 'N/A'),
                'Target Low': info.get('targetLowPrice', 'N/A'),
                'PE Ratio': info.get('trailingPE', 'N/A'),
                'Dividend Yield': info.get('dividendYield', 'N/A'),
                'Volume': info.get('volume', 'N/A'),
                'Avg Volume': info.get('averageVolume', 'N/A'),
            }
            processed_data.append(processed_company)
            
        df = pd.DataFrame(processed_data)
        df.to_csv('sp500_cache.csv', index=False)
        logger.info("Cache saved successfully")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def load_sec_cache():
    """Load the SEC CIK cache from file"""
    try:
        cache_path = Path(SEC_CACHE_FILE)
        if cache_path.exists():
            cache_age = time_module.time() - cache_path.stat().st_mtime
            if cache_age < SEC_CACHE_DURATION:
                with open(cache_path, 'r') as f:
                    return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading SEC cache: {e}")
        return {}

def save_sec_cache(cache_data):
    """Save the SEC CIK cache to file"""
    try:
        with open(SEC_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.error(f"Error saving SEC cache: {e}")

@lru_cache(maxsize=1000)
def get_cik_from_ticker(ticker, retry_count=0, max_retries=3):
    """Get CIK number from SEC using the official company ticker lookup"""
    # Check local cache first
    cache = load_sec_cache()
    if ticker in cache:
        logger.debug(f"CIK for {ticker} found in cache")
        return cache[ticker]
    
    try:
        headers = {
            'User-Agent': 'Luis Rincon lrincon2019@example.com',
            'Accept': 'application/json',
            'Host': 'www.sec.gov'
        }
        
        # Use the correct SEC endpoint for company ticker lookup
        url = "https://www.sec.gov/files/company_tickers.json"
        
        logger.debug(f"Requesting CIK for {ticker} from SEC API: {url}")
        response = requests.get(url, headers=headers)
        
        logger.debug(f"SEC API response status: {response.status_code}")
        logger.debug(f"SEC API response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                logger.debug(f"SEC API response data structure: {type(data)}")
                
                # Find the matching ticker in the response
                for entry in data.values():
                    if entry['ticker'].upper() == ticker.upper():
                        cik = str(entry['cik_str']).zfill(10)
                        cache[ticker] = cik
                        save_sec_cache(cache)
                        logger.info(f"Successfully retrieved CIK {cik} for {ticker}")
                        return cik
                
                logger.error(f"No CIK found in response data for {ticker}")
                raise ValueError(f"No CIK found in response for {ticker}")
                    
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {je}")
                logger.debug(f"Raw response content: {response.text[:500]}...")
                raise
            
        elif response.status_code == 429:  # Rate limit exceeded
            if retry_count < max_retries:
                wait_time = SEC_API_DELAY * (SEC_BACKOFF_FACTOR ** retry_count)
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)
                return get_cik_from_ticker(ticker, retry_count + 1)
            else:
                raise Exception("Max retries exceeded for rate limit")
                
        else:
            logger.error(f"SEC API returned status code: {response.status_code}")
            logger.error(f"Response content: {response.text[:500]}...")
            raise Exception(f"SEC API returned unexpected status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {str(e)}")
        if retry_count < max_retries:
            wait_time = SEC_API_DELAY * (SEC_BACKOFF_FACTOR ** retry_count)
            logger.warning(f"Network error, retrying in {wait_time}s")
            time.sleep(wait_time)
            return get_cik_from_ticker(ticker, retry_count + 1)
        raise Exception(f"Network error fetching CIK for {ticker}: {str(e)}")

def load_sec_config():
    """Load SEC API configuration from file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config
        return None
    except Exception as e:
        logger.error(f"Error loading SEC config: {e}")
        return None

def save_sec_config(config):
    """Save SEC API configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        return True
    except Exception as e:
        logger.error(f"Error saving SEC config: {e}")
        return False

class SP500Viewer:
    def __init__(self, root):
        self.root = root
        self.root.title("S&P 500 Companies")
        self.root.geometry("1000x600")
        
        # Add loading state variables
        self.is_loading = False
        self.is_paused = False
        self.cancel_loading = False
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create control frame for buttons
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Create loading label
        self.loading_label = ttk.Label(self.control_frame, text="Loading S&P 500 data...", font=('Helvetica', 12))
        self.loading_label.pack(side=tk.LEFT, padx=5)
        
        # Create progress label
        self.progress_label = ttk.Label(self.control_frame, text="", font=('Helvetica', 10))
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        # Add pause/resume button
        self.pause_btn = ttk.Button(
            self.control_frame,
            text="Pause",
            command=self.toggle_pause,
            state='disabled'
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Add cancel button
        self.cancel_btn = ttk.Button(
            self.control_frame,
            text="Cancel",
            command=self.cancel_operation,
            state='disabled'
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Add progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.control_frame,
            variable=self.progress_var,
            maximum=100,
            length=200
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Create Treeview
        self.tree = ttk.Treeview(self.main_frame, columns=("Ticker", "Company Name", "Market Cap", "CIK"), show="headings")
        
        # Define columns including CIK
        self.tree.heading("Ticker", text="Ticker", command=lambda: self.sort_treeview("Ticker", False))
        self.tree.heading("Company Name", text="Company Name", command=lambda: self.sort_treeview("Company Name", False))
        self.tree.heading("Market Cap", text="Market Cap", command=lambda: self.sort_treeview("Market Cap", True))
        self.tree.heading("CIK", text="CIK", command=lambda: self.sort_treeview("CIK", False))
        
        # Configure column widths
        self.tree.column("Ticker", width=100)
        self.tree.column("Company Name", width=400)
        self.tree.column("Market Cap", width=200)
        self.tree.column("CIK", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Grid the tree and scrollbar
        self.tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # Create refresh button
        self.refresh_button = ttk.Button(self.main_frame, text="Refresh Data", command=self.refresh_data)
        self.refresh_button.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        # Create back button
        self.back_button = ttk.Button(self.main_frame, text="Back to Menu", command=self.root.destroy)
        self.back_button.grid(row=2, column=0, pady=5, sticky=tk.E)
        
        # Add double-click binding to tree
        self.tree.bind('<Double-1>', self.on_double_click)
        
        # Add cache for company data
        self.company_data = []
        
        # Start loading data
        self.load_data()

    def toggle_pause(self):
        """Toggle between pause and resume"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.configure(text="Resume")
            self.loading_label.configure(text="Loading paused...")
        else:
            self.pause_btn.configure(text="Pause")
            self.loading_label.configure(text="Loading S&P 500 data...")
    
    def cancel_operation(self):
        """Cancel the loading operation"""
        self.cancel_loading = True
        self.loading_label.configure(text="Loading cancelled")
        self.pause_btn.configure(state='disabled')
        self.cancel_btn.configure(state='disabled')
    
    def load_data(self):
        """Start the data loading process"""
        self.is_loading = True
        self.cancel_loading = False
        self.pause_btn.configure(state='normal')
        self.cancel_btn.configure(state='normal')
        
        # Start loading in a background thread
        loading_thread = threading.Thread(target=self._load_sp500_data)
        loading_thread.daemon = True
        loading_thread.start()
    
    def _load_sp500_data(self):
        """Load S&P 500 data with pause/resume support"""
        try:
            # Check cache first
            cached_data = load_cached_data()
            if cached_data is not None:
                self.company_data = cached_data.to_dict('records')
                self.update_treeview()
                return
            
            # Get tickers
            tickers = get_sp500_tickers()
            total_tickers = len(tickers)
            
            self.company_data = []
            for i, ticker in enumerate(tickers):
                # Check for pause
                while self.is_paused and not self.cancel_loading:
                    time.sleep(0.1)
                    continue
                
                # Check for cancellation
                if self.cancel_loading:
                    return
                
                try:
                    # Update progress
                    progress = (i / total_tickers) * 100
                    self.update_progress(progress, f"Loading {ticker} ({i+1}/{total_tickers})")
                    
                    # Get company info
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Get CIK
                    cik = get_cik_from_ticker(ticker)
                    
                    # Add to data
                    self.company_data.append({
                        'Ticker': ticker,
                        'Company Name': info.get('longName', 'N/A'),
                        'Market Cap': f"${info.get('marketCap', 0)/1e9:.2f}B",
                        'CIK': cik
                    })
                    
                    # Update display periodically
                    if i % 5 == 0:
                        self.update_treeview()
                    
                except Exception as e:
                    logger.error(f"Error loading data for {ticker}: {e}")
                    continue
            
            # Final update
            self.update_treeview()
            self.save_to_cache()
            
        except Exception as e:
            logger.error(f"Error loading S&P 500 data: {e}")
            self.loading_label.configure(text=f"Error: {str(e)}")
        finally:
            self.is_loading = False
            self.pause_btn.configure(state='disabled')
            self.cancel_btn.configure(state='disabled')
    
    def update_progress(self, value, message=""):
        """Update progress bar and message"""
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.progress_label.configure(text=message))
    
    def sort_treeview(self, column, is_number):
        """Sort treeview by column"""
        items = [(self.tree.set(item, column), item) for item in self.tree.get_children("")]
        
        if is_number:  # For market cap
            # Convert string like "$123.45B" to float
            items = [(float(item[0].replace("$", "").replace("B", "")), item[1]) for item in items]
        
        items.sort(reverse=True)  # Sort in descending order
        for index, (_, item) in enumerate(items):
            self.tree.move(item, "", index)

    def refresh_data(self):
        """Refresh the S&P 500 data"""
        self.loading_label.configure(text="Refreshing S&P 500 data...")
        self.tree.delete(*self.tree.get_children())
        self.refresh_button.configure(state="disabled")
        
        # Start loading data in a separate thread
        self.load_data()
    
    def update_treeview(self):
        """Update the treeview with current data"""
        self.tree.delete(*self.tree.get_children())
        for company in self.company_data:
            self.tree.insert("", tk.END, values=(
                company['Ticker'],
                company['Company Name'],
                company['Market Cap'],
                company['CIK']
            ))
    
    def save_to_cache(self):
        """Save current data to cache"""
        save_cached_data(self.company_data)
    
    def on_double_click(self, event):
        """Handle double-click on tree item"""
        item = self.tree.selection()[0]
        ticker = self.tree.item(item)['values'][0]
        
        # Create new window with TenKViewer
        company_window = tk.Toplevel(self.root)
        viewer = TenKViewer(company_window)
        
        # Set the ticker and trigger search
        viewer.ticker_entry.insert(0, ticker)
        viewer.search_ticker()

        # Find cached company info
        company_info = next((item for item in self.company_data if item['Ticker'] == ticker), None)
        if company_info and 'Info' in company_info:
            viewer.display_company_info(company_info['Info'])

class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("10-K Viewer")
        self.root.geometry("400x300")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create title
        title_label = ttk.Label(self.main_frame, text="10-K Viewer", font=('Helvetica', 24))
        title_label.grid(row=0, column=0, pady=20)
        
        # Create buttons frame
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.grid(row=1, column=0, pady=20)
        
        # Create buttons
        ticker_button = ttk.Button(buttons_frame, text="Ticker Lookup", command=self.open_ticker_lookup)
        ticker_button.grid(row=0, column=0, pady=10, padx=10, ipadx=20, ipady=10)
        
        sp500_button = ttk.Button(buttons_frame, text="S&P 500", command=self.open_sp500_view)
        sp500_button.grid(row=1, column=0, pady=10, padx=10, ipadx=20, ipady=10)
        
        # Add SEC Config button to buttons_frame
        sec_config_button = ttk.Button(buttons_frame, text="SEC API Settings", 
                                     command=self.open_sec_config)
        sec_config_button.grid(row=2, column=0, pady=10, padx=10, ipadx=20, ipady=10)
        
        # Load SEC configuration on startup
        self.load_sec_settings()
    
    def load_sec_settings(self):
        """Load SEC API settings on startup"""
        config = load_sec_config()
        if config:
            user_agent = f"{config['company']} {config['email']} {config['phone']}"
            global SEC_API_HEADERS
            SEC_API_HEADERS['User-Agent'] = user_agent
        else:
            # Show configuration dialog if no settings exist
            self.open_sec_config()
    
    def open_sec_config(self):
        """Open the SEC configuration dialog"""
        SECConfigDialog(self.root)
    
    def open_ticker_lookup(self):
        self.root.withdraw()  # Hide the main menu
        ticker_window = tk.Toplevel()
        app = TenKViewer(ticker_window)
        ticker_window.protocol("WM_DELETE_WINDOW", lambda: self.on_window_close(ticker_window))

    def open_sp500_view(self):
        self.root.withdraw()  # Hide the main menu
        sp500_window = tk.Toplevel()
        app = SP500Viewer(sp500_window)
        sp500_window.protocol("WM_DELETE_WINDOW", lambda: self.on_window_close(sp500_window))

    def on_window_close(self, window):
        window.destroy()
        self.root.deiconify()  # Show the main menu again

class TenKViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Ticker Lookup")
        self.root.geometry("1000x800")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        # Create search frame
        self.search_frame = ttk.Frame(self.main_frame)
        self.search_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Create search entry and button
        ttk.Label(self.search_frame, text="Ticker:").pack(side=tk.LEFT)
        self.ticker_entry = ttk.Entry(self.search_frame)
        self.ticker_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(self.search_frame, text="Search", command=self.search_ticker).pack(side=tk.LEFT)
        
        # Add CIK display frame below search frame
        self.cik_frame = ttk.Frame(self.main_frame)
        self.cik_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Create CIK label with blue background
        self.cik_label = ttk.Label(self.cik_frame, text="CIK: ", font=('Helvetica', 10, 'bold'))
        self.cik_label.pack(side=tk.LEFT)
        
        # Create CIK value display
        self.cik_value = tk.Label(self.cik_frame, text="", font=('Helvetica', 10),
                                 bg='lightblue', relief='solid', padx=10, pady=2)
        self.cik_value.pack(side=tk.LEFT)
        
        # Create notebook for organized display
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create different tabs for different types of information
        self.general_frame = ttk.Frame(self.notebook, padding="10")
        self.financial_frame = ttk.Frame(self.notebook, padding="10")
        self.market_frame = ttk.Frame(self.notebook, padding="10")
        self.filings_frame = ttk.Frame(self.notebook, padding="10")
        
        self.notebook.add(self.general_frame, text="General Info")
        self.notebook.add(self.financial_frame, text="Financial Info")
        self.notebook.add(self.market_frame, text="Market Data")
        self.notebook.add(self.filings_frame, text="SEC Filings")
        
        # Configure scrolling for each frame
        self.setup_scrollable_frame(self.general_frame)
        self.setup_scrollable_frame(self.financial_frame)
        self.setup_scrollable_frame(self.market_frame)
        self.setup_scrollable_frame(self.filings_frame)
        
        # Add filing-specific widgets to the frame
        self.setup_filings_tab(self.filings_frame)
        
        # Create back button
        self.back_button = ttk.Button(self.main_frame, text="Back to Menu", command=self.root.destroy)
        self.back_button.grid(row=3, column=1, pady=5, sticky=tk.E)

    def setup_scrollable_frame(self, parent):
        """Create a scrollable frame"""
        # Create a canvas and scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        
        # Create a frame inside the canvas
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Add the frame to the canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the widgets
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return scrollable_frame

    def create_info_field(self, parent, label, value):
        """Create a formatted field with label and value"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        label_widget = ttk.Label(frame, text=f"{label}:", width=20, anchor="e")
        label_widget.pack(side=tk.LEFT, padx=(0, 10))
        
        value_widget = ttk.Entry(frame)
        value_widget.insert(0, str(value))
        value_widget.configure(state="readonly")
        value_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        return frame

    def display_company_info(self, info):
        """Display company information in organized tabs"""
        logger.debug("Displaying company info")
        
        # Update CIK display
        cik = info.get('cik', 'N/A')
        self.cik_value.configure(text=cik)
        logger.debug(f"CIK set to: {cik}")
        
        # Clear existing widgets in frames
        for frame in [self.general_frame, self.financial_frame, self.market_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        # Create new scrollable frames
        general_scroll = self.setup_scrollable_frame(self.general_frame)
        financial_scroll = self.setup_scrollable_frame(self.financial_frame)
        market_scroll = self.setup_scrollable_frame(self.market_frame)
        
        # General information
        general_fields = {
            'longName': 'Company Name',
            'industry': 'Industry',
            'sector': 'Sector',
            'country': 'Country',
            'state': 'State',
            'city': 'City',
            'website': 'Website',
            'fullTimeEmployees': 'Employees',
            'longBusinessSummary': 'Business Summary'
        }
        
        # Financial information
        financial_fields = {
            'totalRevenue': 'Total Revenue',
            'operatingMargins': 'Operating Margin',
            'profitMargins': 'Profit Margin',
            'ebitda': 'EBITDA',
            'totalCash': 'Total Cash',
            'totalDebt': 'Total Debt',
            'debtToEquity': 'Debt To Equity',
            'returnOnEquity': 'Return on Equity',
            'returnOnAssets': 'Return on Assets'
        }
        
        # Market data
        market_fields = {
            'currentPrice': 'Current Price',
            'targetHighPrice': 'Target High',
            'targetLowPrice': 'Target Low',
            'trailingPE': 'PE Ratio',
            'dividendYield': 'Dividend Yield',
            'volume': 'Volume',
            'averageVolume': 'Avg Volume',
            'marketCap': 'Market Cap',
            'fiftyTwoWeekHigh': '52 Week High',
            'fiftyTwoWeekLow': '52 Week Low'
        }
        
        # Helper function to format values
        def format_value(key, value):
            if value is None:
                return 'N/A'
            if isinstance(value, (int, float)):
                if 'Price' in key or 'Market' in key or 'Revenue' in key or 'EBITDA' in key or 'Cash' in key or 'Debt' in key:
                    if value >= 1_000_000_000:
                        return f"${value/1_000_000_000:.2f}B"
                    elif value >= 1_000_000:
                        return f"${value/1_000_000:.2f}M"
                    else:
                        return f"${value:,.2f}"
                elif 'Margin' in key or 'Yield' in key:
                    return f"{value*100:.2f}%"
                elif 'Volume' in key:
                    return f"{value:,}"
            return str(value)
        
        # Populate the frames
        for fields, scroll_frame, field_dict in [
            (general_fields, general_scroll, general_fields),
            (financial_fields, financial_scroll, financial_fields),
            (market_fields, market_scroll, market_fields)
        ]:
            for key, display_name in field_dict.items():
                if key in info:
                    formatted_value = format_value(display_name, info[key])
                    self.create_info_field(scroll_frame, display_name, formatted_value)

    def search_ticker(self):
        ticker = self.ticker_entry.get().strip().upper()
        if not ticker:
            logger.warning("Empty ticker symbol entered")
            messagebox.showwarning("Warning", "Please enter a ticker symbol")
            return
            
        try:
            logger.info(f"Searching for ticker: {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get CIK from SEC
            cik = get_cik_from_ticker(ticker)
            info['cik'] = cik  # Add CIK to info dictionary
            
            logger.info(f"Successfully retrieved data for {ticker}")
            self.display_company_info(info)
            
            # Automatically refresh filings after loading company info
            self.refresh_filings()
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            messagebox.showerror("Error", f"Error fetching data: {str(e)}")

    def setup_filings_tab(self, parent):
        """Setup the SEC filings tab with a treeview and refresh button"""
        logger.debug("Setting up filings tab")
        
        # Create a container frame that won't be destroyed
        self.filings_container = ttk.Frame(parent)
        self.filings_container.pack(fill=tk.BOTH, expand=True)
        
        # Create frame for controls at the top
        controls_frame = ttk.Frame(self.filings_container)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Add form type filter
        ttk.Label(controls_frame, text="Form Type:").pack(side=tk.LEFT, padx=5)
        self.form_type_var = tk.StringVar(value="All")
        form_type_combo = ttk.Combobox(controls_frame, 
                                      textvariable=self.form_type_var,
                                      values=list(SEC_FORM_TYPES.keys()),
                                      state='readonly',
                                      width=20)
        form_type_combo.pack(side=tk.LEFT, padx=5)
        form_type_combo.bind('<<ComboboxSelected>>', self.filter_filings)
        
        # Add refresh button
        refresh_btn = ttk.Button(controls_frame, text="Refresh Filings", 
                                command=self.refresh_filings)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Create frame for treeview
        tree_frame = ttk.Frame(self.filings_container)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview for filings
        self.filings_tree = ttk.Treeview(tree_frame, 
                                        columns=("Filing Date", "Form Type", "Form Description", "Description", "Accession Number"),
                                        show="headings")
        
        # Setup columns
        self.filings_tree.heading("Filing Date", text="Filing Date")
        self.filings_tree.heading("Form Type", text="Form Type")
        self.filings_tree.heading("Form Description", text="Form Description")
        self.filings_tree.heading("Description", text="Description")
        self.filings_tree.heading("Accession Number", text="Accession Number")
        
        # Configure column widths
        self.filings_tree.column("Filing Date", width=100)
        self.filings_tree.column("Form Type", width=80)
        self.filings_tree.column("Form Description", width=150)
        self.filings_tree.column("Description", width=400)
        self.filings_tree.column("Accession Number", width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, 
                                 command=self.filings_tree.yview)
        self.filings_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.filings_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click event
        self.filings_tree.bind('<Double-1>', self.open_filing)
        
        logger.debug("Filings tab setup complete")

    def filter_filings(self, event=None):
        """Filter filings based on selected form type"""
        selected_type = self.form_type_var.get()
        
        # Show all items if "All" is selected
        if selected_type == "All":
            for item in self.filings_tree.get_children():
                self.filings_tree.item(item, tags=())
                self.filings_tree.reattach(item, "", "end")
            return
        
        # Hide items that don't match the selected type
        for item in self.filings_tree.get_children():
            values = self.filings_tree.item(item)['values']
            if values[1] != selected_type:  # Form Type is at index 1
                self.filings_tree.detach(item)
            else:
                self.filings_tree.reattach(item, "", "end")

    def refresh_filings(self):
        """Fetch and display SEC filings for the current company"""
        logger.debug("Refreshing filings")
        
        try:
            # Clear existing items
            for item in self.filings_tree.get_children():
                self.filings_tree.delete(item)
            
            # Get CIK for current company
            cik = self.cik_value.cget("text")
            logger.debug(f"Using CIK: {cik}")
            
            if not cik or cik == "N/A":
                raise ValueError("No CIK available for this company")
            
            # Ensure CIK is padded to 10 digits
            cik = cik.zfill(10)
            logger.debug(f"Padded CIK: {cik}")
            
            # Use the submissions endpoint
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            logger.debug(f"Requesting URL: {url}")
            logger.debug(f"Using headers: {SEC_API_HEADERS}")
            
            # Make request with proper headers and rate limiting
            time.sleep(SEC_REQUEST_DELAY)
            response = requests.get(url, headers=SEC_API_HEADERS)
            
            logger.debug(f"Response status code: {response.status_code}")
            
            if response.status_code == 429:
                logger.warning("SEC rate limit hit, waiting before retry")
                time.sleep(10)
                response = requests.get(url, headers=SEC_API_HEADERS)
            
            response.raise_for_status()
            data = response.json()
            
            # Process recent filings
            recent = data.get('filings', {}).get('recent', {})
            logger.debug(f"Found {len(recent.get('filingDate', []))} filings")
            
            if not recent:
                logger.warning("No filings found in response")
                messagebox.showinfo("Info", "No filings found for this company")
                return
            
            # Get the filing data
            dates = recent.get('filingDate', [])
            forms = recent.get('form', [])
            descriptions = recent.get('primaryDocument', [])
            accession_nums = recent.get('accessionNumber', [])
            
            # Insert into treeview with form descriptions
            for i in range(len(dates)):
                form_type = forms[i]
                form_desc = SEC_FORM_TYPES.get(form_type, "Other Filing")
                
                self.filings_tree.insert("", tk.END, values=(
                    dates[i],
                    form_type,
                    form_desc,
                    descriptions[i],
                    accession_nums[i]
                ))
                logger.debug(f"Inserted filing: {form_type} ({form_desc}) from {dates[i]}")
            
            # Apply current filter
            self.filter_filings()
                
        except Exception as e:
            logger.error(f"Error refreshing filings: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to fetch filings: {str(e)}")

    def open_filing(self, event):
        """Open a new window to display the selected filing"""
        selection = self.filings_tree.selection()
        if not selection:
            return
        
        item = self.filings_tree.item(selection[0])
        values = item['values']
        if not values:
            return
        
        # Create filing info dictionary
        filing_info = {
            'date': values[0],
            'form_type': values[1],
            'form_desc': values[2],
            'doc_name': values[3],
            'accession_num': values[4],
            'cik': self.cik_value.cget("text").zfill(10)
        }
        
        # Create filing viewer
        FilingViewer(self.root, filing_info)

    def setup_text_widget_events(self, text_widget):
        """Configure text widget event handling for better performance"""
        
        # Variable to track if we're currently processing a scroll
        processing_scroll = False
        
        def on_scroll(*args):
            nonlocal processing_scroll
            if not processing_scroll:
                processing_scroll = True
                text_widget.after(50, lambda: handle_scroll_end())
        
        def handle_scroll_end():
            nonlocal processing_scroll
            processing_scroll = False
            text_widget.update_idletasks()
        
        # Bind scroll events
        text_widget.bind("<MouseWheel>", on_scroll)
        text_widget.bind("<Button-4>", on_scroll)
        text_widget.bind("<Button-5>", on_scroll)

class SECConfigDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("SEC API Configuration")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        
        # Create main frame
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)
        
        # Create form fields
        ttk.Label(main_frame, text="Company Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.company_entry = ttk.Entry(main_frame, width=40)
        self.company_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(main_frame, text="Email:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.email_entry = ttk.Entry(main_frame, width=40)
        self.email_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(main_frame, text="Phone:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.phone_entry = ttk.Entry(main_frame, width=40)
        self.phone_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Load existing config
        config = load_sec_config()
        if config:
            self.company_entry.insert(0, config.get('company', ''))
            self.email_entry.insert(0, config.get('email', ''))
            self.phone_entry.insert(0, config.get('phone', ''))
        
        # Create buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Save", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Make dialog modal
        self.dialog.grab_set()
        
    def save_config(self):
        """Save the configuration and update headers"""
        config = {
            'company': self.company_entry.get().strip(),
            'email': self.email_entry.get().strip(),
            'phone': self.phone_entry.get().strip()
        }
        
        # Validate inputs
        if not all(config.values()):
            messagebox.showerror("Error", "All fields are required")
            return
        
        if save_sec_config(config):
            # Update the global SEC_API_HEADERS
            user_agent = f"{config['company']} {config['email']} {config['phone']}"
            global SEC_API_HEADERS
            SEC_API_HEADERS['User-Agent'] = user_agent
            
            messagebox.showinfo("Success", "SEC API configuration saved")
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "Failed to save configuration")

class FilingViewer:
    def __init__(self, parent, filing_info):
        self.window = tk.Toplevel(parent)
        self.filing_info = filing_info
        self.raw_content = None
        self.sections = {}
        self.current_section = None
        
        # Setup window
        self.window.title(f"SEC Filing - {filing_info['form_type']} ({filing_info['date']})")
        self.window.geometry("1400x900")
        
        # Configure styles
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Section.TFrame', background='#f5f5f5')
        style.configure('Content.TFrame', background='#ffffff')
        style.configure('Status.TLabel', font=('Helvetica', 10), foreground='#666666')
        
        # Create main container with padding
        self.main_container = ttk.Frame(self.window, padding="10", style='Content.TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create header with filing info
        header_frame = ttk.Frame(self.main_container, style='Section.TFrame', padding="10")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_text = f"{filing_info['form_desc']} - Filed on {filing_info['date']}"
        ttk.Label(header_frame, text=info_text, style='Header.TLabel').pack(side=tk.LEFT)
        
        # Create control panel
        control_frame = ttk.Frame(self.main_container, style='Section.TFrame', padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add section navigation
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(nav_frame, text="Section:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.section_var = tk.StringVar()
        self.section_combo = ttk.Combobox(
            nav_frame,
            textvariable=self.section_var,
            state='readonly',
            width=50,
            font=('Helvetica', 10)
        )
        self.section_combo.pack(side=tk.LEFT, padx=5)
        self.section_combo.bind('<<ComboboxSelected>>', self.on_section_change)
        
        # Add status display
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(side=tk.RIGHT, padx=10)
        
        self.status_var = tk.StringVar(value="Loading document...")
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style='Status.TLabel'
        ).pack(side=tk.LEFT)
        
        # Create content area
        content_frame = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create section list
        self.section_frame = ttk.Frame(content_frame, style='Section.TFrame', padding="5")
        self.section_list = ttk.Treeview(
            self.section_frame,
            show="tree",
            selectmode="browse",
            style='Section.Treeview'
        )
        self.section_list.pack(fill=tk.BOTH, expand=True)
        self.section_list.bind('<<TreeviewSelect>>', self.on_section_select)
        
        # Create document view
        self.text_frame = ttk.Frame(content_frame, style='Content.TFrame', padding="5")
        self.text_widget = scrolledtext.ScrolledText(
            self.text_frame,
            wrap=tk.WORD,
            font=('Consolas', 11),
            background='#ffffff',
            foreground='#333333',
            padx=20,
            pady=20,
            spacing1=2,
            spacing2=2,
            spacing3=2
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Configure content proportions
        content_frame.add(self.section_frame, weight=1)
        content_frame.add(self.text_frame, weight=4)
        
        # Start loading
        self.load_document()
    
    def load_document(self):
        """Start document loading in background"""
        self._update_status("Loading document...")
        threading.Thread(target=self._fetch_and_parse_document, daemon=True).start()
    
    def _fetch_and_parse_document(self):
        """Fetch and parse document with improved error handling"""
        try:
            url = self._build_sec_url()
            logger.info(f"Fetching document from: {url}")
            
            with requests.Session() as session:
                session.headers.update(SEC_API_HEADERS)
                response = session.get(url)
                response.raise_for_status()
                
                self.raw_content = response.text
                self._parse_document()
                
        except Exception as e:
            logger.error(f"Error fetching document: {e}", exc_info=True)
            self._update_status(f"Error: {str(e)}")
    
    def _parse_document(self):
        """Parse document with improved section detection"""
        try:
            self._update_status("Parsing document...")
            content = self.raw_content
            
            # First pass: Find all potential section markers
            section_markers = []
            section_patterns = [
                (r'(?i)^\s*ITEM\s+1\.(?!\s*A|\s*B)\s*BUSINESS', 'Item 1 - Business'),
                (r'(?i)^\s*ITEM\s+1A\.\s*RISK\s+FACTORS', 'Item 1A - Risk Factors'),
                (r'(?i)^\s*ITEM\s+2\.\s*PROPERTIES', 'Item 2 - Properties'),
                (r'(?i)^\s*ITEM\s+3\.\s*LEGAL\s+PROCEEDINGS', 'Item 3 - Legal Proceedings'),
                (r'(?i)^\s*ITEM\s+4\.\s*MINE\s+SAFETY', 'Item 4 - Mine Safety'),
                (r'(?i)^\s*ITEM\s+5\.\s*MARKET', 'Item 5 - Market'),
                (r'(?i)^\s*ITEM\s+7\.\s*MANAGEMENT', 'Item 7 - MD&A'),
                (r'(?i)^\s*ITEM\s+7A\.\s*QUANTITATIVE', 'Item 7A - Market Risk'),
                (r'(?i)^\s*ITEM\s+8\.\s*FINANCIAL', 'Item 8 - Financial Statements'),
                (r'(?i)^\s*ITEM\s+9\.\s*CHANGES', 'Item 9 - Changes'),
                (r'(?i)^\s*ITEM\s+10\.\s*DIRECTORS', 'Item 10 - Directors'),
                (r'(?i)^\s*ITEM\s+11\.\s*EXECUTIVE', 'Item 11 - Executive Compensation'),
                (r'(?i)^\s*ITEM\s+12\.\s*SECURITY', 'Item 12 - Security Ownership'),
                (r'(?i)^\s*ITEM\s+13\.\s*CERTAIN', 'Item 13 - Related Transactions'),
                (r'(?i)^\s*ITEM\s+14\.\s*PRINCIPAL', 'Item 14 - Principal Accountant'),
                (r'(?i)^\s*ITEM\s+15\.\s*EXHIBITS', 'Item 15 - Exhibits')
            ]
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                for pattern, title in section_patterns:
                    if re.search(pattern, line.strip()):
                        section_markers.append((i, title))
                        break
            
            # Sort markers by position
            section_markers.sort()
            
            # Second pass: Extract sections
            self.sections = {'Full Document': content}
            
            for i in range(len(section_markers)):
                start_idx = section_markers[i][0]
                end_idx = section_markers[i+1][0] if i < len(section_markers)-1 else len(lines)
                title = section_markers[i][1]
                
                # Extract section content
                section_content = '\n'.join(lines[start_idx:end_idx])
                
                # Clean up content
                section_content = self._clean_section_content(section_content)
                
                if section_content.strip():
                    self.sections[title] = section_content
            
            # Update UI
            self.window.after(0, self._update_section_list)
            self._update_status("Document loaded successfully")
            
        except Exception as e:
            logger.error(f"Error parsing document: {e}", exc_info=True)
            self._update_status("Error parsing document")
            self.sections = {'Full Document': self.raw_content}
            self.window.after(0, self._update_section_list)
    
    def _clean_section_content(self, content):
        """Clean up section content for better readability"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove HTML tags if present
        content = re.sub(r'<[^>]+>', '', content)
        
        # Clean up special characters
        content = content.replace('\xa0', ' ')
        content = content.replace('\u200b', '')
        
        return content.strip()
    
    def _update_section_list(self):
        """Update section list with better organization"""
        # Clear existing items
        self.section_list.delete(*self.section_list.get_children())
        
        # Define section order
        section_order = {
            'Full Document': 0,
            'Item 1 - Business': 1,
            'Item 1A - Risk Factors': 2,
            'Item 2 - Properties': 3,
            'Item 3 - Legal Proceedings': 4,
            'Item 4 - Mine Safety': 5,
            'Item 5 - Market': 6,
            'Item 7 - MD&A': 7,
            'Item 7A - Market Risk': 8,
            'Item 8 - Financial Statements': 9,
            'Item 9 - Changes': 10,
            'Item 10 - Directors': 11,
            'Item 11 - Executive Compensation': 12,
            'Item 12 - Security Ownership': 13,
            'Item 13 - Related Transactions': 14,
            'Item 14 - Principal Accountant': 15,
            'Item 15 - Exhibits': 16
        }
        
        # Sort sections
        sorted_sections = sorted(
            self.sections.keys(),
            key=lambda x: section_order.get(x, 999)
        )
        
        # Update UI
        self.section_combo['values'] = sorted_sections
        
        for section in sorted_sections:
            self.section_list.insert('', 'end', text=section, tags=(section,))
        
        # Select first section
        if sorted_sections:
            self.section_var.set(sorted_sections[0])
            self.display_section(sorted_sections[0])
    
    def display_section(self, section_name):
        """Display section with improved formatting"""
        try:
            content = self.sections.get(section_name, "Section not found")
            
            self.text_widget.configure(state='normal')
            self.text_widget.delete(1.0, tk.END)
            
            # Add section header
            header = f"\n{section_name}\n{'='*len(section_name)}\n\n"
            self.text_widget.insert(tk.END, header, 'header')
            
            # Add content
            self.text_widget.insert(tk.END, content)
            
            # Configure tags
            self.text_widget.tag_configure('header', font=('Consolas', 12, 'bold'))
            
            self.text_widget.configure(state='disabled')
            self.current_section = section_name
            self._update_status(f"Displaying: {section_name}")
            
            # Scroll to top
            self.text_widget.see("1.0")
            
        except Exception as e:
            logger.error(f"Error displaying section: {e}")
            self._update_status("Error displaying section")
    
    def _build_sec_url(self):
        """Build SEC URL"""
        cik = self.filing_info['cik'].lstrip('0')
        acc_no = self.filing_info['accession_num'].replace('-', '')
        doc_name = self.filing_info['doc_name']
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no}/{doc_name}"
    
    def _update_status(self, message):
        """Update status safely"""
        self.window.after(0, lambda: self.status_var.set(message))
    
    def on_section_change(self, event):
        """Handle section selection from combo box"""
        section = self.section_var.get()
        if section:
            self.display_section(section)
    
    def on_section_select(self, event):
        """Handle section selection from tree"""
        selection = self.section_list.selection()
        if selection:
            section = self.section_list.item(selection[0])['text']
            self.section_var.set(section)
            self.display_section(section)

def main():
    logger.info("Starting 10-K Viewer application")
    root = tk.Tk()
    app = MainMenu(root)
    root.mainloop()
    logger.info("Application closed")

if __name__ == "__main__":
    main() 