import requests
import yfinance as yf
import os
import json
import time
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress
from rich.prompt import Prompt
from rich.panel import Panel
import pandas as pd
from time import sleep
import numpy as np
import datetime

# Load environment variables
logger = logging.getLogger(__name__)
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Loading .env file...")
load_dotenv(override=True)  # Add override=True to ensure it loads
logger.debug(f"Environment variables loaded. OPENAI_API_KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

# Menu Configuration
MENU_OPTIONS = {
    1: "Single Company Analysis",
    2: "Batch Analysis",
    3: "Settings",
    4: "Exit"
}

# OpenAI Configuration
def validate_api_key(api_key):
    """Validate OpenAI API key format"""
    if not api_key:
        return False
    
    # Remove any whitespace
    api_key = api_key.strip()
    
    # Check if it starts with the correct prefix
    if not api_key.startswith(('sk-', 'sk-org-')):
        return False
    
    # Check minimum length
    if len(api_key) < 40:  # OpenAI keys are typically longer
        return False
    
    return True

def initialize_openai_client():
    """Initialize OpenAI client with error handling"""
    logger.debug("Initializing OpenAI client...")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        logger.error("API Key not found")
        return None
        
    logger.debug(f"API Key found: {api_key[:10]}...")
    
    try:
        client = OpenAI()
        
        # Test the connection by making a simple API call
        logger.debug("Testing API connection...")
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        logger.debug("API connection successful")
        return client
        
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        return None

# Initialize OpenAI client
try:
    logger.debug("Initializing OpenAI client...")
    client = initialize_openai_client()
    logger.debug("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    exit(1)

# Default model setting
DEFAULT_MODEL = "gpt-4-1106-preview"  # Most capable model for JSON output
AVAILABLE_MODELS = [
    "gpt-4-1106-preview",  # Latest GPT-4 Turbo
    "gpt-4",               # Standard GPT-4
    "gpt-3.5-turbo",      # GPT-3.5 Turbo
]

class Settings:
    def __init__(self):
        self.model = DEFAULT_MODEL
        self.load_settings()
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    data = json.load(f)
                    self.model = data.get('model', DEFAULT_MODEL)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    
    def save_settings(self):
        """Save settings to file"""
        try:
            with open('settings.json', 'w') as f:
                json.dump({
                    'model': self.model
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

# Initialize settings
settings = Settings()

def get_available_models():
    """Get list of available models from OpenAI"""
    try:
        models = client.models.list()
        gpt_models = [model.id for model in models if model.id.startswith(('gpt-3', 'gpt-4'))]
        return sorted(gpt_models)
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return AVAILABLE_MODELS

def display_settings_menu():
    """Display and handle settings menu"""
    while True:
        console.clear()
        console.print(Panel.fit("Settings", style="bold blue"))
        
        # Show current model
        console.print(f"\nCurrent Model: [bold green]{settings.model}[/bold green]")
        
        # Get available models
        try:
            models = get_available_models()
        except Exception as e:
            models = AVAILABLE_MODELS
            console.print(f"[yellow]Using default model list. Error: {e}[/yellow]")
        
        # Display model options
        console.print("\nAvailable Models:")
        for i, model in enumerate(models, 1):
            console.print(f"[bold]{i}[/bold]. {model}")
        
        console.print("\n[bold]0[/bold]. Back to Main Menu")
        
        # Get user choice
        try:
            choice = int(Prompt.ask("\nSelect model", choices=['0'] + [str(x) for x in range(1, len(models) + 1)]))
            
            if choice == 0:
                break
            
            # Update model setting
            settings.model = models[choice - 1]
            settings.save_settings()
            console.print(f"\n[green]Model updated to: {settings.model}[/green]")
            input("\nPress Enter to continue...")
            
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
            input("\nPress Enter to continue...")

# Analysis Configuration
COMPARISON_PROMPT = """Compare these two risk factor sections and identify:
1. New risks introduced
2. Removed risks
3. Modified language that suggests increased concern
4. Changes in specificity or quantitative details
5. Shifts in priority or emphasis

Then analyze the sentiment changes in these modifications:
1. Is the language more urgent/serious?
2. Are descriptions more specific?
3. Are there new quantitative details?
4. Has the tone become more cautious?

Provide the analysis in the following JSON format:
{
    "risk_changes": {
        "new_risks": [],
        "removed_risks": [],
        "modified_risks": {
            "risk_id": {
                "old_text": "",
                "new_text": "",
                "change_type": "increased_concern|decreased_concern|more_specific|less_specific",
                "significance": "high|medium|low"
            }
        }
    },
    "sentiment_analysis": {
        "overall_tone_shift": "",
        "urgency_change": "",
        "specificity_change": "",
        "key_modifications": []
    }
}

First section:
{first_section}

Second section:
{second_section}
"""

# SEC API Configuration
SEC_REQUEST_DELAY = 0.1  # 100ms delay between requests
SEC_CACHE_FILE = 'sec_cik_cache.json'
SEC_CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds
SEC_API_DELAY = 0.1  # 100ms between requests
SEC_MAX_RETRIES = 3  # Maximum number of retries
SEC_BACKOFF_FACTOR = 2  # Exponential backoff factor

def load_sec_config():
    """Load SEC API configuration from file"""
    try:
        if os.path.exists('sec_config.json'):
            with open('sec_config.json', 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading SEC config: {e}")
        return None

# Initialize SEC API headers
config = load_sec_config()
if config:
    user_agent = f"{config['company']} {config['email']} {config['phone']}"
else:
    user_agent = 'Sample Company Name sample@example.com 555-555-5555'  # Default fallback

SEC_API_HEADERS = {
    'User-Agent': user_agent,
    'Accept': 'application/json'
}

def get_cik_from_ticker(ticker):
    """Get CIK number from SEC using company ticker"""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=SEC_API_HEADERS)
        response.raise_for_status()
        
        data = response.json()
        for entry in data.values():
            if entry['ticker'].upper() == ticker.upper():
                return str(entry['cik_str']).zfill(10)
        
        raise ValueError(f"No CIK found for ticker {ticker}")
        
    except Exception as e:
        logger.error(f"Error getting CIK for {ticker}: {e}")
        raise

def get_10k_filings(cik):
    """Get list of 10-K filings for a company"""
    try:
        # Format CIK properly
        cik = str(cik).zfill(10)  # Keep the padded format
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        logger.debug(f"Requesting filings from: {url}")
        logger.debug(f"Using headers: {SEC_API_HEADERS}")
        
        time.sleep(SEC_REQUEST_DELAY)
        response = requests.get(url, headers=SEC_API_HEADERS)
        
        if response.status_code == 429:  # Rate limit hit
            logger.warning("SEC rate limit hit, waiting before retry")
            time.sleep(SEC_API_DELAY * SEC_BACKOFF_FACTOR)
            response = requests.get(url, headers=SEC_API_HEADERS)
        
        response.raise_for_status()
        data = response.json()
        
        # Extract 10-K filings
        result = []
        filings = data.get('filings', {})
        recent = filings.get('recent', {})
        
        if not recent:
            logger.warning("No filings found in response")
            return []
        
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        docs = recent.get('primaryDocument', [])
        
        for i in range(len(forms)):
            if forms[i] == '10-K':
                result.append({
                    'accession_number': accessions[i],
                    'filing_date': dates[i],
                    'primary_document': docs[i]
                })
        
        logger.debug(f"Found {len(result)} 10-K filings in response")
        return result
        
    except Exception as e:
        logger.error(f"Error getting filings for CIK {cik}: {e}")
        raise

def download_10k(cik, accession_num, doc_name):
    """Download a 10-K filing"""
    try:
        # Remove leading zeros from CIK for URL
        clean_cik = str(int(cik))  # Remove leading zeros
        clean_accession = accession_num.replace('-', '')
        
        url = f"https://www.sec.gov/Archives/edgar/data/{clean_cik}/{clean_accession}/{doc_name}"
        logger.info(f"Downloading filing from: {url}")
        
        time.sleep(SEC_REQUEST_DELAY)
        response = requests.get(url, headers=SEC_API_HEADERS)
        
        if response.status_code == 429:  # Rate limit hit
            logger.warning("SEC rate limit hit, waiting before retry")
            time.sleep(SEC_API_DELAY * SEC_BACKOFF_FACTOR)
            response = requests.get(url, headers=SEC_API_HEADERS)
            
        response.raise_for_status()
        content = response.text
        logger.info(f"Successfully downloaded filing ({len(content)} bytes)")
        return content
        
    except Exception as e:
        logger.error(f"Error downloading 10-K: {e}")
        raise

def save_filing(content, ticker, filing_date, raw_dir):
    """Save filing content to file"""
    try:
        # Create ticker-specific directory
        ticker_dir = raw_dir / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with ticker and date
        filename = f"{ticker.upper()}_10K_{filing_date}.txt"
        filepath = ticker_dir / filename
        
        logger.info(f"Saving filing to {filepath}")
        
        # Check if file already exists
        if filepath.exists():
            logger.info(f"Filing already exists at {filepath}")
            return filepath
        
        # Save content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Successfully saved filing ({len(content)} bytes) to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving filing: {e}")
        raise

def clean_html_content(content):
    """Clean HTML/XML content and extract text"""
    try:
        # First try with html.parser
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted tags
        for tag in soup.find_all(['script', 'style', 'ix:header', 'ix:hidden']):
            tag.decompose()
            
        # Replace line breaks and paragraphs with newlines
        for br in soup.find_all(['br', 'p']):
            br.replace_with('\n' + br.get_text() + '\n')
            
        # Handle tables
        for table in soup.find_all('table'):
            # Convert table to text representation
            rows_text = []
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                row_text = ' | '.join(cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True))
                if row_text:
                    rows_text.append(row_text)
            if rows_text:
                table.replace_with('\n'.join(rows_text) + '\n')
            
        # Get text content
        text = soup.get_text(separator=' ')
        
        # Clean up whitespace and special characters
        text = re.sub(r'\s*\n\s*', '\n', text)  # Clean up newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize horizontal whitespace
        text = re.sub(r'[^\S\n]+', ' ', text)  # Keep newlines but clean other whitespace
        
        # Remove XBRL namespace declarations and other XML artifacts
        text = re.sub(r'xmlns[^>]*>', '>', text)
        text = re.sub(r'<[^>]*>', '', text)
        
        # Clean up any remaining special characters
        text = text.replace('\u00a0', ' ')  # Replace non-breaking spaces
        text = text.replace('\r', '\n')     # Normalize line endings
        
        # Final whitespace cleanup
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)
        
        return text
        
    except Exception as e:
        logger.error(f"Error cleaning HTML content: {e}")
        # Try alternate parser if html.parser fails
        try:
            soup = BeautifulSoup(content, 'lxml')
            text = soup.get_text(separator='\n')
            return clean_html_content(text)  # Recursively clean the extracted text
        except:
            logger.error("Both HTML parsers failed, trying basic cleanup")
            # Basic cleanup as last resort
            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

def extract_risk_section(text):
    """Extract risk factors section from cleaned text"""
    try:
        # Common patterns for risk section start
        start_patterns = [
            r"Item\s*1A\.?\s*Risk\s*Factors",
            r"ITEM\s*1A\.?\s*RISK\s*FACTORS",
            r"RISK\s*FACTORS",
            r"Item\s*1A\.",  # Shorter version
            r"ITEM\s*1A\."   # Shorter version
        ]
        
        # Common patterns for risk section end (next major section)
        end_patterns = [
            r"Item\s*1B\.?",
            r"ITEM\s*1B\.?",
            r"Item\s*2\.?",
            r"ITEM\s*2\.?",
            r"UNRESOLVED\s+STAFF\s+COMMENTS",  # Common next section
            r"PROPERTIES"  # Common next section
        ]
        
        # Find start of risk section
        start_pos = -1
        start_pattern_found = None
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                start_pattern_found = match.group()
                break
                
        if start_pos == -1:
            logger.warning("Could not find start of risk factors section")
            return None
            
        # Find end of risk section
        end_pos = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text[start_pos:], re.IGNORECASE)
            if match:
                end_pos = start_pos + match.start()
                break
                
        # Extract risk section
        risk_section = text[start_pos:end_pos].strip()
        
        # Additional cleaning
        # Remove table alignment artifacts
        risk_section = re.sub(r'ALIGN="[^"]*"', '', risk_section)
        risk_section = re.sub(r'VALIGN="[^"]*"', '', risk_section)
        
        # Clean up whitespace
        risk_section = re.sub(r'\s+', ' ', risk_section)  # Normalize whitespace
        risk_section = re.sub(r'\n\s*\n', '\n\n', risk_section)  # Normalize paragraphs
        
        # Add section header if missing
        if not risk_section.startswith("Item 1A") and not risk_section.startswith("ITEM 1A"):
            risk_section = f"Item 1A. Risk Factors\n\n{risk_section}"
        
        logger.info(f"Found risk section starting with pattern: {start_pattern_found}")
        return risk_section
        
    except Exception as e:
        logger.error(f"Error extracting risk section: {e}")
        raise

def save_clean_filing(content, ticker, filing_date):
    """Save cleaned filing content"""
    try:
        # Create clean directory structure
        clean_dir = Path('clean')
        ticker_dir = clean_dir / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{ticker.upper()}_10K_{filing_date}_RISK.txt"
        filepath = ticker_dir / filename
        
        logger.info(f"Saving cleaned filing to {filepath}")
        
        # Save content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Successfully saved cleaned filing ({len(content)} bytes)")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving cleaned filing: {e}")
        raise

def process_filing(raw_path):
    """Process a single filing to extract risk factors"""
    try:
        logger.info(f"Processing filing: {raw_path}")
        
        # Read raw content
        with open(raw_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Clean HTML/XML
        logger.info("Cleaning HTML/XML content")
        cleaned_text = clean_html_content(content)
        
        # Extract risk section
        logger.info("Extracting risk factors section")
        risk_section = extract_risk_section(cleaned_text)
        
        if not risk_section:
            logger.warning("No risk factors section found")
            return None
            
        # Get ticker and date from filename
        filename = raw_path.name
        parts = filename.split('_')
        ticker = parts[0]
        filing_date = parts[2].replace('.txt', '')
        
        # Save cleaned content
        return save_clean_filing(risk_section, ticker, filing_date)
        
    except Exception as e:
        logger.error(f"Error processing filing {raw_path}: {e}")
        raise

def display_menu():
    """Display main menu and get user choice"""
    console.clear()
    console.print(Panel.fit("10-K Risk Factor Analysis Tool", style="bold blue"))
    
    # Display menu options
    for key, value in MENU_OPTIONS.items():
        console.print(f"[bold]{key}[/bold]. {value}")
    
    # Get user choice
    try:
        choice = int(Prompt.ask("\nSelect an option", choices=[str(x) for x in MENU_OPTIONS.keys()]))
        return choice
    except ValueError:
        console.print("[red]Invalid input. Please enter a number.[/red]")
        input("\nPress Enter to continue...")
        return None

def get_ticker_input():
    """Get and validate ticker symbol from user"""
    while True:
        ticker = Prompt.ask("\nEnter ticker symbol").upper()
        try:
            # Verify ticker exists using yfinance
            stock = yf.Ticker(ticker)
            # Try to get company info to validate ticker
            info = stock.info
            if info and 'symbol' in info:
                return ticker
            else:
                console.print(f"[red]Invalid ticker symbol: {ticker}[/red]")
        except Exception as e:
            console.print(f"[red]Error validating ticker: {ticker}[/red]")
            logger.error(f"Ticker validation error: {e}")

def get_available_risk_files(ticker):
    """Get list of available risk files for a ticker"""
    clean_dir = Path(f"clean/{ticker}")
    if not clean_dir.exists():
        return []
    
    # Look for files matching pattern TICKER_10K_YYYY-MM-DD_RISK.txt
    pattern = f"{ticker}_10K_*_RISK.txt"
    risk_files = list(clean_dir.glob(pattern))
    
    # Sort by date (newest first)
    risk_files.sort(reverse=True)
    return risk_files

def display_available_years(risk_files):
    """Display available years and return mapping of choice to file"""
    if not risk_files:
        return None
    
    console.print("\n[bold]Available 10-K Risk Sections:[/bold]")
    file_map = {}
    
    for idx, file in enumerate(risk_files, 1):
        # Extract date from filename
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file.name)
        if date_match:
            date = date_match.group(1)
            console.print(f"[bold]{idx}[/bold]. {date}")
            file_map[idx] = file
    
    return file_map

def get_year_selection(file_map, prompt_text):
    """Get user's year selection"""
    if not file_map:
        return None
        
    while True:
        try:
            choice = int(Prompt.ask(
                prompt_text,
                choices=[str(x) for x in file_map.keys()]
            ))
            return file_map[choice]
        except (ValueError, KeyError):
            console.print("[red]Invalid selection. Please try again.[/red]")

def load_risk_section(file_path):
    """Load and clean risk section from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content
    except Exception as e:
        logger.error(f"Error loading risk section from {file_path}: {e}")
        return None

def prepare_risk_comparison(recent_file, earlier_file):
    """Prepare risk sections for comparison"""
    # Load both risk sections
    recent_text = load_risk_section(recent_file)
    earlier_text = load_risk_section(earlier_file)
    
    if not recent_text or not earlier_text:
        return None, None
    
    # Extract years from filenames for reference
    recent_year = re.search(r'(\d{4})-\d{2}-\d{2}', recent_file.name).group(1)
    earlier_year = re.search(r'(\d{4})-\d{2}-\d{2}', earlier_file.name).group(1)
    
    return {
        'year': recent_year,
        'text': recent_text
    }, {
        'year': earlier_year,
        'text': earlier_text
    }

def get_market_data(ticker, filing_date):
    """Get comprehensive market data with validation"""
    try:
        # Initialize validator
        validator = DataValidator()
        
        # Add rate limiting
        sleep(0.5)  # 500ms delay between requests
        
        stock = yf.Ticker(ticker)
        spy = yf.Ticker("SPY")  # Get SPY data for comparison
        
        # Convert and validate filing date
        try:
            filing_date = pd.to_datetime(filing_date)
            if pd.isnull(filing_date):
                raise ValueError("Invalid filing date")
            # Ensure timezone is set to market timezone
            filing_date = filing_date.tz_localize('America/New_York')
        except Exception as e:
            logger.error(f"Date conversion error: {e}")
            return None
            
        # Get data for 90 days before and after filing
        start_date = filing_date - pd.Timedelta(days=90)
        end_date = filing_date + pd.Timedelta(days=90)
        
        logger.info(f"Fetching market data for {ticker} from {start_date} to {end_date}")
        
        # Get historical data with retries
        hist = None
        spy_hist = None
        for attempt in range(3):
            try:
                hist = stock.history(start=start_date, end=end_date, interval='1d')
                spy_hist = spy.history(start=start_date, end=end_date, interval='1d')
                if not hist.empty and not spy_hist.empty:
                    break
                sleep(1)  # Wait before retry
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                sleep(1)
        
        if hist is None or hist.empty or spy_hist is None or spy_hist.empty:
            logger.error(f"No historical data found for {ticker} or SPY")
            return None
            
        # Find closest trading day to filing date
        filing_idx = hist.index[hist.index.get_indexer([filing_date], method='nearest')[0]]
        logger.info(f"Using {filing_idx.date()} as filing date reference")
        
        # Split data into pre and post periods
        pre_period = hist.loc[:filing_idx]
        post_period = hist.loc[filing_idx:]
        spy_pre = spy_hist.loc[:filing_idx]
        spy_post = spy_hist.loc[filing_idx:]
        
        # Calculate volatility
        def calculate_volatility(data):
            returns = data['Close'].pct_change().dropna()
            return returns.std() * np.sqrt(252) * 100  # Annualized volatility in percentage
        
        pre_volatility = calculate_volatility(pre_period)
        post_volatility = calculate_volatility(post_period)
        
        # Calculate basic metrics
        pre_price_mean = pre_period['Close'].mean()
        pre_volume_mean = pre_period['Volume'].mean()
        post_price_mean = post_period['Close'].mean()
        post_volume_mean = post_period['Volume'].mean()
        
        # Calculate SPY relative performance
        spy_pre_return = ((spy_pre['Close'].iloc[-1] / spy_pre['Close'].iloc[0]) - 1) * 100
        spy_post_return = ((spy_post['Close'].iloc[-1] / spy_post['Close'].iloc[0]) - 1) * 100
        stock_pre_return = ((pre_period['Close'].iloc[-1] / pre_period['Close'].iloc[0]) - 1) * 100
        stock_post_return = ((post_period['Close'].iloc[-1] / post_period['Close'].iloc[0]) - 1) * 100
        
        # Calculate relative performance
        spy_relative_pre = stock_pre_return - spy_pre_return
        spy_relative_post = stock_post_return - spy_post_return
        
        market_data = {
            'pre_disclosure': {
                'price': pre_price_mean,
                'volume': pre_volume_mean,
                'volatility': pre_volatility,
                'spy_relative': spy_relative_pre,
                'trading_days': len(pre_period)
            },
            'post_disclosure': {
                'price_change': ((post_price_mean - pre_price_mean) / pre_price_mean * 100) if pre_price_mean > 0 else 0,
                'volume_change': ((post_volume_mean - pre_volume_mean) / pre_volume_mean * 100) if pre_volume_mean > 0 else 0,
                'volatility_change': post_volatility - pre_volatility,
                'spy_relative_post': spy_relative_post,
                'trading_days': len(post_period)
            }
        }
        
        # Validate market data
        is_valid, validation_results = validator.validate_market_data(market_data, ticker)
        if not is_valid:
            logger.warning(f"Market data validation failed for {ticker}: {validation_results}")
            
        market_data['validation'] = validation_results
        return market_data
        
    except Exception as e:
        logger.error(f"Error in get_market_data: {e}")
        return None

def analyze_risk_sections(recent_data, earlier_data, client, ticker):
    """Analyze risk sections with improved data collection and validation"""
    try:
        # Initialize progress tracking
        analysis_steps = {
            'company_info': False,
            'market_data': False,
            'institutional_data': False,
            'risk_analysis': False
        }
        
        # Get company info
        logger.info(f"Collecting company info for {ticker}")
        company_info = get_company_info(ticker)
        analysis_steps['company_info'] = bool(company_info)
        
        # Get market performance data
        logger.info(f"Calculating market performance for {ticker}")
        market_performance = get_market_data(ticker, recent_data['year'])
        analysis_steps['market_data'] = bool(market_performance)
        
        # Get institutional data
        logger.info(f"Collecting institutional data for {ticker}")
        institutional_data = get_institutional_data(ticker)
        analysis_steps['institutional_data'] = bool(institutional_data)
        
        # Log collection results
        logger.info(f"Data collection results for {ticker}:")
        for step, success in analysis_steps.items():
            logger.info(f"- {step}: {'Success' if success else 'Failed'}")
        
        # Combine all market-related data
        market_data = {
            'company_info': company_info or {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'company_name': ticker
            },
            'institutional_metrics': institutional_data or {
                'holdings_pct': 0,
                'institutions_count': 0,
                'holdings_change': 0
            },
            **(market_performance or {
                'pre_disclosure': {'price': 0, 'volume': 0, 'volatility': 0},
                'post_disclosure': {'price_change': 0, 'volume_change': 0, 'volatility_change': 0}
            })
        }
        
        # Prepare GPT analysis prompt with market context
        system_prompt = """You are a financial risk analyst specializing in SEC filings and market impact analysis.
        Analyze the provided risk sections and market data to generate a comprehensive analysis that matches this structure:
        {
            "market_performance": {
                "pre_filing": {
                    "price_vs_spy": "percentage relative to SPY",
                    "average_volume": "numeric value",
                    "volatility": "numeric value"
                },
                "post_filing": {
                    "price_change": "percentage",
                    "volume_change": "percentage",
                    "spy_relative": "percentage",
                    "volatility_change": "percentage points"
                }
            },
            "risk_changes": [
                {
                    "type": "NEW|MODIFIED|REMOVED",
                    "category": "operational|financial|market|regulatory|technological",
                    "description": "specific risk description",
                    "significance": "high|medium|low",
                    "market_impact": {
                        "magnitude": "high (>15%)|medium (5-15%)|low (<5%)",
                        "timing": "immediate|gradual|delayed",
                        "volume_pattern": "spike|sustained|normal"
                    }
                }
            ],
            "validation_metrics": {
                "materialization_rate": "percentage",
                "false_positive_rate": "percentage",
                "average_impact": "percentage"
            }
        }"""
        
        user_prompt = f"""Analyze these risk sections and market data for {ticker}:

        Company: {ticker}
        Sector: {market_data['company_info']['sector']}
        Industry: {market_data['company_info']['industry']}
        Market Cap: ${market_data['company_info']['market_cap']:,.0f}

        Market Performance:
        Pre-Filing Metrics:
        - Price vs SPY: {market_data['pre_disclosure']['spy_relative']:.2f}%
        - Average Volume: {market_data['pre_disclosure']['volume']:,.0f}
        - Volatility: {market_data['pre_disclosure']['volatility']:.2f}%

        Post-Filing Changes:
        - Price Change: {market_data['post_disclosure']['price_change']:.2f}%
        - Volume Change: {market_data['post_disclosure']['volume_change']:.2f}%
        - Post-Filing vs SPY: {market_data['post_disclosure']['spy_relative_post']:.2f}%
        - Volatility Change: {market_data['post_disclosure']['volatility_change']:.2f}%
        - Institutional Holdings: {market_data['institutional_metrics']['holdings_pct']:.1f}%

        Recent ({recent_data['year']}):
        {recent_data['text']}

        Earlier ({earlier_data['year']}):
        {earlier_data['text']}

        Use the EXACT numerical values provided above in your market_performance section. Format them as follows:
        - price_vs_spy: Include the exact Pre-Filing Price vs SPY value with % symbol
        - average_volume: Include the exact Pre-Filing Average Volume as a number
        - volatility: Include the exact Pre-Filing Volatility value with % symbol
        - price_change: Include the exact Post-Filing Price Change with % symbol
        - volume_change: Include the exact Post-Filing Volume Change with % symbol
        - spy_relative: Include the exact Post-Filing vs SPY value with % symbol
        - volatility_change: Include the exact Volatility Change value with % symbol

        Focus on significant changes with clear market implications and ensure the response matches the exact JSON structure specified."""
        
        # Get GPT analysis
        logger.info("Requesting GPT analysis")
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # Parse GPT response
        try:
            gpt_analysis = json.loads(response.choices[0].message.content)
            analysis_steps['risk_analysis'] = True
        except Exception as e:
            logger.error(f"Error parsing GPT response: {e}")
            gpt_analysis = {
                "market_performance": {
                    "pre_filing": {"price_vs_spy": "0%", "average_volume": "0", "volatility": "0"},
                    "post_filing": {"price_change": "0%", "volume_change": "0%", "spy_relative": "0%", "volatility_change": "0"}
                },
                "risk_changes": [],
                "validation_metrics": {
                    "materialization_rate": "0%",
                    "false_positive_rate": "0%",
                    "average_impact": "0%"
                }
            }
        
        # Combine all analysis components
        analysis = {
            'market_metrics': market_data,
            'market_performance': gpt_analysis.get('market_performance', {}),
            'risk_changes': gpt_analysis.get('risk_changes', []),
            'validation_metrics': gpt_analysis.get('validation_metrics', {}),
            'analysis_status': {
                'steps_completed': analysis_steps,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in risk analysis for {ticker}: {e}")
        return None

def display_analysis_results(analysis, ticker, recent_year, earlier_year):
    """Display analysis results with improved formatting"""
    try:
        console.clear()
        console.print(Panel.fit(f"Risk Analysis: {ticker} ({recent_year} vs {earlier_year})", style="bold blue"))
        
        # Market Performance Analysis
        market_perf = analysis.get('market_performance', {})
        pre_filing = market_perf.get('pre_filing', {})
        post_filing = market_perf.get('post_filing', {})
        
        console.print("\n[bold]Market Performance Analysis[/bold]")
        market_panel = Panel.fit(
            "[bold]Pre-Filing Performance[/bold]\n"
            f"• Price vs SPY: {pre_filing.get('price_vs_spy', '0%')}\n"
            f"• Average Volume: {pre_filing.get('average_volume', '0')}\n"
            f"• Volatility: {pre_filing.get('volatility', '0')}\n\n"
            "[bold]Post-Filing Changes[/bold]\n"
            f"• Price Change: {post_filing.get('price_change', '0%')}\n"
            f"• Volume Change: {post_filing.get('volume_change', '0%')}\n"
            f"• SPY Relative: {post_filing.get('spy_relative', '0%')}\n"
            f"• Volatility Change: {post_filing.get('volatility_change', '0')}"
        )
        console.print(market_panel)
        
        # Risk Changes
        console.print("\n[bold]Risk Changes[/bold]")
        if 'risk_changes' in analysis:
            for change in analysis['risk_changes']:
                impact = change.get('market_impact', {})
                change_panel = Panel.fit(
                    f"[bold]{change.get('type', 'UNKNOWN')}[/bold]: {change.get('description', '')}\n"
                    f"Category: {change.get('category', 'unknown')}\n"
                    f"Significance: {change.get('significance', 'unknown')}\n\n"
                    "[bold]Market Impact[/bold]\n"
                    f"• Expected Magnitude: {impact.get('magnitude', 'unknown')}\n"
                    f"• Timing: {impact.get('timing', 'unknown')}\n"
                    f"• Volume Pattern: {impact.get('volume_pattern', 'unknown')}"
                )
                console.print(change_panel)
        
        # Validation Metrics
        validation = analysis.get('validation_metrics', {})
        if validation:
            console.print("\n[bold]Validation Metrics[/bold]")
            validation_panel = Panel.fit(
                f"• Risk Materialization Rate: {validation.get('materialization_rate', '0%')}\n"
                f"• False Positive Rate: {validation.get('false_positive_rate', '0%')}\n"
                f"• Average Market Impact: {validation.get('average_impact', '0%')}"
            )
            console.print(validation_panel)
        
    except Exception as e:
        logger.error(f"Error displaying analysis results: {e}")
        console.print(f"[red]Error displaying results: {str(e)}[/red]")

def get_magnitude_description(significance):
    """Convert significance to magnitude description"""
    mapping = {
        'high': 'high (>15%)',
        'medium': 'medium (5-15%)',
        'low': 'low (<5%)'
    }
    return mapping.get(significance, 'unknown')

def save_analysis_results(analysis, ticker, recent_year, earlier_year):
    """Save analysis results to file"""
    analysis_dir = Path(f"analysis/single/{ticker}")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{ticker}_analysis_{recent_year}_vs_{earlier_year}.json"
    filepath = analysis_dir / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'ticker': ticker,
                'recent_year': recent_year,
                'earlier_year': earlier_year,
                'analysis': analysis,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)
        return filepath
    except Exception as e:
        logger.error(f"Error saving analysis results: {e}")
        return None

def single_company_analysis():
    """Handle single company analysis flow"""
    console.clear()
    console.print(Panel.fit("Single Company Analysis", style="bold blue"))
    
    # Get ticker input
    ticker = get_ticker_input()
    if not ticker:
        console.print("[red]Invalid ticker. Returning to main menu.[/red]")
        return
    
    console.print(f"\n[green]Analyzing {ticker}...[/green]")
    
    # Get available risk files
    risk_files = get_available_risk_files(ticker)
    if not risk_files:
        console.print(f"\n[yellow]No risk files found for {ticker}[/yellow]")
        console.print("[yellow]Downloading and processing 10-K filings...[/yellow]")
        
        try:
            # Get CIK
            console.print(f"[blue]Getting CIK for {ticker}...[/blue]")
            cik = get_cik_from_ticker(ticker)
            
            # Get 10-K filings
            console.print("[blue]Fetching 10-K filings...[/blue]")
            filings = get_10k_filings(cik)
            
            if not filings:
                console.print("[red]No 10-K filings found[/red]")
                input("\nPress Enter to continue...")
                return
            
            # Create directories
            raw_dir = Path('raw')
            raw_dir.mkdir(exist_ok=True)
            
            # Download and process each filing
            for filing in filings:
                filing_date = filing['filing_date']
                console.print(f"[blue]Processing 10-K from {filing_date}[/blue]")
                
                # Download filing
                content = download_10k(cik, filing['accession_number'], filing['primary_document'])
                raw_path = save_filing(content, ticker, filing_date, raw_dir)
                
                # Process the filing
                if raw_path:
                    process_filing(raw_path)
                
                time.sleep(SEC_REQUEST_DELAY)  # Rate limiting
            
            # Refresh risk files list
            risk_files = get_available_risk_files(ticker)
            if not risk_files:
                console.print("[red]Failed to process any risk files[/red]")
                input("\nPress Enter to continue...")
                return
                
        except Exception as e:
            logger.error(f"Error downloading/processing filings: {e}")
            console.print(f"[red]Error: {str(e)}[/red]")
            input("\nPress Enter to continue...")
            return
    
    # Check if we have enough years of data
    if len(risk_files) < 2:
        console.print(f"\n[red]Need at least 2 years of data for {ticker}[/red]")
        console.print("[yellow]Please download more 10-K filings.[/yellow]")
        input("\nPress Enter to continue...")
        return
    
    # Display available years and get selection
    file_map = display_available_years(risk_files)
    if not file_map:
        console.print("[red]Error processing available years.[/red]")
        input("\nPress Enter to continue...")
        return
    
    # Get first year selection
    console.print("\n[bold]Select the more recent year to analyze:[/bold]")
    recent_file = get_year_selection(file_map, "\nSelect recent year")
    if not recent_file:
        console.print("[red]Invalid selection. Returning to main menu.[/red]")
        return
    
    # Remove selected year from options
    selected_idx = [k for k, v in file_map.items() if v == recent_file][0]
    del file_map[selected_idx]
    
    # Get second year selection
    console.print("\n[bold]Select the earlier year to compare against:[/bold]")
    earlier_file = get_year_selection(file_map, "\nSelect earlier year")
    if not earlier_file:
        console.print("[red]Invalid selection. Returning to main menu.[/red]")
        return
        
    console.print(f"\n[green]Selected years to compare:[/green]")
    console.print(f"Recent : {recent_file.name}")
    console.print(f"Earlier: {earlier_file.name}")
    
    # Load and prepare risk sections
    console.print("\n[bold]Loading risk sections...[/bold]")
    recent_data, earlier_data = prepare_risk_comparison(recent_file, earlier_file)
    
    if not recent_data or not earlier_data:
        console.print("[red]Error loading risk sections. Please try again.[/red]")
        input("\nPress Enter to continue...")
        return
    
    # Prepare for analysis
    console.print(f"\n[green]Successfully loaded risk sections:[/green]")
    console.print(f"Recent ({recent_data['year']}): {len(recent_data['text'])} characters")
    console.print(f"Earlier ({earlier_data['year']}): {len(earlier_data['text'])} characters")
    
    # Analyze risk sections
    with Progress() as progress:
        task = progress.add_task("[cyan]Analyzing risk sections...", total=1)
        analysis = analyze_risk_sections(recent_data, earlier_data, client, ticker)
        progress.update(task, advance=1)
    
    if not analysis:
        console.print("[red]Error analyzing risk sections. Please try again.[/red]")
        input("\nPress Enter to continue...")
        return
    
    # Save results
    filepath = save_analysis_results(analysis, ticker, recent_data['year'], earlier_data['year'])
    if filepath:
        console.print(f"\n[green]Analysis saved to: {filepath}[/green]")
    
    # Display results
    display_analysis_results(analysis, ticker, recent_data['year'], earlier_data['year'])
    
    input("\nPress Enter to return to main menu...")

def download_10k_filings(ticker):
    """Download 10-K filings for a company"""
    try:
        # Create directories if they don't exist
        raw_dir = Path(f"raw/{ticker}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[blue]Created directory: {raw_dir}[/blue]")
        
        # Load SEC config
        with open('sec_config.json', 'r') as f:
            sec_config = json.load(f)
        
        # Define headers required by SEC with proper user agent
        user_agent = f"{sec_config['company']} {sec_config['email']}"
        headers = {
            'User-Agent': user_agent,
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Host': 'www.sec.gov'
        }
        
        console.print(f"[blue]Using SEC credentials: {user_agent}[/blue]")
        
        # Get the last 2 years of filings
        console.print(f"[blue]Fetching SEC filings for {ticker}...[/blue]")
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&count=2"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract and download filings
        filing_links = soup.find_all('a', {'href': re.compile(r'Archives/edgar/data')})
        if not filing_links:
            console.print(f"[yellow]No 10-K filings found for {ticker}[/yellow]")
            return
            
        console.print(f"[green]Found {len(filing_links)} filings for {ticker}[/green]")
        
        successful_downloads = 0
        for link in filing_links:
            try:
                filing_url = f"https://www.sec.gov{link['href']}"
                console.print(f"[blue]Downloading filing from: {filing_url}[/blue]")
                
                # Extract date from URL
                date_match = re.search(r'/(\d{8})/', filing_url)
                if date_match:
                    filing_date = datetime.datetime.strptime(date_match.group(1), '%Y%m%d').strftime('%Y-%m-%d')
                else:
                    filing_date = time.strftime('%Y-%m-%d')
                
                # Download and save filing
                content = download_filing(filing_url, ticker, raw_dir, headers)
                if content:
                    # Save raw filing
                    raw_file = raw_dir / f"{ticker}_10K_{filing_date}.txt"
                    with open(raw_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Extract and save risk section
                    risk_section = extract_risk_factors(content)
                    if risk_section:
                        clean_dir = Path(f"clean/{ticker}")
                        clean_dir.mkdir(parents=True, exist_ok=True)
                        
                        risk_file = clean_dir / f"{ticker}_10K_{filing_date}_RISK.txt"
                        with open(risk_file, 'w', encoding='utf-8') as f:
                            f.write(risk_section)
                        
                        successful_downloads += 1
                        console.print(f"[green]Successfully processed {raw_file.name}[/green]")
                    else:
                        console.print(f"[yellow]Could not extract risk section from {raw_file.name}[/yellow]")
                
                # Rate limiting
                sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing filing {filing_url}: {e}")
                console.print(f"[red]Error processing filing: {e}[/red]")
                continue
        
        return successful_downloads > 0
        
    except Exception as e:
        logger.error(f"Error downloading filings for {ticker}: {e}")
        console.print(f"[red]Error downloading filings: {e}[/red]")
        return False

def download_filing(filing_url, ticker, raw_dir, headers):
    """Download a single filing with support for multiple formats and fallback methods"""
    try:
        # First try: Get the index page
        response = requests.get(filing_url, headers=headers)
        if response.status_code == 200:
            # Parse the index page to find the actual document link
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Method 1: Look for the primary document table (newer filings)
            doc_link = None
            table = soup.find('table', class_='tableFile')
            if table:
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        doc_type = cols[3].text.strip()
                        if doc_type == '10-K':
                            doc_href = cols[2].find('a')['href'] if cols[2].find('a') else None
                            if doc_href:
                                doc_link = f"https://www.sec.gov{doc_href}" if not doc_href.startswith('http') else doc_href
                                break
            
            # Method 2: Look for .htm files that aren't index files (older filings)
            if not doc_link:
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.endswith('.htm') and not href.endswith('-index.htm'):
                        doc_link = f"https://www.sec.gov{href}" if not href.startswith('http') else href
                        break
            
            if doc_link:
                # Download the actual document with multiple format support
                sleep(0.1)  # Rate limiting
                logger.info(f"Downloading document from: {doc_link}")
                
                # For iXBRL documents, we need to modify the URL to get the raw HTML
                if '/ix?doc=/' in doc_link:
                    raw_doc_link = doc_link.replace('/ix?doc=/', '/')
                    logger.info(f"Converting iXBRL URL to raw document URL: {raw_doc_link}")
                    doc_response = requests.get(raw_doc_link, headers=headers)
                else:
                    doc_response = requests.get(doc_link, headers=headers)
                
                if doc_response.status_code == 200:
                    content_type = doc_response.headers.get('content-type', '').lower()
                    content = None
                    
                    # Handle different content types
                    if 'html' in content_type or 'xml' in content_type:
                        # Parse with appropriate parser
                        try:
                            if 'xml' in content_type:
                                soup = BeautifulSoup(doc_response.text, 'xml')
                            else:
                                soup = BeautifulSoup(doc_response.text, 'html.parser')
                            
                            # Remove XBRL-specific tags but keep their content
                            for tag in soup.find_all(['ix:header', 'ix:hidden', 'ix:footnote']):
                                tag.unwrap()  # Keep content but remove tag
                            
                            # Convert remaining XBRL tags to their content
                            for tag in soup.find_all(lambda t: t.name.startswith('ix:')):
                                tag.unwrap()
                            
                            # Get text content
                            content = str(soup)
                            
                            # Clean up any remaining XML artifacts
                            content = re.sub(r'xmlns[^>]*>', '>', content)
                            
                            logger.info(f"Successfully processed document ({len(content)} bytes)")
                            
                        except Exception as e:
                            logger.warning(f"Error parsing document: {e}, trying basic HTML parsing")
                            content = doc_response.text
                    else:
                        logger.warning(f"Unknown content type: {content_type}, using raw content")
                        content = doc_response.text
                    
                    if content and len(content.strip()) > 0:
                        # Extract date from URL
                        date_match = re.search(r'/(\d{8})/', filing_url)
                        filing_date = datetime.datetime.strptime(date_match.group(1), '%Y%m%d').strftime('%Y-%m-%d') if date_match else time.strftime('%Y-%m-%d')
                        
                        # Save the content
                        output_file = raw_dir / f"{ticker}_10K_{filing_date}.txt"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        logger.info(f"Successfully saved filing to {output_file}")
                        return content
                    else:
                        logger.warning("No content extracted from document")
                else:
                    logger.error(f"Failed to download document: {doc_response.status_code}")
            else:
                logger.warning("Could not find document link in index page")
        else:
            logger.error(f"Failed to download filing: {response.status_code}")
            
        return None
            
    except Exception as e:
        logger.error(f"Error downloading filing {filing_url}: {e}")
        return None

def extract_risk_sections(ticker):
    """Extract risk sections from 10-K filings"""
    try:
        raw_dir = Path(f"raw/{ticker}")
        working_dir = Path(f"working/{ticker}")
        working_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[blue]Processing raw filings for {ticker}...[/blue]")
        
        for filing in raw_dir.glob("*10-K*.txt"):
            console.print(f"[blue]Extracting risk section from {filing.name}[/blue]")
            
            # Read and extract risk section
            with open(filing, 'r', encoding='utf-8') as f:
                content = f.read()
                console.print(f"[blue]File size: {len(content)} bytes[/blue]")
                risk_section = extract_risk_factors(content)
                
            if risk_section:
                console.print(f"[green]Found risk section ({len(risk_section)} bytes)[/green]")
                # Save to working directory
                working_file = working_dir / f"{ticker}_10K_{filing_date}_RISK.txt"
                with open(working_file, 'w', encoding='utf-8') as f:
                    f.write(risk_section)
                console.print(f"[green]Saved risk section to: {working_file}[/green]")
            else:
                console.print(f"[yellow]No risk section found in {filing.name}[/yellow]")
                logger.warning(f"No risk section found in {filing.name}")
                    
    except Exception as e:
        logger.error(f"Error extracting risk sections for {ticker}: {e}")
        raise

def clean_risk_sections(ticker):
    """Clean and prepare risk sections"""
    try:
        working_dir = Path(f"working/{ticker}")
        clean_dir = Path(f"clean/{ticker}")
        clean_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[blue]Cleaning risk sections for {ticker}...[/blue]")
        
        for risk_file in working_dir.glob("*_RISK.txt"):
            console.print(f"[blue]Processing {risk_file.name}[/blue]")
            
            with open(risk_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Clean the content
            cleaned = clean_text(content)
            
            # Save cleaned content to clean directory
            clean_file = clean_dir / risk_file.name
            with open(clean_file, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            console.print(f"[green]Saved cleaned file to: {clean_file}[/green]")
                
    except Exception as e:
        logger.error(f"Error cleaning risk sections for {ticker}: {e}")
        raise

def extract_risk_factors(content):
    """Extract risk factors section from 10-K content"""
    try:
        # Common patterns for risk section headers and endings
        start_patterns = [
            r"Item\s*1A\.?\s*Risk\s*Factors",
            r"ITEM\s*1A\.?\s*RISK\s*FACTORS",
            r"Risk\s*Factors"
        ]
        
        end_patterns = [
            r"Item\s*1B\.?",
            r"ITEM\s*1B\.?",
            r"Item\s*2\.?",
            r"ITEM\s*2\.?",
            r"Unresolved\s+Staff\s+Comments",
            r"Properties"
        ]
        
        # Find start position
        start_pos = -1
        start_pattern_found = None
        content_lower = content.lower()
        
        for pattern in start_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                # Take the last match as it's more likely to be the main section
                match = matches[-1]
                curr_pos = match.start()
                # Only update if we haven't found a position or if this one is earlier
                if start_pos == -1 or curr_pos < start_pos:
                    start_pos = curr_pos
                    start_pattern_found = match.group()
        
        if start_pos == -1:
            logger.warning("Could not find start of risk factors section")
            return None
            
        # Find end position
        end_pos = len(content)
        for pattern in end_patterns:
            # Only search in content after the start position
            matches = list(re.finditer(pattern, content[start_pos:], re.IGNORECASE))
            if matches:
                # Take the first match after the start
                curr_pos = start_pos + matches[0].start()
                # Only update if this position is closer to start than current end
                if curr_pos > start_pos and curr_pos < end_pos:
                    end_pos = curr_pos
        
        # Extract risk section
        risk_section = content[start_pos:end_pos].strip()
        
        # Additional cleaning
        # Remove table formatting
        risk_section = re.sub(r'\|[\s\d]+\|', '', risk_section)
        risk_section = re.sub(r'[\-\+]+\|[\-\+]+', '', risk_section)
        
        # Remove excessive whitespace
        risk_section = re.sub(r'\s+', ' ', risk_section)
        risk_section = re.sub(r'\n\s*\n', '\n\n', risk_section)
        
        # Validate section
        if len(risk_section) < 1000:  # Risk sections are typically very long
            logger.warning(f"Extracted risk section seems too short ({len(risk_section)} chars)")
            return None
            
        logger.info(f"Successfully extracted risk section ({len(risk_section)} chars)")
        return risk_section
        
    except Exception as e:
        logger.error(f"Error extracting risk factors: {e}")
        return None

class DataValidator:
    def __init__(self):
        self.validation_results = {}
    
    def validate_market_data(self, market_data, ticker):
        """Validate market data completeness and quality"""
        try:
            validation = {
                'completeness': True,
                'consistency': True,
                'anomalies': True,
                'statistical_validity': True,
                'warnings': []
            }
            
            # Check data presence
            if not market_data:
                validation['completeness'] = False
                validation['warnings'].append("No market data available")
                return False, validation
            
            # Check required fields
            required_fields = {
                'pre_disclosure': ['price', 'volume', 'volatility', 'spy_relative', 'trading_days'],
                'post_disclosure': ['price_change', 'volume_change', 'volatility_change', 'spy_relative_post', 'trading_days']
            }
            
            for section, fields in required_fields.items():
                if section not in market_data:
                    validation['completeness'] = False
                    validation['warnings'].append(f"Missing {section} data")
                    continue
                    
                for field in fields:
                    if field not in market_data[section]:
                        validation['completeness'] = False
                        validation['warnings'].append(f"Missing {field} in {section}")
            
            # Check for extreme values
            if validation['completeness']:
                # Price changes
                price_change = market_data['post_disclosure']['price_change']
                if abs(price_change) > 30:  # 30% change threshold
                    validation['anomalies'] = False
                    validation['warnings'].append(f"Large price change detected: {price_change:,.2f}%")
                
                # Volume changes
                volume_change = market_data['post_disclosure']['volume_change']
                if abs(volume_change) > 300:  # 300% volume change threshold
                    validation['anomalies'] = False
                    validation['warnings'].append(f"Large volume change detected: {volume_change:,.2f}%")
                
                # Volatility
                pre_vol = market_data['pre_disclosure']['volatility']
                vol_change = market_data['post_disclosure']['volatility_change']
                
                if pre_vol > 100:  # 100% annualized volatility threshold
                    validation['anomalies'] = False
                    validation['warnings'].append(f"High pre-disclosure volatility: {pre_vol:,.2f}%")
                    
                if abs(vol_change) > 50:  # 50 percentage point change threshold
                    validation['anomalies'] = False
                    validation['warnings'].append(f"Large volatility change: {vol_change:,.2f} percentage points")
                
                # Trading days
                pre_days = market_data['pre_disclosure']['trading_days']
                post_days = market_data['post_disclosure']['trading_days']
                if pre_days < 10 or post_days < 10:  # Minimum 10 trading days
                    validation['completeness'] = False
                    validation['warnings'].append(f"Insufficient trading days: pre={pre_days}, post={post_days}")
            
            # Check data consistency
            if validation['completeness']:
                validation['consistency'] = self._check_data_consistency(market_data)
                if not validation['consistency']:
                    validation['warnings'].append("Inconsistent data patterns detected")
            
            # Store results
            self.validation_results[ticker] = validation
            
            return all([validation[key] for key in ['completeness', 'consistency', 'anomalies', 'statistical_validity']]), validation
            
        except Exception as e:
            logger.error(f"Validation error for {ticker}: {e}")
            return False, {'warnings': [str(e)]}
    
    def _check_data_consistency(self, market_data):
        """Check internal consistency of market data"""
        try:
            pre = market_data.get('pre_disclosure', {})
            post = market_data.get('post_disclosure', {})
            
            # Check if we have all required fields
            required_fields = ['price', 'volume', 'volatility']
            if not all(field in pre for field in required_fields):
                return False
            
            # Check for logical consistency
            if pre.get('price', 0) <= 0 or pre.get('volume', 0) < 0:
                return False
            
            # Check for reasonable ranges
            if pre.get('volatility', 0) > 200:  # 200% volatility threshold
                return False
            
            # Check post-disclosure changes are within reasonable bounds
            if abs(post.get('price_change', 0)) > 50:  # 50% price change
                return False
            
            if abs(post.get('volume_change', 0)) > 500:  # 500% volume change
                return False
            
            if abs(post.get('volatility_change', 0)) > 100:  # 100 percentage point volatility change
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data consistency check error: {e}")
            return False
    
    def _calculate_trend(self, prices):
        """Calculate price trend"""
        if len(prices) < 2:
            return "insufficient_data"
        
        try:
            start_price = prices.iloc[0]
            end_price = prices.iloc[-1]
            change = ((end_price - start_price) / start_price) * 100
            
            if change > 5:
                return "upward"
            elif change < -5:
                return "downward"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Trend calculation error: {e}")
            return "calculation_error"
    
    def _calculate_volatility(self, prices):
        """Calculate price volatility"""
        try:
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            if volatility > 50:
                return "high"
            elif volatility > 25:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return "calculation_error"
    
    def _find_support_levels(self, prices):
        """Identify potential support levels"""
        try:
            # Simple implementation using rolling minimums
            return prices.rolling(window=20).min().dropna().unique().tolist()
        except Exception as e:
            logger.error(f"Support level calculation error: {e}")
            return []
    
    def _find_resistance_levels(self, prices):
        """Identify potential resistance levels"""
        try:
            # Simple implementation using rolling maximums
            return prices.rolling(window=20).max().dropna().unique().tolist()
        except Exception as e:
            logger.error(f"Resistance level calculation error: {e}")
            return []
    
    def _detect_volume_spikes(self, volumes):
        """Detect significant volume spikes"""
        try:
            mean_volume = volumes.mean()
            std_volume = volumes.std()
            spikes = volumes[volumes > (mean_volume + 2 * std_volume)]
            return spikes.index.tolist()
        except Exception as e:
            logger.error(f"Volume spike detection error: {e}")
            return []
    
    def _check_seasonality(self, data):
        """Check for seasonal patterns"""
        try:
            # Simple check for day-of-week effects
            returns = data['Close'].pct_change()
            by_day = returns.groupby(returns.index.dayofweek).mean()
            return by_day.to_dict()
        except Exception as e:
            logger.error(f"Seasonality check error: {e}")
            return {}
    
    def _analyze_volatility_patterns(self, data):
        """Analyze volatility patterns"""
        try:
            returns = data['Close'].pct_change()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            
            patterns = {
                'increasing': rolling_vol.is_monotonic_increasing,
                'decreasing': rolling_vol.is_monotonic_decreasing,
                'average': rolling_vol.mean(),
                'max': rolling_vol.max(),
                'min': rolling_vol.min()
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Volatility pattern analysis error: {e}")
            return {}

def get_company_info(ticker):
    """Get comprehensive company information from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Log the raw data for debugging
        logger.debug(f"Raw company info for {ticker}: {info}")
        
        return {
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'company_name': info.get('longName', ticker),
            'exchange': info.get('exchange', 'Unknown'),
            'currency': info.get('currency', 'USD')
        }
    except Exception as e:
        logger.error(f"Error getting company info for {ticker}: {e}")
        return None

def get_institutional_data(ticker):
    """Get institutional holdings and ownership data"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get institutional holders
        holders = stock.institutional_holders
        major_holders = stock.major_holders
        
        logger.debug(f"Raw institutional holders for {ticker}: {holders}")
        logger.debug(f"Raw major holders for {ticker}: {major_holders}")
        
        if holders is None or major_holders is None:
            logger.warning(f"No institutional data available for {ticker}")
            return None
            
        # Get total shares outstanding
        total_shares = stock.info.get('sharesOutstanding', 0)
        if total_shares == 0:
            logger.warning(f"No shares outstanding data for {ticker}")
            return None
            
        # Calculate institutional metrics
        total_inst_shares = holders['Shares'].sum() if not holders.empty else 0
        
        # Get historical institutional ownership changes
        hist_holders = stock.institutional_holders
        if hist_holders is not None and not hist_holders.empty:
            prev_holdings = hist_holders['Shares'].sum()
            holdings_change = ((total_inst_shares - prev_holdings) / prev_holdings * 100) if prev_holdings > 0 else 0
        else:
            holdings_change = 0
            
        return {
            'holdings_pct': (total_inst_shares / total_shares * 100) if total_shares > 0 else 0,
            'institutions_count': len(holders) if holders is not None else 0,
            'holdings_change': holdings_change,
            'major_holders': major_holders.values.tolist() if major_holders is not None else [],
            'top_institutions': holders.head().to_dict('records') if holders is not None else []
        }
        
    except Exception as e:
        logger.error(f"Error getting institutional data for {ticker}: {e}")
        return None

# Add new BATCH_MENU_OPTIONS
BATCH_MENU_OPTIONS = {
    1: "S&P 500 Analysis",
    2: "Custom Batch Analysis",
    3: "Back to Main Menu"
}

# Add new function to display batch menu
def display_batch_menu():
    """Display batch analysis menu and get user choice"""
    console.clear()
    console.print(Panel.fit("Batch Analysis Menu", style="bold blue"))
    
    # Display menu options
    for key, value in BATCH_MENU_OPTIONS.items():
        console.print(f"[bold]{key}[/bold]. {value}")
    
    # Get user choice
    try:
        choice = int(Prompt.ask("\nSelect an option", choices=[str(x) for x in BATCH_MENU_OPTIONS.keys()]))
        return choice
    except ValueError:
        console.print("[red]Invalid input. Please enter a number.[/red]")
        input("\nPress Enter to continue...")
        return None

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def run_sp500_analysis():
    """Run analysis on S&P 500 companies"""
    try:
        logger.info("Starting S&P 500 Analysis")
        console.print("\n[bold blue]Starting S&P 500 Analysis[/bold blue]")
        
        # Create batch results directory with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        batch_dir = Path(f"analysis/batch/sp500_{timestamp}")
        batch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created batch directory: {batch_dir}")
        
        # Configure batch-specific logging
        batch_log = batch_dir / 'batch_process.log'
        file_handler = logging.FileHandler(batch_log)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        try:
            # Load S&P 500 tickers
            logger.info("Fetching S&P 500 tickers...")
            sp500_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = sp500_df['Symbol'].tolist()
            logger.info(f"Found {len(tickers)} S&P 500 companies")
            
            # Process each company sequentially
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing S&P 500 companies...", total=len(tickers))
                
                for idx, ticker in enumerate(tickers, 1):
                    console.rule(f"[bold blue]Processing {ticker} ({idx}/{len(tickers)})")
                    logger.info(f"Starting analysis of {ticker} ({idx}/{len(tickers)})")
                    
                    try:
                        # Phase 1: File Collection
                        console.print(f"[yellow]Phase 1/3:[/yellow] Collecting files for {ticker}")
                        logger.info(f"{ticker}: Starting file collection")
                        
                        risk_files = get_available_risk_files(ticker)
                        if len(risk_files) < 2:
                            console.print(f"[yellow]→ Downloading SEC filings for {ticker}...[/yellow]")
                            logger.info(f"{ticker}: Downloading SEC filings")
                            download_10k_filings(ticker)
                            risk_files = get_available_risk_files(ticker)
                            logger.info(f"{ticker}: Found {len(risk_files)} risk files after download")
                        else:
                            console.print(f"[green]→ Found {len(risk_files)} existing risk files[/green]")
                        
                        # Phase 2: Risk Analysis
                        if len(risk_files) >= 2:
                            console.print(f"[yellow]Phase 2/3:[/yellow] Analyzing risk factors")
                            logger.info(f"{ticker}: Starting risk analysis")
                            
                            recent_file = risk_files[0]
                            earlier_file = risk_files[1]
                            console.print(f"→ Comparing {recent_file.name} with {earlier_file.name}")
                            
                            recent_data, earlier_data = prepare_risk_comparison(recent_file, earlier_file)
                            if not recent_data or not earlier_data:
                                raise Exception("Failed to prepare risk comparison data")
                            
                            logger.info(f"{ticker}: Prepared risk comparison data")
                            
                            # Phase 3: GPT Analysis
                            console.print(f"[yellow]Phase 3/3:[/yellow] Running GPT analysis")
                            logger.info(f"{ticker}: Starting GPT analysis")
                            analysis = analyze_risk_sections(recent_data, earlier_data, client, ticker)
                            
                            if not analysis:
                                raise Exception("GPT analysis failed to return results")
                            
                            console.print("[green]→ GPT analysis completed successfully[/green]")
                            logger.info(f"{ticker}: GPT analysis completed")
                            
                            # Save results
                            console.print("→ Saving analysis results...")
                            company_file = batch_dir / f"{ticker}_analysis.json"
                            result_data = {
                                'ticker': ticker,
                                'recent_year': recent_data['year'],
                                'earlier_year': earlier_data['year'],
                                'analysis': analysis,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'gpt_response': analysis
                            }
                            
                            # Use custom JSON serializer
                            with open(company_file, 'w') as f:
                                json.dump(result_data, f, indent=4, default=json_serial)
                            
                            logger.info(f"{ticker}: Analysis saved to {company_file}")
                            
                            # Update checkpoint with success
                            update_batch_status(batch_dir, ticker, {
                                'status': 'completed',
                                'details': 'Analysis completed successfully',
                                'gpt_response': analysis,
                                'error': None
                            })
                            
                            console.print("[bold green]✓ Company analysis completed successfully[/bold green]")
                        else:
                            raise Exception("Insufficient risk files available")
                            
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"{ticker}: Analysis failed - {error_msg}")
                        console.print(f"[bold red]✗ Error processing {ticker}: {error_msg}[/bold red]")
                        
                        # Update checkpoint with failure
                        update_batch_status(batch_dir, ticker, {
                            'status': 'failed',
                            'details': f"Failed during processing: {error_msg}",
                            'error': error_msg,
                            'gpt_response': None
                        })
                        
                        # Stop processing on failure
                        logger.error("Stopping S&P 500 analysis due to failure")
                        console.print("[bold red]Stopping S&P 500 analysis due to failure[/bold red]")
                        return None
                    
                    # Progress update
                    progress.update(task, advance=1)
                    console.print(f"[dim]Rate limiting - waiting 1 second...[/dim]")
                    sleep(1)
                    
                    # Add visual separator
                    console.print("\n" + "="*80 + "\n")
            
            # Final summary
            console.rule("[bold green]Batch Analysis Complete!")
            
            # Load final checkpoint data
            with open(batch_dir / 'checkpoint.json', 'r') as f:
                final_checkpoint = json.load(f)
            
            console.print(f"\n[bold]Final Statistics:[/bold]")
            console.print(f"Total Companies Processed: {idx}")
            console.print(f"Successfully Completed: {len(final_checkpoint['completed'])}")
            console.print(f"Failed: {len(final_checkpoint['failed'])}")
            console.print(f"\nResults saved in: {batch_dir}")
            
            return batch_dir
            
        finally:
            # Remove batch-specific log handler
            logger.removeHandler(file_handler)
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        console.print(f"[red]Error in batch analysis: {e}[/red]")
        return None

def run_custom_batch_analysis():
    """Run analysis on custom list of tickers"""
    try:
        console.print("\n[bold blue]Custom Batch Analysis[/bold blue]")
        
        # Get ticker input
        console.print("\nEnter tickers separated by commas (e.g., AAPL, MSFT, GOOGL):")
        ticker_input = Prompt.ask("Tickers")
        tickers = [t.strip().upper() for t in ticker_input.split(',')]
        
        if not tickers:
            console.print("[red]No tickers provided[/red]")
            return None
            
        # Create batch directory
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        batch_dir = Path(f"analysis/batch/custom_{timestamp}")
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize batch summary
        batch_summary = {
            'batch_id': f'custom_{timestamp}',
            'total_companies': len(tickers),
            'completed': 0,
            'failed': 0,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'companies': {}
        }
        
        # Process companies
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing companies...", total=len(tickers))
            
            for ticker in tickers:
                try:
                    console.print(f"\n[bold blue]Processing {ticker}...[/bold blue]")
                    
                    # Update checkpoint that we're starting this ticker
                    update_batch_status(batch_dir, ticker, {
                        'status': 'in_progress',
                        'details': 'Starting analysis',
                        'error': None
                    })
                    
                    # Use existing single company analysis logic
                    risk_files = get_available_risk_files(ticker)
                    
                    if len(risk_files) < 2:
                        console.print(f"[yellow]Downloading filings for {ticker}...[/yellow]")
                        download_10k_filings(ticker)
                        risk_files = get_available_risk_files(ticker)
                    
                    if len(risk_files) >= 2:
                        recent_file = risk_files[0]
                        earlier_file = risk_files[1]
                        
                        recent_data, earlier_data = prepare_risk_comparison(recent_file, earlier_file)
                        analysis = analyze_risk_sections(recent_data, earlier_data, client, ticker)
                        
                        if analysis:
                            # Save individual result
                            company_file = batch_dir / f"{ticker}_analysis.json"
                            result_data = {
                                'ticker': ticker,
                                'recent_year': recent_data['year'],
                                'earlier_year': earlier_data['year'],
                                'analysis': analysis,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'gpt_response': analysis  # Save the full GPT response
                            }
                            
                            with open(company_file, 'w') as f:
                                json.dump(result_data, f, indent=4)
                            
                            # Update checkpoint with success
                            update_batch_status(batch_dir, ticker, {
                                'status': 'completed',
                                'details': 'Analysis completed successfully',
                                'gpt_response': analysis,
                                'error': None
                            })
                            
                            batch_summary['completed'] += 1
                    else:
                        raise Exception("Insufficient risk files available")
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error processing {ticker}: {error_msg}")
                    
                    # Update checkpoint with failure details
                    update_batch_status(batch_dir, ticker, {
                        'status': 'failed',
                        'details': f"Failed during processing: {error_msg}",
                        'error': error_msg,
                        'gpt_response': None
                    })
                    
                    batch_summary['failed'] += 1
                
                # Update progress and save current batch summary
                progress.update(task, advance=1)
                batch_summary['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
                with open(batch_dir / 'batch_summary.json', 'w') as f:
                    json.dump(batch_summary, f, indent=4)
                
                # Rate limiting
                sleep(1)
        
        # Final summary
        console.print("\n[bold green]Batch Analysis Complete![/bold green]")
        console.print(f"Total Companies: {batch_summary['total_companies']}")
        console.print(f"Successful: {batch_summary['completed']}")
        console.print(f"Failed: {batch_summary['failed']}")
        console.print(f"Results saved in: {batch_dir}")
        
        return batch_dir
        
    except Exception as e:
        logger.error(f"Custom batch analysis error: {e}")
        console.print(f"[red]Error in custom batch analysis: {e}[/red]")
        return None

def save_checkpoint(batch_dir: Path, checkpoint_data: dict):
    """Save checkpoint data with detailed status"""
    checkpoint_file = batch_dir / 'checkpoint.json'
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=4, default=json_serial)
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")

def update_batch_status(batch_dir: Path, ticker: str, status: dict):
    """Update batch status with detailed information"""
    try:
        checkpoint_file = batch_dir / 'checkpoint.json'
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
        else:
            checkpoint_data = {
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'completed': [],
                'failed': [],
                'skipped': [],
                'in_progress': None,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'companies': {}
            }

        # Update company specific data
        checkpoint_data['companies'][ticker] = {
            'status': status['status'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'details': status['details'],
            'gpt_response': status.get('gpt_response'),
            'error': status.get('error')
        }

        # Update status lists
        if status['status'] == 'completed':
            if ticker not in checkpoint_data['completed']:
                checkpoint_data['completed'].append(ticker)
        elif status['status'] == 'failed':
            if ticker not in checkpoint_data['failed']:
                checkpoint_data['failed'].append({
                    'ticker': ticker,
                    'error': status['error'],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'details': status['details']
                })

        checkpoint_data['in_progress'] = None
        checkpoint_data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')

        save_checkpoint(batch_dir, checkpoint_data)
        return checkpoint_data

    except Exception as e:
        logger.error(f"Error updating batch status for {ticker}: {e}")
        return None

def main():
    """Main application flow"""
    while True:
        choice = display_menu()
        
        if choice == 1:
            single_company_analysis()
            
        elif choice == 2:
            # Handle batch menu
            while True:
                batch_choice = display_batch_menu()
                if batch_choice == 1:
                    batch_dir = run_sp500_analysis()
                    if batch_dir:
                        console.print(f"\n[green]Analysis complete. Results saved in: {batch_dir}[/green]")
                    input("\nPress Enter to continue...")
                elif batch_choice == 2:
                    batch_dir = run_custom_batch_analysis()
                    if batch_dir:
                        console.print(f"\n[green]Analysis complete. Results saved in: {batch_dir}[/green]")
                    input("\nPress Enter to continue...")
                elif batch_choice == 3:
                    break
            
        elif choice == 3:
            display_settings_menu()
            
        elif choice == 4:
            console.print("\n[bold green]Thank you for using the 10-K Risk Factor Analysis tool![/bold green]")
            break

if __name__ == "__main__":
    try:
        if not os.getenv('OPENAI_API_KEY'):
            console.print("[red]Error: OPENAI_API_KEY not found in .env file[/red]")
            console.print("Please add your OpenAI API key to the .env file:")
            console.print("OPENAI_API_KEY=your_key_here")
            exit(1)
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user[/bold red]")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        console.print(f"\n[bold red]An error occurred: {e}[/bold red]") 