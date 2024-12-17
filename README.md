# 10-K Risk Factor Analysis Tool

This tool analyzes risk factors from company 10-K filings using GPT-4 to identify changes and potential market impacts.

## Features

- Single company analysis
- Batch analysis (S&P 500 or custom list)
- Automated SEC filing downloads
- Risk factor extraction and comparison
- Market data correlation
- GPT-4 powered analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/risk-analysis-tool.git
cd risk-analysis-tool
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```plaintext
OPENAI_API_KEY=your_api_key_here
```

5. Create a `sec_config.json` file with your SEC API credentials:
```json
{
    "company": "Your Company Name",
    "email": "your.email@example.com",
    "phone": "555-555-5555"
}
```

## Usage

Run the main script:
```bash
python risk.py
```

### Analysis Options:

1. Single Company Analysis
   - Enter a ticker symbol
   - Compare risk factors between years
   - View detailed analysis and market impact

2. Batch Analysis
   - Analyze S&P 500 companies
   - Analyze custom list of companies
   - Results saved to analysis directory

3. Settings
   - Configure GPT model
   - Adjust analysis parameters

## Directory Structure

```
risk-analysis-tool/
├── risk.py
├── requirements.txt
├── README.md
├── .env
├── sec_config.json
├── raw/              # Raw SEC filings
├── clean/            # Cleaned risk sections
├── working/          # Temporary processing files
└── analysis/         # Analysis results
    ├── single/       # Single company results
    └── batch/        # Batch analysis results
```

## License

MIT License - see LICENSE file for details.