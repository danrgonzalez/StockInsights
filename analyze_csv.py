#!/usr/bin/env python3
"""
Script to analyze CSV data and compare with existing stock classifications
"""

import csv
import re
from typing import Set, Dict, List, Tuple

def normalize_symbol(symbol: str) -> str:
    """Normalize ticker symbols for mapping (e.g., BRK/B -> BRK_B)."""
    return symbol.replace("/", "_").upper()

def normalize_enum_name(name: str) -> str:
    """Convert a name to enum format (uppercase with underscores)."""
    # Replace special characters and spaces with underscores
    name = re.sub(r'[^\w\s]', '', name)  # Remove special chars except word chars and spaces
    name = re.sub(r'\s+', '_', name)     # Replace spaces with underscores
    name = name.upper()
    
    # Handle specific cases
    name = name.replace('&', 'AND')
    name = name.replace('-', '_')
    name = name.replace(',', '')
    
    return name

def parse_csv_file(csv_path: str) -> List[Dict[str, str]]:
    """Parse the CSV file and return stock data with complete classifications."""
    stocks = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
        
        # Find the header line containing "Symbol,SimpleMovingAvg..."
        header_line_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('Symbol,') and 'Sector,Industry,Sub-Industry' in line:
                header_line_idx = i
                break
        
        if header_line_idx is None:
            raise ValueError("Could not find proper CSV header line")
        
        # Use the remaining lines starting from header as a new file-like object
        from io import StringIO
        csv_content = ''.join(lines[header_line_idx:])
        csv_file = StringIO(csv_content)
        
        # Now use csv.DictReader on the properly formatted content
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            symbol = row.get('Symbol', '').strip()
            sector = row.get('Sector', '').strip()
            industry = row.get('Industry', '').strip()
            sub_industry = row.get('Sub-Industry', '').strip()
            
            # Skip rows with incomplete data or empty values
            if not symbol or not sector or not industry or not sub_industry:
                continue
            if sector == '<empty>' or industry == '<empty>' or sub_industry == '<empty>':
                continue
                
            stocks.append({
                'symbol': normalize_symbol(symbol),
                'sector': sector,
                'industry': industry,
                'sub_industry': sub_industry
            })
    
    return stocks

def get_existing_classifications():
    """Get existing classifications from the current file."""
    existing_symbols = {
        'AAL', 'AAPL', 'ABBV', 'ACN', 'ADSK', 'AKAM', 'AMGN', 'AMZN', 'AVGO', 'AXP',
        'BA', 'BAC', 'BLK', 'BMY', 'BRK_B', 'CAT', 'CCI', 'CCL', 'CHTR', 'CL', 'CMCSA',
        'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DFS', 'DHR', 'DIS', 'EBAY',
        'ETN', 'EXC', 'F', 'FCX', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOGL', 'GS', 'HAL',
        'HD', 'HUM', 'IBM', 'INTC', 'ISRG', 'JNJ', 'JPM', 'KHC', 'KO', 'LHX', 'LLY',
        'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK',
        'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NOW', 'NSC', 'NVDA', 'ORCL', 'PEP', 'PFE',
        'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SLB', 'SO', 'SPGI', 'SYK', 'T',
        'TGT', 'TMO', 'TSLA', 'TSM', 'TXN', 'UNH', 'UNP', 'UPS', 'V', 'VZ', 'WFC',
        'WMT', 'XOM'
    }
    
    existing_sectors = {
        'Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy',
        'Financials', 'Health Care', 'Industrials', 'Information Technology', 'Materials',
        'Utilities'
    }
    
    existing_industries = {
        'Aerospace & Defense', 'Automobiles', 'Banks', 'Beverages', 'Biotechnology',
        'Broadline Retail', 'Capital Markets', 'Chemicals', 'Communications Equipment',
        'Construction Materials', 'Consumer Finance', 'Consumer Staples Distribution & Retail',
        'Diversified Telecommunication Services', 'Electric Utilities', 'Electrical Equipment',
        'Electronic Equipment, Instruments & Components', 'Food Products',
        'Health Care Equipment & Supplies', 'Hotels, Restaurants & Leisure', 'Household Durables',
        'Household Products', 'Industrial Conglomerates', 'Insurance', 'Interactive Media & Services',
        'IT Services', 'Life Sciences Tools & Services', 'Machinery', 'Media', 'Metals & Mining',
        'Multiline Retail', 'Oil, Gas & Consumable Fuels', 'Passenger Airlines', 'Personal Products',
        'Pharmaceuticals', 'Real Estate Management & Development', 'Road & Rail',
        'Semiconductors & Semiconductor Equipment', 'Software', 'Specialty Retail',
        'Technology Hardware, Storage & Peripherals', 'Textiles, Apparel & Luxury Goods'
    }
    
    existing_sub_industries = {
        'Aerospace & Defense', 'Apparel Retail', 'Apparel, Accessories & Luxury Goods',
        'Application Software', 'Asset Management & Custody Banks', 'Automobile Manufacturers',
        'Automotive Retail', 'Biotechnology', 'Broadline Retail', 'Cable & Satellite',
        'Casinos & Gaming', 'Communications Equipment', 'Construction Machinery & Heavy Transportation Equipment',
        'Construction Materials', 'Consumer Electronics', 'Consumer Finance',
        'Consumer Staples Distribution & Retail', 'Diversified Banks', 'Drug Retail',
        'Electric Utilities', 'Electrical Components & Equipment', 'Electronic Equipment, Instruments & Components',
        'Food Distributors', 'Food Products', 'Health Care Distributors', 'Health Care Equipment',
        'Health Care Services', 'Health Care Supplies', 'Home Improvement Retail', 'Homebuilding',
        'Hotels, Resorts & Cruise Lines', 'Household Products', 'Industrial Conglomerates',
        'Integrated Oil & Gas', 'Interactive Media & Services', 'Internet Services & Infrastructure',
        'Investment Banking & Brokerage', 'IT Consulting & Other Services', 'Life Sciences Tools & Services',
        'Machinery', 'Movies & Entertainment', 'Multi-Sector Holdings', 'Multiline Insurance',
        'Oil & Gas Equipment & Services', 'Oil & Gas Exploration & Production', 'Oil & Gas Refining & Marketing',
        'Packaged Foods & Meats', 'Passenger Airlines', 'Personal Products', 'Pharmaceuticals',
        'Precious Metals & Minerals', 'Property & Casualty Insurance', 'Railroads',
        'Real Estate Services', 'Restaurants', 'Semiconductors', 'Soft Drinks',
        'Specialized Finance', 'Steel', 'Systems Software', 'Technology Hardware, Storage & Peripherals',
        'Wireless Telecommunication Services'
    }
    
    return existing_symbols, existing_sectors, existing_industries, existing_sub_industries

def main():
    csv_path = '/Users/dgonzalez/Library/CloudStorage/GoogleDrive-danrgonzalez@gmail.com/My Drive/GDrive/python/StockInsights/2025-08-15-Quote.csv'
    
    # Parse CSV data
    csv_stocks = parse_csv_file(csv_path)
    print(f"Found {len(csv_stocks)} stocks with complete classification data in CSV")
    
    # Get existing classifications
    existing_symbols, existing_sectors, existing_industries, existing_sub_industries = get_existing_classifications()
    
    # Find new data
    new_symbols = set()
    new_sectors = set()
    new_industries = set()
    new_sub_industries = set()
    
    csv_sectors = set()
    csv_industries = set()
    csv_sub_industries = set()
    
    for stock in csv_stocks:
        symbol = stock['symbol']
        sector = stock['sector']
        industry = stock['industry']
        sub_industry = stock['sub_industry']
        
        csv_sectors.add(sector)
        csv_industries.add(industry)
        csv_sub_industries.add(sub_industry)
        
        if symbol not in existing_symbols:
            new_symbols.add(symbol)
        if sector not in existing_sectors:
            new_sectors.add(sector)
        if industry not in existing_industries:
            new_industries.add(industry)
        if sub_industry not in existing_sub_industries:
            new_sub_industries.add(sub_industry)
    
    print(f"\nNew symbols to add: {len(new_symbols)}")
    print(f"New sectors to add: {len(new_sectors)}")
    print(f"New industries to add: {len(new_industries)}")
    print(f"New sub-industries to add: {len(new_sub_industries)}")
    
    # Print details
    if new_symbols:
        print(f"\nNew Symbols ({len(new_symbols)}):")
        for symbol in sorted(new_symbols):
            print(f"  {symbol}")
    
    if new_sectors:
        print(f"\nNew Sectors ({len(new_sectors)}):")
        for sector in sorted(new_sectors):
            print(f"  {sector} -> {normalize_enum_name(sector)}")
    
    if new_industries:
        print(f"\nNew Industries ({len(new_industries)}):")
        for industry in sorted(new_industries):
            print(f"  {industry} -> {normalize_enum_name(industry)}")
    
    if new_sub_industries:
        print(f"\nNew Sub-Industries ({len(new_sub_industries)}):")
        for sub_industry in sorted(new_sub_industries):
            print(f"  {sub_industry} -> {normalize_enum_name(sub_industry)}")
    
    # Generate code for new enums
    print("\n" + "="*80)
    print("CODE GENERATION")
    print("="*80)
    
    if new_sectors:
        print("\n# New Sector enum values to add:")
        for sector in sorted(new_sectors):
            enum_name = normalize_enum_name(sector)
            print(f'    {enum_name} = "{sector}"')
    
    if new_industries:
        print("\n# New Industry enum values to add:")
        for industry in sorted(new_industries):
            enum_name = normalize_enum_name(industry)
            print(f'    {enum_name} = "{industry}"')
    
    if new_sub_industries:
        print("\n# New SubIndustry enum values to add:")
        for sub_industry in sorted(new_sub_industries):
            enum_name = normalize_enum_name(sub_industry)
            print(f'    {enum_name} = "{sub_industry}"')
    
    if new_symbols:
        print("\n# New StockSymbol enum values to add:")
        for symbol in sorted(new_symbols):
            # Handle special case for BRK_B
            original_symbol = symbol.replace("_", "/") if symbol == "BRK_B" else symbol
            print(f'    {symbol} = "{original_symbol}"')
    
    # Generate classification mappings for new stocks
    print("\n# New classification mappings to add to STOCK_CLASSIFICATIONS:")
    new_stock_data = []
    for stock in csv_stocks:
        if stock['symbol'] in new_symbols:
            new_stock_data.append(stock)
    
    # Sort by symbol for consistency
    new_stock_data.sort(key=lambda x: x['symbol'])
    
    for stock in new_stock_data:
        symbol = stock['symbol']
        sector = stock['sector']
        industry = stock['industry']
        sub_industry = stock['sub_industry']
        
        sector_enum = normalize_enum_name(sector)
        industry_enum = normalize_enum_name(industry)
        sub_industry_enum = normalize_enum_name(sub_industry)
        
        print(f"""    StockSymbol.{symbol}: StockClassification(
        sector=Sector.{sector_enum},
        industry=Industry.{industry_enum},
        sub_industry=SubIndustry.{sub_industry_enum}
    ),""")

if __name__ == "__main__":
    main()