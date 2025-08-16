"""
Stock Symbol Sector Classifications Module

This module provides Pydantic enumerations and models for mapping stock symbols
to their corresponding sectors, industries, and sub-industries based on the
TOS data from 2024-06-21.
"""

from enum import Enum
from typing import NamedTuple, Optional, List
from pydantic import BaseModel


class Sector(str, Enum):
    """Enumeration of stock market sectors."""
    COMMUNICATION_SERVICES = "Communication Services"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    ENERGY = "Energy"
    FINANCIALS = "Financials"
    HEALTH_CARE = "Health Care"
    INDUSTRIALS = "Industrials"
    INFORMATION_TECHNOLOGY = "Information Technology"
    MATERIALS = "Materials"
    UTILITIES = "Utilities"


class Industry(str, Enum):
    """Enumeration of stock market industries."""
    AEROSPACE_DEFENSE = "Aerospace & Defense"
    AUTOMOBILES = "Automobiles"
    BANKS = "Banks"
    BEVERAGES = "Beverages"
    BIOTECHNOLOGY = "Biotechnology"
    BROADLINE_RETAIL = "Broadline Retail"
    CAPITAL_MARKETS = "Capital Markets"
    CHEMICALS = "Chemicals"
    COMMUNICATIONS_EQUIPMENT = "Communications Equipment"
    CONSTRUCTION_MATERIALS = "Construction Materials"
    CONSUMER_FINANCE = "Consumer Finance"
    CONSUMER_STAPLES_DISTRIBUTION_RETAIL = "Consumer Staples Distribution & Retail"
    DIVERSIFIED_TELECOMMUNICATION_SERVICES = "Diversified Telecommunication Services"
    ELECTRIC_UTILITIES = "Electric Utilities"
    ELECTRICAL_EQUIPMENT = "Electrical Equipment"
    ELECTRONIC_EQUIPMENT_INSTRUMENTS = "Electronic Equipment, Instruments & Components"
    ENERGY_EQUIPMENT_SERVICES = "Energy Equipment & Services"
    ENTERTAINMENT = "Entertainment"
    FINANCIAL_SERVICES = "Financial Services"
    FOOD_PRODUCTS = "Food Products"
    GROUND_TRANSPORTATION = "Ground Transportation"
    HEALTH_CARE_EQUIPMENT_SUPPLIES = "Health Care Equipment & Supplies"
    HEALTH_CARE_PROVIDERS_SERVICES = "Health Care Providers & Services"
    HOTELS_RESTAURANTS_LEISURE = "Hotels, Restaurants & Leisure"
    HOUSEHOLD_DURABLES = "Household Durables"
    HOUSEHOLD_PRODUCTS = "Household Products"
    INDUSTRIAL_CONGLOMERATES = "Industrial Conglomerates"
    INSURANCE = "Insurance"
    INTERACTIVE_MEDIA_SERVICES = "Interactive Media & Services"
    IT_SERVICES = "IT Services"
    LIFE_SCIENCES_TOOLS_SERVICES = "Life Sciences Tools & Services"
    MACHINERY = "Machinery"
    MEDIA = "Media"
    METALS_MINING = "Metals & Mining"
    MULTILINE_RETAIL = "Multiline Retail"
    MULTI_UTILITIES = "Multi-Utilities"
    OIL_GAS_CONSUMABLE_FUELS = "Oil, Gas & Consumable Fuels"
    PASSENGER_AIRLINES = "Passenger Airlines"
    PERSONAL_PRODUCTS = "Personal Products"
    PHARMACEUTICALS = "Pharmaceuticals"
    PROFESSIONAL_SERVICES = "Professional Services"
    REAL_ESTATE_MANAGEMENT_DEVELOPMENT = "Real Estate Management & Development"
    ROAD_RAIL = "Road & Rail"
    SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT = "Semiconductors & Semiconductor Equipment"
    SOFTWARE = "Software"
    SPECIALTY_RETAIL = "Specialty Retail"
    TECHNOLOGY_HARDWARE_STORAGE_PERIPHERALS = "Technology Hardware, Storage & Peripherals"
    TEXTILES_APPAREL_LUXURY_GOODS = "Textiles, Apparel & Luxury Goods"
    TOBACCO = "Tobacco"


class SubIndustry(str, Enum):
    """Enumeration of stock market sub-industries."""
    AEROSPACE_DEFENSE = "Aerospace & Defense"
    APPAREL_RETAIL = "Apparel Retail"
    APPAREL_ACCESSORIES_LUXURY_GOODS = "Apparel, Accessories & Luxury Goods"
    APPLICATION_SOFTWARE = "Application Software"
    ASSET_MANAGEMENT_CUSTODY_BANKS = "Asset Management & Custody Banks"
    AUTOMOBILE_MANUFACTURERS = "Automobile Manufacturers"
    AUTOMOTIVE_RETAIL = "Automotive Retail"
    BIOTECHNOLOGY = "Biotechnology"
    BROADLINE_RETAIL = "Broadline Retail"
    CABLE_SATELLITE = "Cable & Satellite"
    CASINOS_GAMING = "Casinos & Gaming"
    COMMUNICATIONS_EQUIPMENT = "Communications Equipment"
    CONSTRUCTION_MACHINERY_HEAVY_TRANSPORTATION_EQUIPMENT = "Construction Machinery & Heavy Transportation Equipment"
    CONSTRUCTION_MATERIALS = "Construction Materials"
    CONSUMER_ELECTRONICS = "Consumer Electronics"
    CONSUMER_FINANCE = "Consumer Finance"
    CONSUMER_STAPLES_DISTRIBUTION_RETAIL = "Consumer Staples Distribution & Retail"
    CONSUMER_STAPLES_MERCHANDISE_RETAIL = "Consumer Staples Merchandise Retail"
    DISTILLERS_VINTNERS = "Distillers & Vintners"
    DIVERSIFIED_BANKS = "Diversified Banks"
    DRUG_RETAIL = "Drug Retail"
    ELECTRIC_UTILITIES = "Electric Utilities"
    ELECTRICAL_COMPONENTS_EQUIPMENT = "Electrical Components & Equipment"
    ELECTRONIC_EQUIPMENT_INSTRUMENTS_COMPONENTS = "Electronic Equipment, Instruments & Components"
    FOOD_DISTRIBUTORS = "Food Distributors"
    FOOD_PRODUCTS = "Food Products"
    FOOD_RETAIL = "Food Retail"
    FOOTWEAR = "Footwear"
    HEALTH_CARE_DISTRIBUTORS = "Health Care Distributors"
    HEALTH_CARE_EQUIPMENT = "Health Care Equipment"
    HEALTH_CARE_SERVICES = "Health Care Services"
    HEALTH_CARE_SUPPLIES = "Health Care Supplies"
    HOME_IMPROVEMENT_RETAIL = "Home Improvement Retail"
    HOMEBUILDING = "Homebuilding"
    HOMEFURNISHING_RETAIL = "Homefurnishing Retail"
    HOTELS_RESORTS_CRUISE_LINES = "Hotels, Resorts & Cruise Lines"
    HOUSEHOLD_PRODUCTS = "Household Products"
    INDUSTRIAL_CONGLOMERATES = "Industrial Conglomerates"
    INTEGRATED_OIL_GAS = "Integrated Oil & Gas"
    INTEGRATED_TELECOMMUNICATION_SERVICES = "Integrated Telecommunication Services"
    INTERACTIVE_HOME_ENTERTAINMENT = "Interactive Home Entertainment"
    INTERACTIVE_MEDIA_SERVICES = "Interactive Media & Services"
    INTERNET_SERVICES_INFRASTRUCTURE = "Internet Services & Infrastructure"
    INVESTMENT_BANKING_BROKERAGE = "Investment Banking & Brokerage"
    IT_CONSULTING_OTHER_SERVICES = "IT Consulting & Other Services"
    LIFE_SCIENCES_TOOLS_SERVICES = "Life Sciences Tools & Services"
    MACHINERY = "Machinery"
    MANAGED_HEALTH_CARE = "Managed Health Care"
    MOVIES_ENTERTAINMENT = "Movies & Entertainment"
    MULTI_SECTOR_HOLDINGS = "Multi-Sector Holdings"
    MULTI_UTILITIES = "Multi-Utilities"
    MULTILINE_INSURANCE = "Multiline Insurance"
    OIL_GAS_EQUIPMENT_SERVICES = "Oil & Gas Equipment & Services"
    OIL_GAS_EXPLORATION_PRODUCTION = "Oil & Gas Exploration & Production"
    OIL_GAS_REFINING_MARKETING = "Oil & Gas Refining & Marketing"
    OIL_GAS_STORAGE_TRANSPORTATION = "Oil & Gas Storage & Transportation"
    OTHER_SPECIALTY_RETAIL = "Other Specialty Retail"
    PACKAGED_FOODS_MEATS = "Packaged Foods & Meats"
    PASSENGER_AIRLINES = "Passenger Airlines"
    PERSONAL_PRODUCTS = "Personal Products"
    PHARMACEUTICALS = "Pharmaceuticals"
    PRECIOUS_METALS_MINERALS = "Precious Metals & Minerals"
    PROPERTY_CASUALTY_INSURANCE = "Property & Casualty Insurance"
    RAIL_TRANSPORTATION = "Rail Transportation"
    RAILROADS = "Railroads"
    REAL_ESTATE_SERVICES = "Real Estate Services"
    RESEARCH_CONSULTING_SERVICES = "Research & Consulting Services"
    RESTAURANTS = "Restaurants"
    SEMICONDUCTORS = "Semiconductors"
    SOFT_DRINKS = "Soft Drinks"
    SOFT_DRINKS_NONALCOHOLIC_BEVERAGES = "Soft Drinks & Non-alcoholic Beverages"
    SPECIALIZED_FINANCE = "Specialized Finance"
    SPECIALTY_CHEMICALS = "Specialty Chemicals"
    STEEL = "Steel"
    SYSTEMS_SOFTWARE = "Systems Software"
    TECHNOLOGY_HARDWARE_STORAGE_PERIPHERALS = "Technology Hardware, Storage & Peripherals"
    TOBACCO = "Tobacco"
    TRANSACTION_PAYMENT_PROCESSING_SERVICES = "Transaction & Payment Processing Services"
    WIRELESS_TELECOMMUNICATION_SERVICES = "Wireless Telecommunication Services"


class StockClassification(NamedTuple):
    """Classification data for a stock symbol."""
    sector: Sector
    industry: Industry
    sub_industry: SubIndustry


class StockSymbol(str, Enum):
    """Enumeration of stock symbols with their classifications."""
    AAL = "AAL"
    AAPL = "AAPL"
    ABBV = "ABBV"
    ACN = "ACN"
    ADSK = "ADSK"
    AKAM = "AKAM"
    AMGN = "AMGN"
    AMZN = "AMZN"
    AVGO = "AVGO"
    AXP = "AXP"
    AZO = "AZO"
    BA = "BA"
    BABA = "BABA"
    BAC = "BAC"
    BIDU = "BIDU"
    BIIB = "BIIB"
    BK = "BK"
    BKNG = "BKNG"
    BLK = "BLK"
    BMY = "BMY"
    BRK_B = "BRK.B"
    BSX = "BSX"
    C = "C"
    CAT = "CAT"
    CCI = "CCI"
    CCL = "CCL"
    CHTR = "CHTR"
    CL = "CL"
    CLX = "CLX"
    CMCSA = "CMCSA"
    CMG = "CMG"
    COF = "COF"
    COP = "COP"
    COST = "COST"
    CRM = "CRM"
    CSCO = "CSCO"
    CVS = "CVS"
    CVX = "CVX"
    D = "D"
    DAL = "DAL"
    DECK = "DECK"
    DFS = "DFS"
    DHR = "DHR"
    DIS = "DIS"
    DPZ = "DPZ"
    DUK = "DUK"
    EA = "EA"
    EBAY = "EBAY"
    EFX = "EFX"
    EOG = "EOG"
    ETN = "ETN"
    EW = "EW"
    EXC = "EXC"
    EXPE = "EXPE"
    F = "F"
    FCX = "FCX"
    FDX = "FDX"
    FFIV = "FFIV"
    FSLR = "FSLR"
    GD = "GD"
    GE = "GE"
    GILD = "GILD"
    GM = "GM"
    GOOGL = "GOOGL"
    GPRO = "GPRO"
    GRMN = "GRMN"
    GS = "GS"
    H = "H"
    HAIN = "HAIN"
    HAL = "HAL"
    HD = "HD"
    HON = "HON"
    HUM = "HUM"
    IBM = "IBM"
    INCY = "INCY"
    INTC = "INTC"
    ISRG = "ISRG"
    JNJ = "JNJ"
    JPM = "JPM"
    JWN = "JWN"
    KHC = "KHC"
    KMB = "KMB"
    KMI = "KMI"
    KO = "KO"
    KR = "KR"
    KSS = "KSS"
    KTOS = "KTOS"
    LHX = "LHX"
    LLY = "LLY"
    LMT = "LMT"
    LOW = "LOW"
    LULU = "LULU"
    LUV = "LUV"
    LVS = "LVS"
    M = "M"
    MA = "MA"
    MAR = "MAR"
    MCD = "MCD"
    MCHP = "MCHP"
    MDLZ = "MDLZ"
    MDT = "MDT"
    MET = "MET"
    META = "META"
    MGM = "MGM"
    MMM = "MMM"
    MO = "MO"
    MRK = "MRK"
    MS = "MS"
    MSFT = "MSFT"
    NEE = "NEE"
    NFLX = "NFLX"
    NKE = "NKE"
    NOW = "NOW"
    NSC = "NSC"
    NVDA = "NVDA"
    ORCL = "ORCL"
    ORLY = "ORLY"
    PANW = "PANW"
    PEP = "PEP"
    PFE = "PFE"
    PG = "PG"
    PM = "PM"
    PPG = "PPG"
    PVH = "PVH"
    PYPL = "PYPL"
    QCOM = "QCOM"
    REGN = "REGN"
    ROST = "ROST"
    RTX = "RTX"
    S = "S"
    SBH = "SBH"
    SBUX = "SBUX"
    SKX = "SKX"
    SLB = "SLB"
    STZ = "STZ"
    SWKS = "SWKS"
    SO = "SO"
    SPGI = "SPGI"
    SYK = "SYK"
    T = "T"
    TGT = "TGT"
    TJX = "TJX"
    TMO = "TMO"
    TSLA = "TSLA"
    TSN = "TSN"
    TSM = "TSM"
    TXN = "TXN"
    UAA = "UAA"
    UAL = "UAL"
    ULTA = "ULTA"
    UNH = "UNH"
    UNP = "UNP"
    UPS = "UPS"
    USB = "USB"
    V = "V"
    VFC = "VFC"
    VMC = "VMC"
    VZ = "VZ"
    WDAY = "WDAY"
    WFC = "WFC"
    WMT = "WMT"
    WSM = "WSM"
    WYNN = "WYNN"
    XOM = "XOM"


# Stock symbol classification mappings
STOCK_CLASSIFICATIONS = {
    StockSymbol.AAL: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.PASSENGER_AIRLINES,
        sub_industry=SubIndustry.PASSENGER_AIRLINES
    ),
    StockSymbol.AAPL: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.TECHNOLOGY_HARDWARE_STORAGE_PERIPHERALS,
        sub_industry=SubIndustry.TECHNOLOGY_HARDWARE_STORAGE_PERIPHERALS
    ),
    StockSymbol.ABBV: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.BIOTECHNOLOGY,
        sub_industry=SubIndustry.BIOTECHNOLOGY
    ),
    StockSymbol.ACN: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.IT_SERVICES,
        sub_industry=SubIndustry.IT_CONSULTING_OTHER_SERVICES
    ),
    StockSymbol.ADSK: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SOFTWARE,
        sub_industry=SubIndustry.APPLICATION_SOFTWARE
    ),
    StockSymbol.AKAM: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.IT_SERVICES,
        sub_industry=SubIndustry.INTERNET_SERVICES_INFRASTRUCTURE
    ),
    StockSymbol.AMGN: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.BIOTECHNOLOGY,
        sub_industry=SubIndustry.BIOTECHNOLOGY
    ),
    StockSymbol.AMZN: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.BROADLINE_RETAIL,
        sub_industry=SubIndustry.BROADLINE_RETAIL
    ),
    StockSymbol.AVGO: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT,
        sub_industry=SubIndustry.SEMICONDUCTORS
    ),
    StockSymbol.AXP: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.CONSUMER_FINANCE,
        sub_industry=SubIndustry.CONSUMER_FINANCE
    ),
    StockSymbol.AZO: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.AUTOMOTIVE_RETAIL
    ),
    StockSymbol.BA: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.AEROSPACE_DEFENSE,
        sub_industry=SubIndustry.AEROSPACE_DEFENSE
    ),
    StockSymbol.BABA: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.BROADLINE_RETAIL,
        sub_industry=SubIndustry.BROADLINE_RETAIL
    ),
    StockSymbol.BAC: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.BANKS,
        sub_industry=SubIndustry.DIVERSIFIED_BANKS
    ),
    StockSymbol.BIDU: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.INTERACTIVE_MEDIA_SERVICES,
        sub_industry=SubIndustry.INTERACTIVE_MEDIA_SERVICES
    ),
    StockSymbol.BIIB: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.BIOTECHNOLOGY,
        sub_industry=SubIndustry.BIOTECHNOLOGY
    ),
    StockSymbol.BK: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.CAPITAL_MARKETS,
        sub_industry=SubIndustry.ASSET_MANAGEMENT_CUSTODY_BANKS
    ),
    StockSymbol.BKNG: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.HOTELS_RESORTS_CRUISE_LINES
    ),
    StockSymbol.BLK: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.CAPITAL_MARKETS,
        sub_industry=SubIndustry.ASSET_MANAGEMENT_CUSTODY_BANKS
    ),
    StockSymbol.BMY: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.PHARMACEUTICALS,
        sub_industry=SubIndustry.PHARMACEUTICALS
    ),
    StockSymbol.BRK_B: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.FINANCIAL_SERVICES,
        sub_industry=SubIndustry.MULTI_SECTOR_HOLDINGS
    ),
    StockSymbol.BSX: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.HEALTH_CARE_EQUIPMENT_SUPPLIES,
        sub_industry=SubIndustry.HEALTH_CARE_EQUIPMENT
    ),
    StockSymbol.C: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.BANKS,
        sub_industry=SubIndustry.DIVERSIFIED_BANKS
    ),
    StockSymbol.CAT: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.MACHINERY,
        sub_industry=SubIndustry.CONSTRUCTION_MACHINERY_HEAVY_TRANSPORTATION_EQUIPMENT
    ),
    StockSymbol.CL: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.HOUSEHOLD_PRODUCTS,
        sub_industry=SubIndustry.HOUSEHOLD_PRODUCTS
    ),
    StockSymbol.CLX: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.HOUSEHOLD_PRODUCTS,
        sub_industry=SubIndustry.HOUSEHOLD_PRODUCTS
    ),
    StockSymbol.CMCSA: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.MEDIA,
        sub_industry=SubIndustry.CABLE_SATELLITE
    ),
    StockSymbol.CMG: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.RESTAURANTS
    ),
    StockSymbol.COF: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.CONSUMER_FINANCE,
        sub_industry=SubIndustry.CONSUMER_FINANCE
    ),
    StockSymbol.COP: StockClassification(
        sector=Sector.ENERGY,
        industry=Industry.OIL_GAS_CONSUMABLE_FUELS,
        sub_industry=SubIndustry.OIL_GAS_EXPLORATION_PRODUCTION
    ),
    StockSymbol.COST: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.CONSUMER_STAPLES_DISTRIBUTION_RETAIL,
        sub_industry=SubIndustry.CONSUMER_STAPLES_MERCHANDISE_RETAIL
    ),
    StockSymbol.CRM: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SOFTWARE,
        sub_industry=SubIndustry.APPLICATION_SOFTWARE
    ),
    StockSymbol.CSCO: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.COMMUNICATIONS_EQUIPMENT,
        sub_industry=SubIndustry.COMMUNICATIONS_EQUIPMENT
    ),
    StockSymbol.CVS: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.HEALTH_CARE_PROVIDERS_SERVICES,
        sub_industry=SubIndustry.HEALTH_CARE_SERVICES
    ),
    StockSymbol.CVX: StockClassification(
        sector=Sector.ENERGY,
        industry=Industry.OIL_GAS_CONSUMABLE_FUELS,
        sub_industry=SubIndustry.INTEGRATED_OIL_GAS
    ),
    StockSymbol.D: StockClassification(
        sector=Sector.UTILITIES,
        industry=Industry.MULTI_UTILITIES,
        sub_industry=SubIndustry.MULTI_UTILITIES
    ),
    StockSymbol.DAL: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.PASSENGER_AIRLINES,
        sub_industry=SubIndustry.PASSENGER_AIRLINES
    ),
    StockSymbol.DECK: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.TEXTILES_APPAREL_LUXURY_GOODS,
        sub_industry=SubIndustry.FOOTWEAR
    ),
    StockSymbol.DIS: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.ENTERTAINMENT,
        sub_industry=SubIndustry.MOVIES_ENTERTAINMENT
    ),
    StockSymbol.DPZ: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.RESTAURANTS
    ),
    StockSymbol.DUK: StockClassification(
        sector=Sector.UTILITIES,
        industry=Industry.ELECTRIC_UTILITIES,
        sub_industry=SubIndustry.ELECTRIC_UTILITIES
    ),
    StockSymbol.EA: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.ENTERTAINMENT,
        sub_industry=SubIndustry.INTERACTIVE_HOME_ENTERTAINMENT
    ),
    StockSymbol.EBAY: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.BROADLINE_RETAIL,
        sub_industry=SubIndustry.BROADLINE_RETAIL
    ),
    StockSymbol.EFX: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.PROFESSIONAL_SERVICES,
        sub_industry=SubIndustry.RESEARCH_CONSULTING_SERVICES
    ),
    StockSymbol.EOG: StockClassification(
        sector=Sector.ENERGY,
        industry=Industry.OIL_GAS_CONSUMABLE_FUELS,
        sub_industry=SubIndustry.OIL_GAS_EXPLORATION_PRODUCTION
    ),
    StockSymbol.ETN: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.ELECTRICAL_EQUIPMENT,
        sub_industry=SubIndustry.ELECTRICAL_COMPONENTS_EQUIPMENT
    ),
    StockSymbol.EW: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.HEALTH_CARE_EQUIPMENT_SUPPLIES,
        sub_industry=SubIndustry.HEALTH_CARE_EQUIPMENT
    ),
    StockSymbol.EXPE: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.HOTELS_RESORTS_CRUISE_LINES
    ),
    StockSymbol.F: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.AUTOMOBILES,
        sub_industry=SubIndustry.AUTOMOBILE_MANUFACTURERS
    ),
    StockSymbol.FFIV: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.COMMUNICATIONS_EQUIPMENT,
        sub_industry=SubIndustry.COMMUNICATIONS_EQUIPMENT
    ),
    StockSymbol.FSLR: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT,
        sub_industry=SubIndustry.SEMICONDUCTORS
    ),
    StockSymbol.GD: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.AEROSPACE_DEFENSE,
        sub_industry=SubIndustry.AEROSPACE_DEFENSE
    ),
    StockSymbol.GE: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.AEROSPACE_DEFENSE,
        sub_industry=SubIndustry.AEROSPACE_DEFENSE
    ),
    StockSymbol.GILD: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.BIOTECHNOLOGY,
        sub_industry=SubIndustry.BIOTECHNOLOGY
    ),
    StockSymbol.GM: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.AUTOMOBILES,
        sub_industry=SubIndustry.AUTOMOBILE_MANUFACTURERS
    ),
    StockSymbol.GOOGL: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.INTERACTIVE_MEDIA_SERVICES,
        sub_industry=SubIndustry.INTERACTIVE_MEDIA_SERVICES
    ),
    StockSymbol.GPRO: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOUSEHOLD_DURABLES,
        sub_industry=SubIndustry.CONSUMER_ELECTRONICS
    ),
    StockSymbol.GRMN: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOUSEHOLD_DURABLES,
        sub_industry=SubIndustry.CONSUMER_ELECTRONICS
    ),
    StockSymbol.GS: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.CAPITAL_MARKETS,
        sub_industry=SubIndustry.INVESTMENT_BANKING_BROKERAGE
    ),
    StockSymbol.H: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.HOTELS_RESORTS_CRUISE_LINES
    ),
    StockSymbol.HAIN: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.FOOD_PRODUCTS,
        sub_industry=SubIndustry.PACKAGED_FOODS_MEATS
    ),
    StockSymbol.HD: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.HOME_IMPROVEMENT_RETAIL
    ),
    StockSymbol.HON: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.INDUSTRIAL_CONGLOMERATES,
        sub_industry=SubIndustry.INDUSTRIAL_CONGLOMERATES
    ),
    StockSymbol.HUM: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.HEALTH_CARE_PROVIDERS_SERVICES,
        sub_industry=SubIndustry.MANAGED_HEALTH_CARE
    ),
    StockSymbol.IBM: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.IT_SERVICES,
        sub_industry=SubIndustry.IT_CONSULTING_OTHER_SERVICES
    ),
    StockSymbol.INCY: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.BIOTECHNOLOGY,
        sub_industry=SubIndustry.BIOTECHNOLOGY
    ),
    StockSymbol.INTC: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT,
        sub_industry=SubIndustry.SEMICONDUCTORS
    ),
    StockSymbol.JNJ: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.PHARMACEUTICALS,
        sub_industry=SubIndustry.PHARMACEUTICALS
    ),
    StockSymbol.JPM: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.BANKS,
        sub_industry=SubIndustry.DIVERSIFIED_BANKS
    ),
    StockSymbol.JWN: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.BROADLINE_RETAIL,
        sub_industry=SubIndustry.BROADLINE_RETAIL
    ),
    StockSymbol.KMB: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.HOUSEHOLD_PRODUCTS,
        sub_industry=SubIndustry.HOUSEHOLD_PRODUCTS
    ),
    StockSymbol.KMI: StockClassification(
        sector=Sector.ENERGY,
        industry=Industry.OIL_GAS_CONSUMABLE_FUELS,
        sub_industry=SubIndustry.OIL_GAS_STORAGE_TRANSPORTATION
    ),
    StockSymbol.KO: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.BEVERAGES,
        sub_industry=SubIndustry.SOFT_DRINKS_NONALCOHOLIC_BEVERAGES
    ),
    StockSymbol.KR: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.CONSUMER_STAPLES_DISTRIBUTION_RETAIL,
        sub_industry=SubIndustry.FOOD_RETAIL
    ),
    StockSymbol.KSS: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.BROADLINE_RETAIL,
        sub_industry=SubIndustry.BROADLINE_RETAIL
    ),
    StockSymbol.KTOS: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.AEROSPACE_DEFENSE,
        sub_industry=SubIndustry.AEROSPACE_DEFENSE
    ),
    StockSymbol.LLY: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.PHARMACEUTICALS,
        sub_industry=SubIndustry.PHARMACEUTICALS
    ),
    StockSymbol.LMT: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.AEROSPACE_DEFENSE,
        sub_industry=SubIndustry.AEROSPACE_DEFENSE
    ),
    StockSymbol.LOW: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.HOME_IMPROVEMENT_RETAIL
    ),
    StockSymbol.LULU: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.TEXTILES_APPAREL_LUXURY_GOODS,
        sub_industry=SubIndustry.APPAREL_ACCESSORIES_LUXURY_GOODS
    ),
    StockSymbol.LUV: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.PASSENGER_AIRLINES,
        sub_industry=SubIndustry.PASSENGER_AIRLINES
    ),
    StockSymbol.LVS: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.CASINOS_GAMING
    ),
    StockSymbol.M: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.BROADLINE_RETAIL,
        sub_industry=SubIndustry.BROADLINE_RETAIL
    ),
    StockSymbol.MA: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.FINANCIAL_SERVICES,
        sub_industry=SubIndustry.TRANSACTION_PAYMENT_PROCESSING_SERVICES
    ),
    StockSymbol.MAR: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.HOTELS_RESORTS_CRUISE_LINES
    ),
    StockSymbol.MCD: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.RESTAURANTS
    ),
    StockSymbol.MCHP: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT,
        sub_industry=SubIndustry.SEMICONDUCTORS
    ),
    StockSymbol.MDLZ: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.FOOD_PRODUCTS,
        sub_industry=SubIndustry.PACKAGED_FOODS_MEATS
    ),
    StockSymbol.MDT: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.HEALTH_CARE_EQUIPMENT_SUPPLIES,
        sub_industry=SubIndustry.HEALTH_CARE_EQUIPMENT
    ),
    StockSymbol.META: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.INTERACTIVE_MEDIA_SERVICES,
        sub_industry=SubIndustry.INTERACTIVE_MEDIA_SERVICES
    ),
    StockSymbol.MGM: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.CASINOS_GAMING
    ),
    StockSymbol.MMM: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.INDUSTRIAL_CONGLOMERATES,
        sub_industry=SubIndustry.INDUSTRIAL_CONGLOMERATES
    ),
    StockSymbol.MO: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.TOBACCO,
        sub_industry=SubIndustry.TOBACCO
    ),
    StockSymbol.MRK: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.PHARMACEUTICALS,
        sub_industry=SubIndustry.PHARMACEUTICALS
    ),
    StockSymbol.MS: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.CAPITAL_MARKETS,
        sub_industry=SubIndustry.INVESTMENT_BANKING_BROKERAGE
    ),
    StockSymbol.MSFT: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SOFTWARE,
        sub_industry=SubIndustry.SYSTEMS_SOFTWARE
    ),
    StockSymbol.NFLX: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.ENTERTAINMENT,
        sub_industry=SubIndustry.MOVIES_ENTERTAINMENT
    ),
    StockSymbol.NKE: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.TEXTILES_APPAREL_LUXURY_GOODS,
        sub_industry=SubIndustry.FOOTWEAR
    ),
    StockSymbol.NVDA: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT,
        sub_industry=SubIndustry.SEMICONDUCTORS
    ),
    StockSymbol.ORCL: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SOFTWARE,
        sub_industry=SubIndustry.SYSTEMS_SOFTWARE
    ),
    StockSymbol.ORLY: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.AUTOMOTIVE_RETAIL
    ),
    StockSymbol.PANW: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SOFTWARE,
        sub_industry=SubIndustry.SYSTEMS_SOFTWARE
    ),
    StockSymbol.PEP: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.BEVERAGES,
        sub_industry=SubIndustry.SOFT_DRINKS_NONALCOHOLIC_BEVERAGES
    ),
    StockSymbol.PFE: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.PHARMACEUTICALS,
        sub_industry=SubIndustry.PHARMACEUTICALS
    ),
    StockSymbol.PG: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.HOUSEHOLD_PRODUCTS,
        sub_industry=SubIndustry.HOUSEHOLD_PRODUCTS
    ),
    StockSymbol.PM: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.TOBACCO,
        sub_industry=SubIndustry.TOBACCO
    ),
    StockSymbol.PPG: StockClassification(
        sector=Sector.MATERIALS,
        industry=Industry.CHEMICALS,
        sub_industry=SubIndustry.SPECIALTY_CHEMICALS
    ),
    StockSymbol.PVH: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.TEXTILES_APPAREL_LUXURY_GOODS,
        sub_industry=SubIndustry.APPAREL_ACCESSORIES_LUXURY_GOODS
    ),
    StockSymbol.QCOM: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT,
        sub_industry=SubIndustry.SEMICONDUCTORS
    ),
    StockSymbol.REGN: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.BIOTECHNOLOGY,
        sub_industry=SubIndustry.BIOTECHNOLOGY
    ),
    StockSymbol.ROST: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.APPAREL_RETAIL
    ),
    StockSymbol.RTX: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.AEROSPACE_DEFENSE,
        sub_industry=SubIndustry.AEROSPACE_DEFENSE
    ),
    StockSymbol.S: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SOFTWARE,
        sub_industry=SubIndustry.SYSTEMS_SOFTWARE
    ),
    StockSymbol.SBH: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.OTHER_SPECIALTY_RETAIL
    ),
    StockSymbol.SBUX: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.RESTAURANTS
    ),
    StockSymbol.SKX: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.TEXTILES_APPAREL_LUXURY_GOODS,
        sub_industry=SubIndustry.FOOTWEAR
    ),
    StockSymbol.SLB: StockClassification(
        sector=Sector.ENERGY,
        industry=Industry.ENERGY_EQUIPMENT_SERVICES,
        sub_industry=SubIndustry.OIL_GAS_EQUIPMENT_SERVICES
    ),
    StockSymbol.STZ: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.BEVERAGES,
        sub_industry=SubIndustry.DISTILLERS_VINTNERS
    ),
    StockSymbol.SWKS: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT,
        sub_industry=SubIndustry.SEMICONDUCTORS
    ),
    StockSymbol.T: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.DIVERSIFIED_TELECOMMUNICATION_SERVICES,
        sub_industry=SubIndustry.INTEGRATED_TELECOMMUNICATION_SERVICES
    ),
    StockSymbol.TGT: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.CONSUMER_STAPLES_DISTRIBUTION_RETAIL,
        sub_industry=SubIndustry.CONSUMER_STAPLES_MERCHANDISE_RETAIL
    ),
    StockSymbol.TJX: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.APPAREL_RETAIL
    ),
    StockSymbol.TSLA: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.AUTOMOBILES,
        sub_industry=SubIndustry.AUTOMOBILE_MANUFACTURERS
    ),
    StockSymbol.TSN: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.FOOD_PRODUCTS,
        sub_industry=SubIndustry.PACKAGED_FOODS_MEATS
    ),
    StockSymbol.TXN: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SEMICONDUCTORS_SEMICONDUCTOR_EQUIPMENT,
        sub_industry=SubIndustry.SEMICONDUCTORS
    ),
    StockSymbol.UAA: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.TEXTILES_APPAREL_LUXURY_GOODS,
        sub_industry=SubIndustry.APPAREL_ACCESSORIES_LUXURY_GOODS
    ),
    StockSymbol.UAL: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.PASSENGER_AIRLINES,
        sub_industry=SubIndustry.PASSENGER_AIRLINES
    ),
    StockSymbol.ULTA: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.OTHER_SPECIALTY_RETAIL
    ),
    StockSymbol.UNH: StockClassification(
        sector=Sector.HEALTH_CARE,
        industry=Industry.HEALTH_CARE_PROVIDERS_SERVICES,
        sub_industry=SubIndustry.MANAGED_HEALTH_CARE
    ),
    StockSymbol.UNP: StockClassification(
        sector=Sector.INDUSTRIALS,
        industry=Industry.GROUND_TRANSPORTATION,
        sub_industry=SubIndustry.RAIL_TRANSPORTATION
    ),
    StockSymbol.USB: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.BANKS,
        sub_industry=SubIndustry.DIVERSIFIED_BANKS
    ),
    StockSymbol.V: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.FINANCIAL_SERVICES,
        sub_industry=SubIndustry.TRANSACTION_PAYMENT_PROCESSING_SERVICES
    ),
    StockSymbol.VFC: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.TEXTILES_APPAREL_LUXURY_GOODS,
        sub_industry=SubIndustry.APPAREL_ACCESSORIES_LUXURY_GOODS
    ),
    StockSymbol.VMC: StockClassification(
        sector=Sector.MATERIALS,
        industry=Industry.CONSTRUCTION_MATERIALS,
        sub_industry=SubIndustry.CONSTRUCTION_MATERIALS
    ),
    StockSymbol.VZ: StockClassification(
        sector=Sector.COMMUNICATION_SERVICES,
        industry=Industry.DIVERSIFIED_TELECOMMUNICATION_SERVICES,
        sub_industry=SubIndustry.INTEGRATED_TELECOMMUNICATION_SERVICES
    ),
    StockSymbol.WDAY: StockClassification(
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.SOFTWARE,
        sub_industry=SubIndustry.APPLICATION_SOFTWARE
    ),
    StockSymbol.WFC: StockClassification(
        sector=Sector.FINANCIALS,
        industry=Industry.BANKS,
        sub_industry=SubIndustry.DIVERSIFIED_BANKS
    ),
    StockSymbol.WMT: StockClassification(
        sector=Sector.CONSUMER_STAPLES,
        industry=Industry.CONSUMER_STAPLES_DISTRIBUTION_RETAIL,
        sub_industry=SubIndustry.CONSUMER_STAPLES_MERCHANDISE_RETAIL
    ),
    StockSymbol.WSM: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.SPECIALTY_RETAIL,
        sub_industry=SubIndustry.HOMEFURNISHING_RETAIL
    ),
    StockSymbol.WYNN: StockClassification(
        sector=Sector.CONSUMER_DISCRETIONARY,
        industry=Industry.HOTELS_RESTAURANTS_LEISURE,
        sub_industry=SubIndustry.CASINOS_GAMING
    ),
    StockSymbol.XOM: StockClassification(
        sector=Sector.ENERGY,
        industry=Industry.OIL_GAS_CONSUMABLE_FUELS,
        sub_industry=SubIndustry.INTEGRATED_OIL_GAS
    ),
}


class StockInfoModel(BaseModel):
    """Pydantic model for stock information with validation."""
    symbol: StockSymbol
    sector: Sector
    industry: Industry
    sub_industry: SubIndustry

    class Config:
        use_enum_values = True


def get_stock_classification(symbol):
    # type: (str) -> Optional[StockClassification]
    """
    Get the classification for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (will be cleaned and normalized)
        
    Returns:
        StockClassification if found, None otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return None
        
    try:
        # Clean and normalize the symbol
        cleaned_symbol = symbol.strip().upper()
        normalized = normalize_symbol(cleaned_symbol)
        stock_symbol = StockSymbol(normalized)
        return STOCK_CLASSIFICATIONS.get(stock_symbol)
    except (ValueError, AttributeError):
        return None


def get_stocks_by_sector(sector):
    # type: (Sector) -> List[StockSymbol]
    """
    Get all stock symbols belonging to a specific sector.

    Args:
        sector: The sector to filter by

    Returns:
        List of stock symbols in that sector
    """
    return [
        symbol for symbol, classification in STOCK_CLASSIFICATIONS.items()
        if classification.sector == sector
    ]


def get_stocks_by_industry(industry):
    # type: (Industry) -> List[StockSymbol]
    """
    Get all stock symbols belonging to a specific industry.

    Args:
        industry: The industry to filter by

    Returns:
        List of stock symbols in that industry
    """
    return [
        symbol for symbol, classification in STOCK_CLASSIFICATIONS.items()
        if classification.industry == industry
    ]


def get_stocks_by_sub_industry(sub_industry):
    # type: (SubIndustry) -> List[StockSymbol]
    """
    Get all stock symbols belonging to a specific sub-industry.

    Args:
        sub_industry: The sub-industry to filter by

    Returns:
        List of stock symbols in that sub-industry
    """
    return [
        symbol for symbol, classification in STOCK_CLASSIFICATIONS.items()
        if classification.sub_industry == sub_industry
    ]


def normalize_symbol(symbol: str) -> str:
    """
    Normalize ticker symbols for mapping to handle various formats.
    
    Examples:
        BRK/B -> BRK.B
        BRK.B -> BRK.B  
        BRK_B -> BRK.B
        brk/b -> BRK.B
    """
    if not symbol or not isinstance(symbol, str):
        return symbol
        
    # Clean whitespace and convert to upper
    cleaned = symbol.strip().upper()
    
    # Handle BRK special case - normalize all variants to BRK.B
    if cleaned.startswith('BRK') and len(cleaned) == 5:
        if cleaned[3] in ['/', '_', '.']:
            return 'BRK.B'
    
    # For other symbols, replace slashes and underscores with dots for consistency
    # This maintains compatibility while standardizing format
    normalized = cleaned.replace('/', '.').replace('_', '.')
    
    return normalized


# Example usage and validation
if __name__ == "__main__":
    # Example 1: Get classification for a specific stock
    apple_classification = get_stock_classification("AAPL")
    if apple_classification:
        print(
            "AAPL: {} -> {} -> {}".format(
                apple_classification.sector.value,
                apple_classification.industry.value,
                apple_classification.sub_industry.value
            )
        )

    # Example 2: Get all tech stocks
    tech_stocks = get_stocks_by_sector(Sector.INFORMATION_TECHNOLOGY)
    print("Information Technology stocks: {}".format([stock.value for stock in tech_stocks]))

    # Example 3: Validate with Pydantic model
    stock_info = StockInfoModel(
        symbol=StockSymbol.AAPL,
        sector=Sector.INFORMATION_TECHNOLOGY,
        industry=Industry.TECHNOLOGY_HARDWARE_STORAGE_PERIPHERALS,
        sub_industry=SubIndustry.TECHNOLOGY_HARDWARE_STORAGE_PERIPHERALS
    )
    print("Validated stock info: {}".format(stock_info.dict()))