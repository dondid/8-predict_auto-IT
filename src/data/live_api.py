import yfinance as yf
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

# Mapping Common Car Brands to Stock Tickers
BRAND_TO_TICKER = {
    'alfa-romero': 'STLA', # Stellantis
    'audi': 'VWAGY',       # VW Group
    'bmw': 'BMW.DE',
    'chevrolet': 'GM',
    'dodge': 'STLA',
    'honda': 'HMC',
    'isuzu': '7202.T',
    'jaguar': 'TTM',       # Tata Motors
    'mazda': 'MZDAY',
    'mercedes-benz': 'MBG.DE',
    'mercury': 'F',        # Defunct, Ford
    'mitsubishi': 'MSBHF',
    'nissan': 'NSANY',
    'peugot': 'STLA',
    'plymouth': 'STLA',    # Defunct
    'porsche': 'POAHY',
    'renault': 'RNO.PA',
    'saab': 'NEVS',        # Complicated
    'subaru': 'FUJHY',
    'toyota': 'TM',
    'volkswagen': 'VWAGY',
    'volvo': 'VLVLY'
}

class LiveMarket:
    def __init__(self):
        pass

    def get_stock_data(self, brand):
        """
        Fetches live stock data for the brand's parent company.
        Returns: dict with price, change, and news.
        """
        ticker_symbol = BRAND_TO_TICKER.get(brand.lower())
        if not ticker_symbol:
            return None

        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.fast_info
            
            # Get latest price
            current_price = info.last_price
            prev_close = info.previous_close
            change_percent = ((current_price - prev_close) / prev_close) * 100
            
            # Get News (limit 3)
            news = ticker.news[:3] if ticker.news else []
            processed_news = []
            for item in news:
                processed_news.append({
                    'title': item.get('title'),
                    'link': item.get('link'),
                    'publisher': item.get('publisher'),
                    'relatedTickers': item.get('relatedTickers')
                })

            return {
                'symbol': ticker_symbol,
                'price': current_price,
                'change_p': change_percent,
                'currency': info.currency,
                'news': processed_news
            }
        except Exception as e:
            logger.error(f"Stock fetch error for {brand}: {e}")
            return None
            
    def get_safety_recalls_sample(self, brand):
        """
        Fetches sample recall data (simulated or via simple Public API query if possible).
        For now, we return a structured placeholder referencing NHTSA data 
        which the AI can then elaborate on.
        """
        # NHTSA API is complex for generic "Brand" query without Model/Year.
        # We will let the AI handle the specifics, but we return the link.
        return {
            'source': 'NHTSA (US Gov)',
            'link': f'https://www.nhtsa.gov/recalls?make={brand}'
        }
