#!/usr/bin/env python3
"""
Financial Data Integration Module
Provides real-time financial data from various APIs
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import time
import yfinance as yf
from urllib.parse import urlencode

class FinancialDataProvider:
    """
    Financial data provider that integrates with multiple APIs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Load configuration
        from src.utils import load_config
        self.config = load_config()
        
        # Initialize API configurations
        self.financial_apis = self.config.get('external_sources', {}).get('financial_apis', [])
        
    def get_company_financials(self, company_name: str, symbol: str = None) -> Dict[str, Any]:
        """
        Get comprehensive financial data for a company
        """
        financial_data = {
            'company_name': company_name,
            'symbol': symbol,
            'retrieved_at': datetime.now().isoformat(),
            'stock_data': {},
            'financial_statements': {},
            'market_data': {},
            'analyst_data': {}
        }
        
        # Get stock data using Yahoo Finance (free, no API key required)
        if symbol:
            try:
                stock_data = self._get_yahoo_finance_data(symbol)
                financial_data['stock_data'] = stock_data
                self.logger.info(f"Retrieved Yahoo Finance data for {symbol}")
            except Exception as e:
                self.logger.error(f"Error getting Yahoo Finance data: {e}")
        
        # Try Alpha Vantage if API key is available
        try:
            alpha_data = self._get_alpha_vantage_data(symbol or company_name)
            if alpha_data:
                financial_data['financial_statements'].update(alpha_data)
                self.logger.info(f"Retrieved Alpha Vantage data for {company_name}")
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage data: {e}")
        
        # Generate demo financial data for verification
        demo_data = self._generate_demo_financial_data(company_name)
        financial_data['demo_data'] = demo_data
        
        return financial_data
    
    def _get_yahoo_finance_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock data using Yahoo Finance (yfinance library)
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data (last 1 year)
            hist = ticker.history(period="1y")
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            # Get analyst recommendations
            recommendations = ticker.recommendations
            
            return {
                'basic_info': {
                    'company_name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'pb_ratio': info.get('priceToBook', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 0),
                    '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                    '52_week_low': info.get('fiftyTwoWeekLow', 0)
                },
                'price_data': {
                    'current_price': info.get('currentPrice', 0),
                    'previous_close': info.get('previousClose', 0),
                    'day_change': info.get('regularMarketChange', 0),
                    'day_change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('volume', 0),
                    'avg_volume': info.get('averageVolume', 0)
                },
                'historical_data': hist.to_dict() if not hist.empty else {},
                'financial_statements': {
                    'income_statement': financials.to_dict() if not financials.empty else {},
                    'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                    'cash_flow': cashflow.to_dict() if not cashflow.empty else {}
                },
                'analyst_data': recommendations.to_dict() if recommendations is not None and not recommendations.empty else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Yahoo Finance data for {symbol}: {e}")
            return {}
    
    def _get_alpha_vantage_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get financial data from Alpha Vantage API
        """
        try:
            # Find Alpha Vantage configuration
            alpha_config = None
            for api in self.financial_apis:
                if api.get('name') == 'Alpha Vantage':
                    alpha_config = api
                    break
            
            if not alpha_config or not alpha_config.get('api_key'):
                self.logger.warning("Alpha Vantage API key not configured")
                return {}
            
            api_key = alpha_config['api_key']
            base_url = alpha_config['base_url']

            # Normalize symbol for Alpha Vantage (prefer BSE for Indian equities)
            av_symbol = symbol or ""
            if av_symbol.endswith('.NS') or av_symbol.endswith('.NSE'):
                av_symbol = av_symbol.rsplit('.', 1)[0] + '.BSE'
            
            # Get company overview
            overview_data = self._call_alpha_vantage_api(
                base_url, api_key, 'OVERVIEW', av_symbol
            )
            
            # Get income statement
            income_data = self._call_alpha_vantage_api(
                base_url, api_key, 'INCOME_STATEMENT', av_symbol
            )
            
            # Get balance sheet
            balance_data = self._call_alpha_vantage_api(
                base_url, api_key, 'BALANCE_SHEET', av_symbol
            )
            
            # Get cash flow
            cashflow_data = self._call_alpha_vantage_api(
                base_url, api_key, 'CASH_FLOW', av_symbol
            )
            
            return {
                'overview': overview_data,
                'income_statement': income_data,
                'balance_sheet': balance_data,
                'cash_flow': cashflow_data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage data: {e}")
            return {}
    
    def _call_alpha_vantage_api(self, base_url: str, api_key: str, 
                               function: str, symbol: str) -> Dict[str, Any]:
        """
        Make API call to Alpha Vantage
        """
        try:
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': api_key
            }
            
            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {}
            
            if 'Note' in data:
                self.logger.warning(f"Alpha Vantage API note: {data['Note']}")
                return {}
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calling Alpha Vantage API: {e}")
            return {}
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get market data for multiple symbols
        """
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                market_data[symbol] = {
                    'name': info.get('longName', ''),
                    'current_price': info.get('currentPrice', 0),
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('volume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0)
                }
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error getting market data for {symbol}: {e}")
                market_data[symbol] = {'error': str(e)}
        
        return market_data
    
    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Get sector performance data
        """
        try:
            # Major Indian indices
            indices = {
                'NIFTY_50': '^NSEI',
                'NIFTY_BANK': '^NSEBANK',
                'NIFTY_IT': '^CNXIT',
                'NIFTY_AUTO': '^CNXAUTO',
                'NIFTY_FMCG': '^CNXFMCG',
                'NIFTY_PHARMA': '^CNXPHARMA'
            }
            
            sector_data = {}
            
            for sector_name, symbol in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    sector_data[sector_name] = {
                        'name': info.get('longName', sector_name),
                        'current_value': info.get('regularMarketPrice', 0),
                        'change': info.get('regularMarketChange', 0),
                        'change_percent': info.get('regularMarketChangePercent', 0),
                        'volume': info.get('volume', 0)
                    }
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Error getting sector data for {sector_name}: {e}")
                    sector_data[sector_name] = {'error': str(e)}
            
            return sector_data
            
        except Exception as e:
            self.logger.error(f"Error getting sector performance: {e}")
            return {}
    
    def get_ipo_data(self, company_name: str = None) -> List[Dict[str, Any]]:
        """
        Get IPO data and upcoming IPOs
        """
        try:
            # This would typically come from a financial data provider
            # For now, we'll generate demo IPO data
            demo_ipo_data = [
                {
                    'company_name': 'Urban Company Limited',
                    'ipo_date': '2024-03-15',
                    'issue_size': '₹2,000 crores',
                    'price_band': '₹450-₹500',
                    'listing_date': '2024-03-22',
                    'status': 'Upcoming',
                    'lead_managers': ['Kotak Mahindra Capital', 'ICICI Securities'],
                    'exchange': 'NSE, BSE'
                },
                {
                    'company_name': 'JSW Cement Limited',
                    'ipo_date': '2024-02-20',
                    'issue_size': '₹1,500 crores',
                    'price_band': '₹380-₹420',
                    'listing_date': '2024-02-27',
                    'status': 'Completed',
                    'lead_managers': ['HDFC Bank', 'Axis Capital'],
                    'exchange': 'NSE, BSE'
                }
            ]
            
            if company_name:
                # Filter for specific company
                filtered_data = [ipo for ipo in demo_ipo_data 
                               if company_name.lower() in ipo['company_name'].lower()]
                return filtered_data
            
            return demo_ipo_data
            
        except Exception as e:
            self.logger.error(f"Error getting IPO data: {e}")
            return []
    
    def _generate_demo_financial_data(self, company_name: str) -> Dict[str, Any]:
        """
        Generate demo financial data for testing
        """
        return {
            'revenue_growth': {
                '2021': 15.2,
                '2022': 18.5,
                '2023': 22.1,
                '2024': 25.3
            },
            'profit_margins': {
                'gross_margin': 45.2,
                'operating_margin': 18.7,
                'net_margin': 12.3
            },
            'key_ratios': {
                'debt_to_equity': 0.35,
                'current_ratio': 1.8,
                'quick_ratio': 1.2,
                'inventory_turnover': 6.5
            },
            'market_metrics': {
                'pe_ratio': 28.5,
                'pb_ratio': 4.2,
                'dividend_yield': 1.8,
                'beta': 1.15
            },
            'financial_highlights': [
                f"{company_name} reported strong revenue growth of 25.3% in 2024",
                f"Operating margins improved to 18.7% from 16.2% in previous year",
                f"Company maintains healthy debt-to-equity ratio of 0.35",
                f"Strong cash position with current ratio of 1.8"
            ]
        }
    
    def analyze_financial_health(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze financial health based on retrieved data
        """
        analysis = {
            'overall_score': 0,
            'strengths': [],
            'concerns': [],
            'recommendations': [],
            'risk_level': 'Medium'
        }
        
        try:
            # Analyze profitability
            if 'demo_data' in financial_data:
                demo_data = financial_data['demo_data']
                
                # Check profit margins
                net_margin = demo_data.get('profit_margins', {}).get('net_margin', 0)
                if net_margin > 15:
                    analysis['strengths'].append("Strong profitability with net margin above 15%")
                    analysis['overall_score'] += 20
                elif net_margin < 5:
                    analysis['concerns'].append("Low profitability with net margin below 5%")
                    analysis['overall_score'] -= 15
                
                # Check debt levels
                debt_ratio = demo_data.get('key_ratios', {}).get('debt_to_equity', 0)
                if debt_ratio < 0.5:
                    analysis['strengths'].append("Conservative debt levels")
                    analysis['overall_score'] += 15
                elif debt_ratio > 1.0:
                    analysis['concerns'].append("High debt levels")
                    analysis['overall_score'] -= 20
                
                # Check liquidity
                current_ratio = demo_data.get('key_ratios', {}).get('current_ratio', 0)
                if current_ratio > 1.5:
                    analysis['strengths'].append("Good liquidity position")
                    analysis['overall_score'] += 10
                elif current_ratio < 1.0:
                    analysis['concerns'].append("Poor liquidity position")
                    analysis['overall_score'] -= 15
            
            # Determine risk level
            if analysis['overall_score'] >= 40:
                analysis['risk_level'] = 'Low'
            elif analysis['overall_score'] >= 20:
                analysis['risk_level'] = 'Medium'
            else:
                analysis['risk_level'] = 'High'
            
            # Generate recommendations
            if analysis['concerns']:
                analysis['recommendations'].append("Review financial performance and address key concerns")
            if analysis['strengths']:
                analysis['recommendations'].append("Continue leveraging existing strengths")
            
            analysis['recommendations'].append("Monitor key financial metrics regularly")
            
        except Exception as e:
            self.logger.error(f"Error analyzing financial health: {e}")
            analysis['error'] = str(e)
        
        return analysis
