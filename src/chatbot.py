import google.generativeai as genai
import openai
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import os
import sys

# Import Enhanced News Scraper
NEWS_SCRAPER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'News Web Scraper')
if NEWS_SCRAPER_PATH not in sys.path:
    sys.path.insert(0, NEWS_SCRAPER_PATH)

try:
    from newsscraper.enhanced_scraper import EnhancedNewsScraper
    ENHANCED_NEWS_AVAILABLE = True
except ImportError:
    try:
        from newsscraper.scraper_new import NewsScraperAPI
        ENHANCED_NEWS_AVAILABLE = True
        EnhancedNewsScraper = NewsScraperAPI  # Alias for compatibility
    except ImportError:
        ENHANCED_NEWS_AVAILABLE = False
        print("⚠️ Enhanced News Scraper not available")

class DRHPChatbot:
    """
    AI-powered chatbot for DRHP insights and analysis using OpenAI or Gemini API
    """
    
    def __init__(self, api_key: str = None, api_type: str = "openai"):
        self.logger = logging.getLogger(__name__)
        self.api_type = api_type.lower()
        self.model = None
        self.client = None
        
        # Initialize Enhanced News Scraper for dynamic news fetching
        self.news_scraper = None
        if ENHANCED_NEWS_AVAILABLE:
            try:
                self.news_scraper = EnhancedNewsScraper()
                self.logger.info("✅ Enhanced News Scraper integrated into chatbot")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize Enhanced News Scraper: {e}")
        
        # Initialize API client
        if api_key:
            self._initialize_api_client(api_key)
        else:
            # Try to load from config or environment
            self._load_api_from_config()
        
        # Chatbot configuration
        self.max_tokens = 1000
        self.temperature = 0.7
        
        # Conversation context
        self.conversation_history = []
        
        # DRHP-specific knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Response templates for common queries
        self.response_templates = {
            'anomaly_explanation': self._get_anomaly_explanation_template(),
            'plagiarism_explanation': self._get_plagiarism_explanation_template(),
            'section_analysis': self._get_section_analysis_template(),
            'general_help': self._get_general_help_template()
        }
    
    def _initialize_api_client(self, api_key: str):
        """Initialize API client based on API type"""
        try:
            if self.api_type == "openai":
                self.client = openai.OpenAI(api_key=api_key)
                self.model = "gpt-3.5-turbo"  # Default model
                self.logger.info("OpenAI API client initialized successfully")
            elif self.api_type == "gemini":
                genai.configure(api_key=api_key)
                # Get model name from config
                from src.utils import load_config
                config = load_config()
                model_name = config.get('chatbot_settings', {}).get('model', 'gemini-1.5-flash')
                self.model = genai.GenerativeModel(model_name)
                self.logger.info(f"Gemini API client initialized successfully with model: {model_name}")
            else:
                self.logger.warning(f"Unsupported API type: {self.api_type}")
                self.model = None
        except Exception as e:
            self.logger.error(f"Failed to initialize API client: {e}")
            self.model = None
    
    def _load_api_from_config(self):
        """Load API configuration from config file or environment variables"""
        try:
            from src.utils import load_config
            config = load_config()
            
            # Check for Gemini API key first (preferred)
            gemini_key = config.get('gemini_api_key') or os.getenv('GEMINI_API_KEY')
            if gemini_key:
                self.api_type = "gemini"
                self._initialize_api_client(gemini_key)
                return
            
            # Fallback to OpenAI API key only if Gemini is not available
            openai_key = config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            if openai_key:
                self.api_type = "openai"
                self._initialize_api_client(openai_key)
                return
            
            self.logger.warning("No API key found. Chatbot will use fallback responses.")
            self.model = None
            
        except Exception as e:
            self.logger.warning(f"Could not load API configuration: {e}")
            self.model = None
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """
        Initialize knowledge base with DRHP-specific information
        """
        return {
            'drhp_sections': {
                'business_overview': 'Contains information about the company\'s business model, operations, and strategic direction',
                'financial_statements': 'Includes audited financial statements, balance sheets, profit & loss statements, and cash flow statements',
                'risk_factors': 'Lists potential risks and uncertainties that could affect the company\'s performance',
                'management_discussion': 'Management\'s analysis of financial performance and business outlook',
                'use_of_proceeds': 'Details how the IPO proceeds will be utilized',
                'capital_structure': 'Information about the company\'s share capital and ownership structure',
                'promoter_details': 'Information about the company\'s promoters and their background',
                'directors_profile': 'Profiles of board members and key management personnel',
                'auditors_report': 'Independent auditor\'s report on financial statements'
            },
            'common_anomalies': {
                'financial_anomaly': 'Unrealistic financial ratios or missing financial data',
                'risk_anomaly': 'High number of risk keywords or missing risk categories',
                'structural_anomaly': 'Missing critical sections or insufficient content',
                'content_anomaly': 'Excessive repetition or formatting issues',
                'inconsistency': 'Date or number inconsistencies across sections'
            },
            'plagiarism_types': {
                'internal_plagiarism': 'Similar content within the document',
                'external_plagiarism': 'Content similar to external sources',
                'boilerplate_content': 'Generic template-like content',
                'template_content': 'Unfilled template placeholders'
            }
        }
    
    def get_response(self, user_input: str, enhanced_context: Dict = None) -> str:
        """
        Generate response to user input using OpenAI or Gemini API with enhanced context
        """
        try:
            # Extract analysis results and other context
            analysis_results = enhanced_context.get('analysis_results', {}) if enhanced_context else {}
            extracted_sections = enhanced_context.get('extracted_sections', {}) if enhanced_context else {}
            uploaded_file = enhanced_context.get('uploaded_file') if enhanced_context else None
            web_scraper = enhanced_context.get('web_scraper') if enhanced_context else None
            
            # Analyze user intent
            intent = self._analyze_intent(user_input)
            self.logger.info(f"🎯 Detected intent: {intent} for query: {user_input[:50]}...")
            
            # Generate context-aware response
            if intent == 'news_summary_query':
                # NEW: Handle news summary requests (e.g., "give 50 lines summary of news on X")
                return self._handle_news_summary_query(user_input)
            elif intent == 'compare_query':
                # NEW: Handle comparison queries (e.g., "compare risk factors with news")
                return self._handle_compare_query(user_input, analysis_results, extracted_sections)
            elif intent == 'company_info_query':
                # NEW: Handle "tell me about X company" queries
                return self._handle_company_info_query(user_input)
            elif intent == 'anomaly_query' and analysis_results:
                return self._handle_anomaly_query(user_input, analysis_results)
            elif intent == 'plagiarism_query' and analysis_results:
                return self._handle_plagiarism_query(user_input, analysis_results)
            elif intent == 'section_query' and (analysis_results or extracted_sections):
                return self._handle_section_query(user_input, analysis_results, extracted_sections)
            elif intent == 'financial_query' and (analysis_results or extracted_sections):
                return self._handle_financial_query(user_input, analysis_results, extracted_sections)
            elif intent == 'web_search_query' and web_scraper:
                return self._handle_web_search_query(user_input, web_scraper, uploaded_file)
            elif intent == 'news_query':
                # Use enhanced scraper for news queries
                return self._handle_news_query_enhanced(user_input)
            elif intent == 'company_query' and web_scraper:
                return self._handle_company_query(user_input, web_scraper, uploaded_file)
            elif intent == 'general_query':
                return self._handle_general_query(user_input)
            else:
                return self._handle_fallback_query(user_input)
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    def _generate_ai_response(self, prompt: str) -> str:
        """
        Generate response using the configured AI API
        """
        try:
            if self.api_type == "openai" and self.client:
                return self._generate_openai_response(prompt)
            elif self.api_type == "gemini" and self.model:
                return self._generate_gemini_response(prompt)
            else:
                return self._handle_fallback_query(prompt)
        except Exception as e:
            self.logger.error(f"AI API error: {e}")
            return self._handle_fallback_query(prompt)
    
    def _generate_openai_response(self, prompt: str) -> str:
        """
        Generate response using OpenAI API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert DRHP (Draft Red Herring Prospectus) analyst assistant. Provide detailed, accurate, and helpful responses about DRHP analysis, anomalies, plagiarism detection, and financial document insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self._handle_fallback_query(prompt)
    
    def _generate_gemini_response(self, prompt: str) -> str:
        """
        Generate response using Gemini API
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return self._handle_fallback_query(prompt)
    
    def _analyze_intent(self, user_input: str) -> str:
        """
        Analyze user intent from input with improved keyword detection
        """
        input_lower = user_input.lower()
        
        # NEW: Summary requests (highest priority)
        summary_keywords = [
            'summarize', 'summary', 'give.*summary', 'give.*lines summary',
            '\\d+ lines summary', 'brief summary', 'quick summary'
        ]
        if any(re.search(keyword, input_lower) for keyword in summary_keywords):
            # Check if it's about news or DRHP section
            if any(kw in input_lower for kw in ['news', 'articles', 'latest']):
                return 'news_summary_query'
            else:
                return 'section_query'  # DRHP section summary
        
        # NEW: Compare/Analysis keywords (for comparative questions)
        compare_keywords = [
            'compare', 'comparison', 'versus', 'vs', 'difference between', 
            'contrast', 'how does', 'relate', 'correlation', 'match',
            'compare with news', 'compare to news', 'draw conclusion'
        ]
        if any(keyword in input_lower for keyword in compare_keywords):
            return 'compare_query'
        
        # NEW: "Tell me about" company queries
        tell_me_keywords = [
            'tell me about', 'what is', 'who is', 'explain', 'describe',
            'information on', 'details about', 'give me info', 'info on'
        ]
        # Check if asking about a specific company
        if any(keyword in input_lower for keyword in tell_me_keywords):
            # Check if company name follows
            for company in ['reliance', 'urban company', 'tcs', 'infosys', 'wipro', 'hdfc', 'icici', 'sbi', 'physicswallah', 'byjus']:
                if company in input_lower:
                    return 'company_info_query'
        
        # Anomaly-related keywords (expanded)
        anomaly_keywords = [
            'anomaly', 'anomalies', 'issue', 'problem', 'error', 'warning', 'alert',
            'concern', 'risk', 'problematic', 'inconsistent', 'unusual', 'suspicious',
            'red flag', 'flag', 'issue', 'trouble', 'difficulty', 'challenge'
        ]
        if any(keyword in input_lower for keyword in anomaly_keywords):
            return 'anomaly_query'
        
        # Plagiarism-related keywords (expanded)
        plagiarism_keywords = [
            'plagiarism', 'similarity', 'copy', 'duplicate', 'original', 'originality',
            'similar', 'match', 'identical', 'copied', 'borrowed', 'stolen',
            'template', 'boilerplate', 'generic', 'standard', 'common'
        ]
        if any(keyword in input_lower for keyword in plagiarism_keywords):
            return 'plagiarism_query'
        
        # Section-related keywords (expanded)
        section_keywords = [
            'section', 'business', 'management', 'proceeds',
            'overview', 'factors', 'discussion',
            'capital', 'structure', 'promoter', 'director', 'auditor',
            'balance sheet', 'profit loss', 'cash flow', 'md&a'
        ]
        if any(keyword in input_lower for keyword in section_keywords):
            return 'section_query'
        
        # Financial-related keywords
        financial_keywords = [
            'financial', 'revenue', 'profit', 'income', 'statement', 'proceeds',
            'earnings', 'growth', 'performance', 'trends', 'ratios', 'metrics',
            'balance sheet', 'profit loss', 'cash flow', 'financials', 'ipo proceeds'
        ]
        if any(keyword in input_lower for keyword in financial_keywords):
            return 'financial_query'
        
        # News-related keywords
        news_keywords = [
            'news', 'headlines', 'articles', 'latest updates', 'latest news',
            'business news', 'financial news', 'market news', 'current news',
            'what\'s happening', 'recent news', 'today\'s news'
        ]
        if any(keyword in input_lower for keyword in news_keywords):
            return 'news_query'
        
        # Always default to general_query for better AI handling
        return 'general_query'
    
    def _handle_anomaly_query(self, user_input: str, analysis_results: Dict) -> str:
        """
        Handle queries about anomalies using AI
        """
        anomalies = analysis_results.get('anomalies', [])
        
        # Create context for AI
        context = f"""
        User Question: {user_input}
        
        Anomaly Analysis Results:
        - Total anomalies found: {len(anomalies)}
        - Anomaly details: {json.dumps(anomalies, indent=2) if anomalies else "No anomalies detected"}
        
        Please provide a detailed, helpful response about the anomalies found in this DRHP document. 
        If no anomalies were found, explain what this means and what the user should focus on instead.
        Be specific and actionable in your advice.
        """
        
        return self._generate_ai_response(context)
    
    def _handle_plagiarism_query(self, user_input: str, analysis_results: Dict) -> str:
        """
        Handle queries about plagiarism using AI
        """
        plagiarism_cases = analysis_results.get('plagiarism', [])
        
        # Create context for AI
        context = f"""
        User Question: {user_input}
        
        Plagiarism Analysis Results:
        - Total plagiarism cases found: {len(plagiarism_cases)}
        - Plagiarism details: {json.dumps(plagiarism_cases, indent=2) if plagiarism_cases else "No plagiarism detected"}
        
        Please provide a detailed, helpful response about the plagiarism analysis of this DRHP document. 
        If no plagiarism was found, explain what this means and how to maintain originality.
        If plagiarism was found, explain the implications and provide actionable recommendations.
        Be specific and professional in your advice.
        """
        
        return self._generate_ai_response(context)
    
    def _handle_section_query(self, user_input: str, analysis_results: Dict, extracted_sections: Dict = None) -> str:
        """
        Handle queries about document sections using AI
        """
        # Use extracted_sections if available, otherwise fall back to analysis_results
        sections = extracted_sections if extracted_sections else analysis_results.get('sections', {})
        
        # Create context for AI
        context = f"""
        User Question: {user_input}
        
        Document Sections Available:
        - Total sections found: {len(sections)}
        - Section names: {list(sections.keys()) if sections else "No sections found"}
        - Section details: {json.dumps(sections, indent=2) if sections else "No section data available"}
        
        Please provide a detailed, helpful response about the sections in this DRHP document. 
        Explain what each section contains, its importance, and provide insights about the document structure.
        Be specific and educational in your response.
        """
        
        return self._generate_ai_response(context)
    
    def _handle_general_query(self, user_input: str) -> str:
        """
        Handle general queries about DRHP analysis using AI with real-time data integration
        """
        # Check if the query is about a specific company or financial data
        company_name = self._extract_company_name(user_input)
        
        if company_name:
            # Fetch real-time data for the company
            real_time_data = self._fetch_real_time_data(company_name)
            # Return the real-time summary directly to avoid generic AI responses
            header = f"Real-time summary for {company_name}:\n\n"
            return header + real_time_data
        else:
            # General DRHP analysis context
            context = f"""
            User Question: {user_input}
            
            Context: You are an expert DRHP (Draft Red Herring Prospectus) analyst assistant. 
            The user is asking a general question about DRHP analysis or concepts.
            
            Please provide a helpful, detailed response that directly answers their question.
            If they're asking about DRHP concepts, explain them clearly.
            If they're asking about analysis capabilities, explain what you can help with.
            Be conversational, informative, and specific to their question.
            """
        
        return self._generate_ai_response(context)
    
    def _handle_fallback_query(self, user_input: str) -> str:
        """
        Handle queries that don't match specific patterns using AI
        """
        # Create context for AI
        context = f"""
        User Question: {user_input}
        
        Context: You are an expert DRHP (Draft Red Herring Prospectus) analyst assistant. 
        The user has asked a question that doesn't clearly fit into anomaly, plagiarism, or section analysis categories.
        
        Please provide a helpful response that:
        1. Directly addresses their question
        2. Explains how it relates to DRHP analysis if relevant
        3. Offers to help with specific DRHP analysis tasks
        4. Be conversational and helpful
        
        If the question is completely unrelated to DRHP analysis, politely redirect them to DRHP-related topics.
        """
        
        return self._generate_ai_response(context)
    
    def _extract_company_name(self, user_input: str) -> str:
        """
        Extract company name from user input - with cleaning
        """
        # Common company names and patterns (check first)
        company_patterns = [
            'PhysicsWallah', 'Urban Company', 'JSW Cement', 'Reliance', 'TCS', 'Infosys', 'HDFC Bank',
            'Wipro', 'Bharti Airtel', 'ITC', 'HDFC', 'ICICI Bank', 'Kotak Mahindra',
            'Asian Paints', 'Titan', 'Maruti Suzuki', 'Mahindra', 'Bajaj Finance',
            'Axis Bank', 'SBI', 'Nestle', 'Hindustan Unilever', 'Sun Pharma', 'Lenskart'
        ]
        
        input_lower = user_input.lower()
        
        # Check for known companies first
        for company in company_patterns:
            if company.lower() in input_lower:
                return company
        
        # Try to extract from context words
        for keyword in ['about', 'on', 'regarding', 'for', 'of']:
            if keyword in input_lower:
                parts = user_input.split(keyword, 1)
                if len(parts) > 1:
                    # Get text after the keyword
                    potential_name = parts[1].strip()
                    # Clean common noise words
                    potential_name = re.sub(r'^(the|a|an)\s+', '', potential_name, flags=re.IGNORECASE)
                    potential_name = potential_name.strip('?.,!')
                    # Clean DRHP artifacts
                    potential_name = re.sub(r'\s+and\s+.*', '', potential_name)
                    potential_name = re.sub(r'-DRHP.*', '', potential_name, flags=re.IGNORECASE)
                    if potential_name:
                        return potential_name
        
        return None
    
    def _fetch_real_time_data(self, company_name: str) -> str:
        """
        Fetch real-time data for a company
        """
        try:
            from src.scraping.web_scraper import WebScraper
            from src.financial_data import FinancialDataProvider
            # Selenium-based live headlines (optional)
            selenium_available = True
            try:
                from scrapers import scrape_google_news
            except Exception:
                selenium_available = False
            
            # Initialize providers with Selenium support
            web_scraper = WebScraper(use_selenium=True)
            financial_provider = FinancialDataProvider()
            
            # Get company symbol mapping
            symbol_mapping = {
                'Reliance': 'RELIANCE.NS',
                'TCS': 'TCS.NS',
                'Infosys': 'INFY.NS',
                'HDFC Bank': 'HDFCBANK.NS',
                'Wipro': 'WIPRO.NS',
                'Bharti Airtel': 'BHARTIARTL.NS',
                'ITC': 'ITC.NS',
                'HDFC': 'HDFC.NS',
                'ICICI Bank': 'ICICIBANK.NS',
                'Kotak Mahindra': 'KOTAKBANK.NS',
                'Asian Paints': 'ASIANPAINT.NS',
                'Titan': 'TITAN.NS',
                'Maruti Suzuki': 'MARUTI.NS',
                'Mahindra': 'M&M.NS',
                'Bajaj Finance': 'BAJFINANCE.NS',
                'Axis Bank': 'AXISBANK.NS',
                'SBI': 'SBIN.NS',
                'Nestle': 'NESTLEIND.NS',
                'Hindustan Unilever': 'HINDUNILVR.NS',
                'Sun Pharma': 'SUNPHARMA.NS'
            }
            
            symbol = symbol_mapping.get(company_name)
            
            # Fetch comprehensive data
            aggregated_data = web_scraper.aggregate_external_data(
                company_name=company_name,
                symbol=symbol
            )
            
            # Format the data for AI context
            data_summary = f"""
            📰 NEWS ARTICLES ({len(aggregated_data.get('news_articles', []))} found):
            """
            
            # If Selenium available, prepend live Google News headlines
            if selenium_available:
                try:
                    live_headlines: List[str] = scrape_google_news(company_name, max_results=5)
                    if live_headlines and isinstance(live_headlines, list):
                        data_summary += "\nLive Google News headlines:"
                        for i, h in enumerate(live_headlines, 1):
                            data_summary += f"\n• {h}"
                        data_summary += "\n"
                except Exception as e:
                    self.logger.warning(f"Selenium live headlines failed: {e}")

            # Add recent news headlines with URLs
            news_articles = aggregated_data.get('news_articles', [])
            for i, article in enumerate(news_articles[:5], 1):
                data_summary += f"\n{i}. {article.get('title', 'No title')}"
                data_summary += f"\n   Source: {article.get('source', 'Unknown')}"
                data_summary += f"\n   Published: {article.get('publish_date', 'Unknown')}"
                # Add URL if available
                if article.get('url'):
                    data_summary += f"\n   🔗 Link: {article.get('url')}"
                # Add description if available
                if article.get('description'):
                    data_summary += f"\n   Description: {article.get('description')[:150]}..."
            
            # Add financial data
            financial_data = aggregated_data.get('financial_data', {})
            if financial_data.get('stock_data'):
                stock_data = financial_data['stock_data']
                data_summary += f"""
            
            💰 FINANCIAL DATA:
            """
                
                if stock_data.get('basic_info'):
                    info = stock_data['basic_info']
                    data_summary += f"\n• Company: {info.get('company_name', 'N/A')}"
                    data_summary += f"\n• Sector: {info.get('sector', 'N/A')}"
                    data_summary += f"\n• Market Cap: ₹{info.get('market_cap', 0):,}"
                    data_summary += f"\n• P/E Ratio: {info.get('pe_ratio', 0)}"
                
                if stock_data.get('price_data'):
                    price_data = stock_data['price_data']
                    data_summary += f"\n• Current Price: ₹{price_data.get('current_price', 0)}"
                    data_summary += f"\n• Day Change: ₹{price_data.get('day_change', 0)} ({price_data.get('day_change_percent', 0):.2f}%)"
                    data_summary += f"\n• Volume: {price_data.get('volume', 0):,}"
            
            # Add IPO data
            ipo_data = aggregated_data.get('ipo_data', [])
            if ipo_data:
                data_summary += f"""
            
            🚀 IPO INFORMATION:
            """
                for ipo in ipo_data:
                    data_summary += f"\n• Status: {ipo.get('status', 'Unknown')}"
                    data_summary += f"\n• Issue Size: {ipo.get('issue_size', 'Unknown')}"
                    data_summary += f"\n• Price Band: {ipo.get('price_band', 'Unknown')}"
                    data_summary += f"\n• Listing Date: {ipo.get('listing_date', 'Unknown')}"
                    # Add source URL if available
                    if ipo.get('source_url'):
                        data_summary += f"\n• 🔗 Source: {ipo.get('source_url')}"
            
            # Add market data
            market_data = aggregated_data.get('market_data', {})
            if market_data:
                data_summary += f"""
            
            📊 MARKET DATA:
            """
                for symbol, data in market_data.items():
                    if 'error' not in data:
                        data_summary += f"\n• {symbol}: ₹{data.get('current_price', 0)} ({data.get('change_percent', 0):.2f}%)"
            
            # Add regulatory filings and company website
            regulatory_filings = aggregated_data.get('regulatory_filings', [])
            if regulatory_filings:
                data_summary += f"""
            
            📋 REGULATORY FILINGS:
            """
                for filing in regulatory_filings[:3]:
                    data_summary += f"\n• {filing.get('title', 'No title')}"
                    if filing.get('url'):
                        data_summary += f"\n  🔗 Link: {filing.get('url')}"
            
            # Add company website info
            company_website = aggregated_data.get('company_website', {})
            if company_website and company_website.get('url'):
                data_summary += f"""
            
            🌐 COMPANY WEBSITE:
            • 🔗 {company_website.get('url')}
            • Description: {company_website.get('description', 'N/A')[:150]}
            """
            
            # Add useful links section
            data_summary += f"""
            
            🔍 USEFUL LINKS FOR MORE INFORMATION:
            • SEBI Filings: https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes
            • BSE India: https://www.bseindia.com/corporates/List_Scrips.aspx
            • NSE India: https://www.nseindia.com/market-data/live-equity-market
            • IPO Central: https://www.nseindia.com/market-data/upcoming-ipo
            • Moneycontrol IPOs: https://www.moneycontrol.com/ipo/
            """
            
            return data_summary
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data: {e}")
            return f"Error fetching real-time data for {company_name}: {str(e)}"
    
    def _summarize_anomalies(self, anomalies: List[Dict]) -> Dict:
        """
        Summarize anomalies for chatbot response
        """
        summary = {
            'total': len(anomalies),
            'by_severity': {},
            'by_type': {}
        }
        
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'unknown')
            anomaly_type = anomaly.get('type', 'unknown')
            
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            summary['by_type'][anomaly_type] = summary['by_type'].get(anomaly_type, 0) + 1
        
        return summary
    
    def _summarize_plagiarism(self, plagiarism_cases: List[Dict]) -> Dict:
        """
        Summarize plagiarism cases for chatbot response
        """
        summary = {
            'total': len(plagiarism_cases),
            'by_severity': {},
            'by_type': {}
        }
        
        for case in plagiarism_cases:
            severity = case.get('severity', 'unknown')
            case_type = case.get('type', 'unknown')
            
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            summary['by_type'][case_type] = summary['by_type'].get(case_type, 0) + 1
        
        return summary
    
    def _identify_section_from_query(self, user_input: str) -> Optional[str]:
        """
        Identify which section the user is asking about
        """
        input_lower = user_input.lower()
        
        section_mappings = {
            'business': 'business_overview',
            'financial': 'financial_statements',
            'risk': 'risk_factors',
            'management': 'management_discussion',
            'proceeds': 'use_of_proceeds',
            'capital': 'capital_structure',
            'promoter': 'promoter_details',
            'director': 'directors_profile',
            'auditor': 'auditors_report'
        }
        
        for keyword, section in section_mappings.items():
            if keyword in input_lower:
                return section
        
        return None
    
    def _provide_educational_response(self, user_input: str) -> str:
        """
        Provide educational information about DRHP concepts
        """
        input_lower = user_input.lower()
        
        if 'drhp' in input_lower or 'draft red herring prospectus' in input_lower:
            return """A **Draft Red Herring Prospectus (DRHP)** is a preliminary document filed by a company planning to go public. It contains:

📋 **Key Information**:
- Business overview and operations
- Financial statements and performance
- Risk factors and challenges
- Management team and promoters
- Use of IPO proceeds
- Capital structure details

🎯 **Purpose**: To provide potential investors with comprehensive information about the company before the IPO.

📊 **Our Analysis**: We help identify anomalies, plagiarism, and inconsistencies in DRHP documents to ensure quality and compliance."""
        
        elif 'anomaly' in input_lower:
            return """**Anomalies** in DRHP documents are unusual patterns or issues that could indicate problems:

🚨 **Types of Anomalies**:
- **Financial**: Unrealistic ratios, missing data
- **Risk**: High-risk keywords, missing categories  
- **Structural**: Missing sections, insufficient content
- **Content**: Repetition, formatting issues
- **Inconsistency**: Date/number conflicts

🔍 **Our Detection**: We use advanced algorithms to identify these issues automatically."""
        
        elif 'plagiarism' in input_lower:
            return """**Plagiarism Detection** identifies copied or similar content:

🔍 **Detection Methods**:
- **Semantic Similarity**: Using AI to understand meaning
- **Cross-Reference**: Comparing with external sources
- **Internal Analysis**: Finding repeated content within document

⚠️ **Types Found**:
- Internal plagiarism (within document)
- External plagiarism (from other sources)
- Boilerplate content (generic templates)
- Template placeholders (unfilled content)

✅ **Our Analysis**: We provide detailed similarity scores and recommendations."""
        
        else:
            return "I can explain various aspects of DRHP analysis. What specific topic would you like to learn about?"
    
    def _handle_financial_query(self, user_input: str, analysis_results: Dict, extracted_sections: Dict) -> str:
        """
        Handle financial-related queries about the DRHP document
        """
        try:
            # Extract financial information from sections
            financial_sections = {}
            for section_name, section_data in extracted_sections.items():
                if any(keyword in section_name.lower() for keyword in ['financial', 'revenue', 'profit', 'income', 'statement', 'proceeds']):
                    financial_sections[section_name] = section_data
            
            # Build context for financial analysis
            context = f"""
            User Question: {user_input}
            
            Financial Context from DRHP:
            - Available financial sections: {list(financial_sections.keys())}
            - Financial section details: {json.dumps(financial_sections, indent=2) if financial_sections else "No financial sections found"}
            
            Analysis Results:
            - Anomalies: {analysis_results.get('anomalies', [])}
            - Plagiarism cases: {len(analysis_results.get('plagiarism_report', {}).get('plagiarism_cases', []))}
            
            Please provide a detailed financial analysis based on the DRHP document content.
            Focus on revenue trends, profitability, growth patterns, and use of IPO proceeds.
            Be specific and provide actionable insights.
            """
            
            return self._generate_ai_response(context)
            
        except Exception as e:
            self.logger.error(f"Error handling financial query: {e}")
            return "I encountered an error analyzing the financial information. Please try again."
    
    def _handle_news_query(self, user_input: str, web_scraper) -> str:
        """
        Handle IPO and finance-specific news queries using web scraping
        """
        try:
            # Extract search terms from user input
            search_terms = self._extract_search_terms(user_input)
            
            if not search_terms:
                # Default to IPO/finance news if no specific terms found
                search_terms = ["IPO OR initial public offering OR stock market OR equity OR shares OR DRHP OR SEBI"]
            
            # Get detailed IPO/finance news articles using NewsAPI
            articles = web_scraper.get_detailed_news_articles(search_terms[0], limit=5)
            
            if not articles:
                # Fallback to Selenium-based IPO/finance news scraping
                self.logger.info("NewsAPI failed, trying Selenium-based IPO/finance news scraping")
                
                # Try multiple sources with Selenium for IPO/finance content
                sources = ['bbc_business', 'reuters_business']
                all_headlines = []
                
                for source in sources:
                    try:
                        headlines = web_scraper.fetch_latest_news(source=source, limit=3)
                        if headlines:
                            all_headlines.extend(headlines)
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch IPO/finance news from {source}: {e}")
                        continue
                
                if all_headlines:
                    response = f"📰 **Latest IPO & Finance News Headlines:**\n\n"
                    for i, headline in enumerate(all_headlines[:5], 1):
                        response += f"**{i}. {headline}**\n"
                    response += f"\n💡 **Source**: Selenium WebDriver + Multiple Sources\n"
                    response += f"📅 **Updated**: Just now\n"
                    response += f"🎯 **Focus**: IPO, Finance, Stock Market, DRHP, SEBI\n"
                    return response
                else:
                    return f"⚠️ Sorry, I couldn't fetch any IPO/finance news headlines at the moment. The news service might be temporarily unavailable."
            
            # Format detailed IPO/finance news response
            response = f"📰 **Latest IPO & Finance News Articles** (Search: '{search_terms[0]}'):\n\n"
            
            for i, article in enumerate(articles, 1):
                response += f"**{i}. {article.get('title', 'No title')}**\n"
                response += f"   📰 Source: {article.get('source', 'Unknown')}\n"
                response += f"   📅 Date: {article.get('publish_date', 'Unknown')[:10] if article.get('publish_date') else 'Unknown'}\n"
                # Make links clickable in markdown
                url = article.get('url', '')
                if url:
                    response += f"   🔗 Link: [{url}]({url})\n"
                else:
                    response += f"   🔗 Link: No link available\n"
                if article.get('description'):
                    response += f"   📝 Summary: {article.get('description', 'No summary available')[:150]}...\n"
                response += f"   🏷️ Category: {article.get('category', 'IPO/Finance')}\n"
                response += "\n"
            
            response += f"💡 **Source**: NewsAPI.org + Selenium WebDriver\n"
            response += f"📅 **Updated**: Just now\n"
            response += f"🔗 **Note**: Click on the blue links above to read the full articles\n"
            response += f"🎯 **Focus**: IPO, Finance, Stock Market, DRHP, SEBI, Equity, Shares"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling IPO/finance news query: {e}")
            return "I encountered an error fetching the latest IPO/finance news. Please try again later."

    def _handle_web_search_query(self, user_input: str, web_scraper, uploaded_file) -> str:
        """
        Handle web search queries using the web scraper
        """
        try:
            # Extract search terms from user input
            search_terms = self._extract_search_terms(user_input)
            
            if not search_terms:
                return "I need more specific search terms to help you find relevant information. Please specify what you're looking for."
            
            # Perform web search
            search_results = []
            for term in search_terms:
                try:
                    # Use the web scraper to search for news articles
                    articles = web_scraper.scrape_news_articles(term)
                    search_results.extend(articles[:3])  # Limit to 3 articles per term
                except Exception as e:
                    self.logger.warning(f"Web search failed for term '{term}': {e}")
            
            if not search_results:
                return f"I couldn't find recent news articles for '{', '.join(search_terms)}'. The web scraper might be temporarily unavailable."
            
            # Format search results
            response = f"🌐 **Search Results for '{', '.join(search_terms)}':**\n\n"
            
            for i, article in enumerate(search_results[:5], 1):  # Show top 5 results
                response += f"**{i}. {article.get('title', 'No title')}**\n"
                response += f"   📰 Source: {article.get('source', 'Unknown')}\n"
                response += f"   📅 Date: {article.get('date', 'Unknown')}\n"
                response += f"   🔗 Link: {article.get('url', 'No link')}\n"
                response += f"   📝 Summary: {article.get('summary', 'No summary available')[:200]}...\n\n"
            
            response += "💡 **Tip**: Click on the links to read the full articles for more detailed information."
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling web search query: {e}")
            return "I encountered an error performing the web search. Please try again later."
    
    def _handle_company_query(self, user_input: str, web_scraper, uploaded_file) -> str:
        """
        Handle company-specific queries using web scraping
        """
        try:
            # Extract company name from uploaded file or user input
            company_name = None
            if uploaded_file:
                company_name = uploaded_file.name.replace('.pdf', '').replace('_', ' ')
            
            if not company_name:
                company_name = self._extract_company_name(user_input)
            
            if not company_name:
                return "I need to know which company you're asking about. Please specify the company name or upload a DRHP document."
            
            # Search for company-specific information
            try:
                articles = web_scraper.scrape_news_articles(company_name)
                
                if not articles:
                    return f"I couldn't find recent news articles about {company_name}. The company might not be in the news recently, or the web scraper might be temporarily unavailable."
                
                # Format company information
                response = f"🏢 **Recent News about {company_name}:**\n\n"
                
                for i, article in enumerate(articles[:5], 1):  # Show top 5 articles
                    response += f"**{i}. {article.get('title', 'No title')}**\n"
                    response += f"   📰 Source: {article.get('source', 'Unknown')}\n"
                    response += f"   📅 Date: {article.get('date', 'Unknown')}\n"
                    response += f"   🔗 Link: {article.get('url', 'No link')}\n"
                    response += f"   📝 Summary: {article.get('summary', 'No summary available')[:200]}...\n\n"
                
                response += f"💡 **Analysis**: Based on recent news, {company_name} appears to be {'actively covered' if len(articles) > 3 else 'moderately covered'} in financial media."
                
                return response
                
            except Exception as e:
                self.logger.warning(f"Company search failed: {e}")
                return f"I encountered an error searching for information about {company_name}. Please try again later."
            
        except Exception as e:
            self.logger.error(f"Error handling company query: {e}")
            return "I encountered an error processing your company query. Please try again."
    
    def _extract_search_terms(self, user_input: str) -> List[str]:
        """
        Extract search terms from user input
        """
        # Common search patterns
        search_patterns = [
            r'find (?:news about|information about|updates on) (.+)',
            r'search for (.+)',
            r'look up (.+)',
            r'get (.+) news',
            r'find (.+) information'
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return [match.group(1).strip()]
        
        # If no pattern matches, try to extract key terms
        words = user_input.lower().split()
        key_terms = []
        
        for word in words:
            if len(word) > 3 and word not in ['find', 'search', 'look', 'get', 'about', 'information', 'news', 'updates']:
                key_terms.append(word)
        
        return key_terms[:3]  # Return top 3 terms
    
    def _get_anomaly_explanation_template(self) -> str:
        """Template for anomaly explanations"""
        return "Based on the analysis, I found {count} anomalies in your DRHP document..."
    
    def _get_plagiarism_explanation_template(self) -> str:
        """Template for plagiarism explanations"""
        return "The plagiarism analysis detected {count} potential cases of similar content..."
    
    def _get_section_analysis_template(self) -> str:
        """Template for section analysis"""
        return "The {section} section contains {word_count} words and covers..."
    
    def _get_general_help_template(self) -> str:
        """Template for general help"""
        return "I can help you analyze your DRHP document. What specific aspect would you like to explore?"
    
    def get_conversation_summary(self) -> str:
        """
        Get summary of conversation history
        """
        if not self.conversation_history:
            return "No conversation history available."
        
        return f"Conversation contains {len(self.conversation_history)} exchanges."
    
    def clear_conversation_history(self):
        """
        Clear conversation history
        """
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def _handle_news_summary_query(self, user_input: str) -> str:
        """
        Handle news summary requests like "give 50 lines summary of news on PhysicsWallah"
        Creates X lines of summary for EACH article
        """
        try:
            # Extract number of lines requested
            lines_match = re.search(r'(\d+)\s*lines?', user_input, re.IGNORECASE)
            num_lines = int(lines_match.group(1)) if lines_match else 30  # Default 30 lines
            
            # Extract company name from query
            patterns = [
                r'(?:summary|summarize).*?(?:news|articles).*?(?:on|about|for)\s+(.+?)(?:\s|$)',
                r'(?:news|articles).*?(?:on|about|for)\s+(.+?)(?:\s+summary|\s|$)',
                r'(.+?)\s+(?:news|articles)\s+summary'
            ]
            
            company_name = None
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    company_name = match.group(1).strip().strip('?.,')
                    # Clean up common words
                    company_name = re.sub(r'\b(the|a|an|news|summary|give|lines|\d+)\b', '', company_name, flags=re.IGNORECASE).strip()
                    if company_name:
                        break
            
            if not company_name:
                return "Please specify which company or topic you'd like a news summary for. Example: 'Give 50 lines summary of news on PhysicsWallah'"
            
            self.logger.info(f"📰 News summary requested for: {company_name} ({num_lines} lines per article)")
            
            # Fetch news
            if not self.news_scraper:
                return f"Enhanced News Scraper is not available. Cannot fetch news summary for {company_name}."
            
            try:
                # Try enhanced scraper first
                df_news = self.news_scraper.get_all_news(company_name)
                
                # If empty, try direct NewsAPI
                if df_news.empty:
                    self.logger.info("Enhanced scraper returned no results, trying direct NewsAPI...")
                    from newsapi import NewsApiClient
                    newsapi = NewsApiClient(api_key="1dcc360ce32f44c7b28fe66eb6529ebb")
                    news_response = newsapi.get_everything(
                        q=company_name,
                        language='en',
                        sort_by='publishedAt',
                        page_size=10
                    )
                    
                    if news_response['status'] == 'ok' and news_response['articles']:
                        articles_list = []
                        for article in news_response['articles']:
                            articles_list.append({
                                'title': article.get('title', ''),
                                'url': article.get('url', ''),
                                'source_and_time': f"{article.get('source', {}).get('name', 'Unknown')} - {article.get('publishedAt', '')}",
                                'summary': article.get('description', '') or article.get('content', ''),
                                'content': article.get('content', '')
                            })
                        
                        import pandas as pd
                        df_news = pd.DataFrame(articles_list)
                
                if df_news.empty:
                    return f"I couldn't find recent news articles about '{company_name}'. The company might not be in recent news, or try a different search term."
                
                # Create comprehensive line-based summary
                response = f"# 📰 {num_lines}-Line News Summaries: {company_name}\n\n"
                response += f"**Found {len(df_news)} recent articles - Generating {num_lines} lines summary for each:**\n\n"
                response += "="*80 + "\n\n"
                
                # Process each article with line-based summary
                for i, (_, article) in enumerate(df_news.iterrows(), 1):
                    response += f"## Article {i}: {article.get('title', 'No title')}\n\n"
                    response += f"**Source:** {article.get('source_and_time', 'Unknown')}\n"
                    
                    # Get full content for summary
                    full_content = article.get('content', article.get('summary', ''))
                    
                    if full_content and len(full_content) > 100:
                        # Generate X-line summary
                        summary_lines = self._generate_line_summary(full_content, num_lines)
                        response += f"\n**{num_lines}-Line Summary:**\n\n"
                        response += summary_lines + "\n"
                    else:
                        response += f"\n**Summary:** {full_content}\n"
                    
                    # Add URL at the end of each summary
                    response += f"\n📎 **Read Full Article:** [{article.get('url', 'No URL')}]({article.get('url', '#')})\n\n"
                    response += "="*80 + "\n\n"
                
                response += f"\n✅ **Complete!** Generated {num_lines}-line summaries for {len(df_news)} articles about {company_name}\n"
                
                return response
                
            except Exception as e:
                self.logger.error(f"Error fetching news for summary: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return f"I encountered an error fetching news about '{company_name}': {str(e)}"
                
        except Exception as e:
            self.logger.error(f"Error handling news summary query: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return "I encountered an error processing your news summary request. Please try again."
    
    def _generate_line_summary(self, content: str, num_lines: int) -> str:
        """Generate a summary with approximately num_lines of text"""
        if not content:
            return "No content available for summarization."
        
        # Split into sentences
        try:
            import nltk
            sentences = nltk.sent_tokenize(content)
        except:
            # Fallback to basic splitting
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate approximate sentences needed for num_lines
        # Assuming ~2-3 sentences per line in formatted output
        num_sentences = max(3, min(len(sentences), num_lines // 2))
        
        # Take first num_sentences for coherent summary
        summary_sentences = sentences[:num_sentences]
        summary = ' '.join(summary_sentences)
        
        # Format into lines (approximately 80 chars per line)
        words = summary.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 > 80:
                lines.append(current_line.strip())
                current_line = word
            else:
                current_line += " " + word if current_line else word
        
        if current_line:
            lines.append(current_line.strip())
        
        # Limit to requested number of lines
        lines = lines[:num_lines]
        
        return '\n'.join(lines)
    
    def _handle_company_info_query(self, user_input: str) -> str:
        """
        Handle "Tell me about X company" queries using Enhanced News Scraper
        """
        try:
            # Extract company name
            company_name = self._extract_company_name(user_input)
            
            if not company_name:
                # Try to extract from query directly
                words = user_input.split()
                for i, word in enumerate(words):
                    if word.lower() in ['about', 'on', 'regarding']:
                        if i + 1 < len(words):
                            company_name = ' '.join(words[i+1:]).strip('?.,')
                            break
            
            if not company_name:
                return "I need to know which company you're asking about. Please specify the company name."
            
            # Use Enhanced News Scraper if available
            if self.news_scraper:
                self.logger.info(f"📰 Fetching news about {company_name} using Enhanced Scraper...")
                
                try:
                    # Get news articles with FULL content
                    df_news = self.news_scraper.get_all_news(company_name)
                    
                    if df_news.empty:
                        return f"I couldn't find recent news articles about {company_name}. The company might not be in recent news."
                    
                    # Format response with full article summaries
                    response = f"# 📰 Latest Information About {company_name}\n\n"
                    response += f"**Found {len(df_news)} recent articles with full analysis:**\n\n"
                    
                    for i, (_, article) in enumerate(df_news.iterrows(), 1):
                        if i > 5:  # Limit to 5 articles
                            break
                        
                        response += f"## {i}. {article.get('title', 'No title')}\n"
                        response += f"**Source:** {article.get('source_and_time', 'Unknown')}\n"
                        response += f"**URL:** [{article.get('url', 'No URL')}]({article.get('url', '#')})\n\n"
                        
                        # Include full summary/content
                        summary = article.get('summary', article.get('content', 'No summary available'))
                        if len(summary) > 500:
                            response += f"**Summary:** {summary[:500]}... [Read more at link]\n\n"
                        else:
                            response += f"**Summary:** {summary}\n\n"
                    
                    response += f"\n💡 **Analysis:** Based on {len(df_news)} recent articles, {company_name} has been {'actively' if len(df_news) > 5 else 'moderately'} covered in financial news.\n"
                    
                    return response
                    
                except Exception as e:
                    self.logger.error(f"Error using Enhanced News Scraper: {e}")
                    return f"I encountered an error fetching news about {company_name}: {str(e)}"
            else:
                return f"I don't have access to the Enhanced News Scraper. Please use the web scraper through the UI to gather external data about {company_name}."
                
        except Exception as e:
            self.logger.error(f"Error handling company info query: {e}")
            return "I encountered an error processing your company query. Please try again."
    
    def _handle_compare_query(self, user_input: str, analysis_results: Dict, extracted_sections: Dict) -> str:
        """
        Handle comparison queries - ALWAYS helpful and proactive
        """
        try:
            input_lower = user_input.lower()
            
            # Determine what to compare
            is_risk_comparison = 'risk' in input_lower
            is_financial_comparison = any(kw in input_lower for kw in ['financial', 'revenue', 'profit', 'growth'])
            is_management_comparison = any(kw in input_lower for kw in ['management', 'director', 'promoter'])
            
            # Extract company name from uploaded file or sections
            company_name = None
            
            # Method 1: Extract from sections content
            all_sections = analysis_results.get('sections', {}) or extracted_sections
            if all_sections:
                for section_name, section_data in all_sections.items():
                    if isinstance(section_data, dict):
                        content = section_data.get('content', '')
                    else:
                        content = str(section_data)
                    
                    # Look for company name patterns - first occurrence only
                    match = re.search(r'([A-Z][A-Za-z\s&]+(?:Limited|Ltd|Inc|Corporation|Corp))', content)
                    if match:
                        raw_name = match.group(0).strip()
                        # Clean up: Remove common DRHP artifacts and trailing text
                        raw_name = re.sub(r'\s+and\s+.*', '', raw_name)  # Remove "and XYZ Ltd"
                        raw_name = re.sub(r'-DRHP.*', '', raw_name)  # Remove "-DRHP" suffix
                        raw_name = re.sub(r'\s+Ltd\s+.*', ' Ltd', raw_name)  # Clean trailing after Ltd
                        raw_name = re.sub(r'\s+Limited\s+.*', ' Limited', raw_name)  # Clean trailing after Limited
                        company_name = raw_name.strip()
                        self.logger.info(f"Extracted company name from sections: {company_name}")
                        break
            
            # Method 2: Extract from uploaded file name
            if not company_name:
                from src.ui.app import st
                if hasattr(st, 'session_state') and st.session_state.get('uploaded_files'):
                    filename = st.session_state.uploaded_files[-1].get('name', '')
                    # Clean filename more aggressively
                    company_name = filename.replace('.pdf', '').replace('_', ' ')
                    company_name = re.sub(r'-?DRHP.*', '', company_name, flags=re.IGNORECASE)
                    company_name = re.sub(r'\d{10,}', '', company_name)  # Remove long numbers (timestamps)
                    company_name = company_name.strip()
                    self.logger.info(f"Extracted company name from filename: {company_name}")
            
            # Method 3: Use default or extract from query
            if not company_name:
                words = user_input.split()
                for i, word in enumerate(words):
                    if word.lower() in ['about', 'for', 'of']:
                        if i + 1 < len(words):
                            company_name = ' '.join(words[i+1:]).strip('?.,')
                            break
                
                if not company_name:
                    company_name = "the company from DRHP"
            
            self.logger.info(f"📊 Comparison request for: {company_name}")
            
            # STEP 1: Always fetch news regardless of anomalies
            news_data = []
            news_status = "❌ No news available"
            
            if self.news_scraper:
                try:
                    self.logger.info(f"📰 Fetching live news for: {company_name}")
                    df_news = self.news_scraper.get_all_news(company_name)
                    
                    if not df_news.empty:
                        news_data = df_news.to_dict('records')
                        news_status = f"✅ Fetched {len(news_data)} articles with full content"
                        self.logger.info(news_status)
                    else:
                        news_status = "⚠️ No recent news found for this company"
                except Exception as e:
                    news_status = f"⚠️ News fetching failed: {str(e)[:100]}"
                    self.logger.error(f"Error fetching news: {e}")
            else:
                news_status = "⚠️ Enhanced News Scraper not initialized"
            
            # STEP 2: Build comprehensive response - ALWAYS helpful!
            response = f"# 📊 Comprehensive Analysis: {company_name}\n\n"
            response += f"**News Status:** {news_status}\n\n"
            
            # STEP 3: Summarize DRHP sections
            if is_risk_comparison:
                response += "## 📋 DRHP Risk Factors Summary\n\n"
                
                # Extract and summarize risk factors
                risk_section = None
                for section_name, section_data in (analysis_results.get('sections', {}) or extracted_sections).items():
                    if 'risk' in section_name.lower():
                        if isinstance(section_data, dict):
                            risk_section = section_data.get('content', '')
                        else:
                            risk_section = str(section_data)
                        break
                
                if risk_section:
                    # Extract risk points using simple text analysis
                    risk_points = self._extract_risk_points(risk_section)
                    response += f"**Key Risk Factors Disclosed in DRHP:**\n\n"
                    for i, risk in enumerate(risk_points[:10], 1):
                        response += f"{i}. {risk}\n"
                    
                    if len(risk_points) > 10:
                        response += f"\n*...and {len(risk_points) - 10} more risk factors*\n"
                else:
                    response += "⚠️ No risk factors section found in DRHP.\n"
                
                response += "\n---\n\n"
            
            elif is_financial_comparison:
                response += "## 💰 DRHP Financial Information Summary\n\n"
                
                # Extract financial section
                financial_section = None
                for section_name, section_data in (analysis_results.get('sections', {}) or extracted_sections).items():
                    if any(kw in section_name.lower() for kw in ['financial', 'statement', 'performance']):
                        if isinstance(section_data, dict):
                            financial_section = section_data.get('content', '')
                        else:
                            financial_section = str(section_data)
                        break
                
                if financial_section:
                    # Extract financial metrics
                    financial_metrics = self._extract_financial_metrics(financial_section)
                    response += f"**Financial Metrics from DRHP:**\n\n"
                    for metric, value in financial_metrics.items():
                        response += f"- **{metric}**: {value}\n"
                else:
                    response += "⚠️ No financial statements section found in DRHP.\n"
                
                response += "\n---\n\n"
            
            elif is_management_comparison:
                response += "## 👥 DRHP Management & Directors Summary\n\n"
                
                # Extract management section
                mgmt_section = None
                for section_name, section_data in (analysis_results.get('sections', {}) or extracted_sections).items():
                    if any(kw in section_name.lower() for kw in ['management', 'director', 'promoter']):
                        if isinstance(section_data, dict):
                            mgmt_section = section_data.get('content', '')
                        else:
                            mgmt_section = str(section_data)
                        break
                
                if mgmt_section:
                    # Extract key management info
                    response += f"**Management Team (from DRHP):**\n\n"
                    # Simple extraction - first 500 chars
                    response += f"{mgmt_section[:500]}...\n"
                else:
                    response += "⚠️ No management section found in DRHP.\n"
                
                response += "\n---\n\n"
            
            # STEP 4: Add news summary
            if news_data:
                response += "## 📰 Recent News Summary\n\n"
                response += f"**Found {len(news_data)} relevant articles:**\n\n"
                
                for i, article in enumerate(news_data[:5], 1):
                    response += f"### {i}. {article.get('title', 'No title')}\n"
                    response += f"- **Source:** {article.get('source_and_time', 'Unknown')}\n"
                    response += f"- **URL:** [{article.get('url', '#')}]({article.get('url', '#')})\n"
                    
                    summary = article.get('summary', article.get('content', 'No summary'))
                    if len(summary) > 300:
                        response += f"- **Summary:** {summary[:300]}...\n\n"
                    else:
                        response += f"- **Summary:** {summary}\n\n"
                
                if len(news_data) > 5:
                    response += f"*Showing 5 of {len(news_data)} articles*\n\n"
                
                response += "---\n\n"
            else:
                response += "## 📰 Recent News\n\n"
                response += f"⚠️ {news_status}\n\n"
                response += "---\n\n"
            
            # STEP 5: Add conclusions section
            response += "## 🎯 Analysis & Conclusions\n\n"
            
            if news_data:
                response += f"Based on {len(news_data)} recent articles about {company_name}:\n\n"
                
                # Extract key themes from news
                news_themes = self._extract_news_themes(news_data)
                response += f"**Key Themes in News:**\n"
                for theme in news_themes[:5]:
                    response += f"- {theme}\n"
                response += "\n"
            
            # Add recommendations
            response += "### 💡 Recommendations\n\n"
            
            if is_risk_comparison:
                response += "1. Review all risk factors disclosed in DRHP\n"
                response += "2. Cross-check with recent news for any undisclosed risks\n"
                response += "3. Evaluate if current events match risk disclosures\n"
                response += "4. Look for patterns of recurring issues in news\n"
            elif is_financial_comparison:
                response += "1. Verify financial claims against news reports\n"
                response += "2. Check for consistency across different sources\n"
                response += "3. Look for trends in financial performance\n"
                response += "4. Evaluate if reported numbers are realistic\n"
            else:
                response += "1. Review the DRHP sections carefully\n"
                response += "2. Monitor recent news for relevant developments\n"
                response += "3. Look for consistency between DRHP and news\n"
            
            response += "\n---\n\n"
            response += f"✅ **Analysis Complete!** Fetched live data and summarized DRHP sections for {company_name}\n"
            
            return response
                
        except Exception as e:
            self.logger.error(f"Error handling compare query: {e}")
            return "I encountered an error processing your comparison request. Please try again."
    
    def _handle_general_comparison(self, user_input: str, analysis_results: Dict, extracted_sections: Dict, news_data: List[Dict] = None) -> str:
        """Handle general comparison queries"""
        # Get all available DRHP data
        all_sections = analysis_results.get('sections', {}) or extracted_sections
        
        context = f"""
        User Question: {user_input}
        
        DRHP DOCUMENT SECTIONS:
        {json.dumps(list(all_sections.keys()), indent=2)}
        
        RECENT NEWS:
        """
        
        if news_data:
            for i, article in enumerate(news_data[:5], 1):
                context += f"\n{i}. {article.get('title', 'No title')}"
        else:
            context += "\nNo recent news available for comparison."
        
        context += """
        
        Please provide a comprehensive comparison addressing the user's question.
        Draw meaningful conclusions and provide actionable insights.
        """
        
        return self._generate_ai_response(context)
    
    def _handle_news_query_enhanced(self, user_input: str) -> str:
        """
        Handle news queries using Enhanced News Scraper - flexible and comprehensive
        """
        try:
            if not self.news_scraper:
                return "Enhanced News Scraper is not available. Please check the configuration."
            
            # Extract search terms
            input_lower = user_input.lower()
            
            # Check if user wants "all news" or "regardless of relevance"
            show_all = any(phrase in input_lower for phrase in ['all news', 'regardless', 'any news', 'everything'])
            
            # Extract company/topic name
            query = None
            
            # Try to extract from common patterns
            patterns = [
                r'news (?:related to|about|on|for)\s+(.+?)(?:\s+IPO)?$',
                r'(.+?)\s+(?:IPO|news|articles)',
                r'(?:about|on|for)\s+(.+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    query = match.group(1).strip().strip('?.,')
                    break
            
            if not query:
                # Use full input minus common words
                words = user_input.split()
                query_words = [w for w in words if w.lower() not in ['news', 'about', 'latest', 'recent', 'all', 'ipo', 'related', 'to', 'on', 'for', 'regardless', 'of', 'relevance']]
                query = ' '.join(query_words) if query_words else "IPO stock market"
            
            self.logger.info(f"📰 Searching news for: '{query}' (show_all={show_all})")
            
            # Fetch news with full content
            try:
                df_news = self.news_scraper.get_all_news(query)
            except Exception as e:
                # If enhanced scraper fails, try with simpler query
                self.logger.warning(f"Enhanced scraper failed, trying simpler approach: {e}")
                # Use NewsAPI directly with looser filtering
                try:
                    from newsapi import NewsApiClient
                    newsapi = NewsApiClient(api_key="1dcc360ce32f44c7b28fe66eb6529ebb")
                    news_response = newsapi.get_everything(
                        q=query,
                        language='en',
                        sort_by='publishedAt',
                        page_size=10
                    )
                    
                    if news_response['status'] == 'ok':
                        articles_list = []
                        for article in news_response['articles']:
                            articles_list.append({
                                'title': article.get('title', ''),
                                'url': article.get('url', ''),
                                'source_and_time': f"{article.get('source', {}).get('name', 'Unknown')} - {article.get('publishedAt', '')}",
                                'summary': article.get('description', '') or article.get('content', ''),
                                'content': article.get('content', '')
                            })
                        
                        import pandas as pd
                        df_news = pd.DataFrame(articles_list)
                    else:
                        df_news = pd.DataFrame()
                except Exception as e2:
                    self.logger.error(f"Direct NewsAPI also failed: {e2}")
                    return f"❌ I couldn't fetch news about '{query}'. Error: {str(e2)}"
            
            if df_news.empty:
                return f"I couldn't find recent news articles about '{query}'. This might mean:\n- Company not in recent news\n- Search term too specific\n- Try a broader search term like just the company name"
            
            # Format response
            response = f"# 📰 Latest News: {query}\n\n"
            response += f"**Found {len(df_news)} articles:**\n\n"
            
            # Show more articles if "all news" requested
            max_articles = 10 if show_all else 5
            
            for i, (_, article) in enumerate(df_news.iterrows(), 1):
                if i > max_articles:
                    break
                
                response += f"## {i}. {article.get('title', 'No title')}\n"
                response += f"- **Source:** {article.get('source_and_time', 'Unknown')}\n"
                response += f"- **URL:** [{article.get('url', 'No URL')}]({article.get('url', '#')})\n"
                
                summary = article.get('summary', article.get('content', 'No summary available'))
                if summary and len(summary) > 300:
                    response += f"- **Summary:** {summary[:300]}...\n\n"
                elif summary:
                    response += f"- **Summary:** {summary}\n\n"
                else:
                    response += f"- **Summary:** No summary available\n\n"
            
            if len(df_news) > max_articles:
                response += f"\n*Showing top {max_articles} of {len(df_news)} articles*\n"
                if not show_all:
                    response += f"*Add 'all news' or 'regardless of relevance' to see more articles*\n"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling news query: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"I encountered an error fetching news about '{query}': {str(e)}\n\nTry:\n- Simplifying your search term\n- Just the company name without 'IPO' or other keywords"
    
    def _extract_risk_points(self, risk_section: str) -> List[str]:
        """Extract risk points from risk factors section"""
        risks = []
        
        # Split by common delimiters
        lines = risk_section.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered points or bullet points
            if re.match(r'^\d+[\.\)]\s+', line) or line.startswith('•') or line.startswith('-'):
                # Clean and add
                clean_line = re.sub(r'^\d+[\.\)]\s+|^[•\-]\s+', '', line).strip()
                if len(clean_line) > 20 and len(clean_line) < 200:  # Reasonable risk description length
                    risks.append(clean_line)
        
        # If no formatted risks found, try to extract sentences mentioning risk keywords
        if not risks:
            risk_keywords = ['risk', 'may', 'could', 'potential', 'uncertain', 'challenge', 'threat']
            sentences = re.split(r'[.!?]+', risk_section)
            for sent in sentences:
                if any(keyword in sent.lower() for keyword in risk_keywords) and len(sent) > 30:
                    risks.append(sent.strip())
        
        return risks[:15]  # Return top 15 risks
    
    def _extract_financial_metrics(self, financial_section: str) -> Dict[str, str]:
        """Extract financial metrics from financial section"""
        metrics = {}
        
        # Common patterns for Indian financial statements
        patterns = {
            'Revenue': r'[Rr]evenue.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|lakhs?|million)',
            'Profit': r'[Pp]rofit.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|lakhs?|million)',
            'Assets': r'[Aa]ssets.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|lakhs?|million)',
            'Debt': r'[Dd]ebt.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|lakhs?|million)',
            'Growth': r'[Gg]rowth.*?(\d+(?:\.\d+)?)\s*%',
            'EBITDA': r'EBITDA.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|lakhs?|million)'
        }
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, financial_section)
            if matches:
                metrics[metric_name] = matches[0] + (' crore' if 'crore' in pattern else '%' if '%' in pattern else '')
        
        return metrics
    
    def _extract_news_themes(self, news_data: List[Dict]) -> List[str]:
        """Extract key themes from news articles"""
        themes = []
        
        # Collect all titles
        titles = [article.get('title', '') for article in news_data]
        
        # Look for common keywords
        common_keywords = ['IPO', 'funding', 'expansion', 'growth', 'profit', 'revenue', 
                          'acquisition', 'merger', 'partnership', 'launch', 'investment']
        
        keyword_counts = {}
        for keyword in common_keywords:
            count = sum(1 for title in titles if keyword.lower() in title.lower())
            if count > 0:
                keyword_counts[keyword] = count
        
        # Sort by frequency
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        
        for keyword, count in sorted_keywords[:5]:
            themes.append(f"{keyword} ({count} articles)")
        
        return themes if themes else ["General business news"]
    
