# Make src a package and provide convenient re-exports for legacy paths

# Re-export commonly used classes for backward compatibility
try:
    from src.parsing.pdf_parser import PDFParser  # noqa: F401
except Exception:
    pass

try:
    from src.scraping.web_scraper import WebScraper  # noqa: F401
except Exception:
    pass


