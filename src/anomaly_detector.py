import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Advanced anomaly detection system for DRHP documents
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Financial ratio patterns for anomaly detection
        self.financial_patterns = {
            'revenue_growth': r'revenue.*growth.*(\d+\.?\d*)\s*%',
            'profit_margin': r'profit.*margin.*(\d+\.?\d*)\s*%',
            'debt_equity': r'debt.*equity.*(\d+\.?\d*)',
            'current_ratio': r'current.*ratio.*(\d+\.?\d*)',
            'roe': r'return.*equity.*(\d+\.?\d*)\s*%',
            'roa': r'return.*assets.*(\d+\.?\d*)\s*%'
        }
        
        # Risk factor patterns
        self.risk_patterns = {
            'high_risk_keywords': [
                'litigation', 'legal proceedings', 'regulatory action',
                'fraud', 'misconduct', 'violation', 'penalty',
                'default', 'bankruptcy', 'insolvency'
            ],
            'financial_risk_keywords': [
                'debt burden', 'liquidity crisis', 'cash flow problems',
                'working capital deficit', 'high leverage', 'debt restructuring'
            ],
            'market_risk_keywords': [
                'market volatility', 'economic downturn', 'recession',
                'currency fluctuation', 'interest rate risk'
            ]
        }
        
        # Inconsistency patterns
        self.inconsistency_patterns = {
            'date_mismatch': r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            'number_inconsistency': r'(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'percentage_inconsistency': r'(\d+\.?\d*)\s*%'
        }
    
    def detect_anomalies(self, sections: Dict[str, str]) -> List[Dict]:
        """
        Main method to detect anomalies across all sections
        """
        anomalies = []
        
        # Financial anomalies
        financial_anomalies = self._detect_financial_anomalies(sections)
        anomalies.extend(financial_anomalies)
        
        # Risk factor anomalies
        risk_anomalies = self._detect_risk_anomalies(sections)
        anomalies.extend(risk_anomalies)
        
        # Structural anomalies
        structural_anomalies = self._detect_structural_anomalies(sections)
        anomalies.extend(structural_anomalies)
        
        # Content anomalies
        content_anomalies = self._detect_content_anomalies(sections)
        anomalies.extend(content_anomalies)
        
        # Inconsistency anomalies
        inconsistency_anomalies = self._detect_inconsistencies(sections)
        anomalies.extend(inconsistency_anomalies)
        
        self.logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies

    # -------------------------
    # Optional Enhancements API
    # -------------------------
    def detect_anomalies_with_stats(self,
                                    sections: Dict[str, Dict],
                                    zscore_threshold: float = 3.0,
                                    ratio_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict]:
        """
        Optional extension: run additional statistical checks without changing
        default behavior of detect_anomalies().

        - Z-score on section lengths to spot unusually short/long sections
        - Simple ratio bounds (e.g., digit ratio) per section content

        Args:
            sections: Section mapping as produced by the parser
            zscore_threshold: Z threshold for flagging anomalies
            ratio_bounds: Optional bounds mapping like:
                {
                    'digit_ratio': (0.0, 0.6),
                    'uppercase_ratio': (0.0, 0.5)
                }

        Returns:
            A list of additional anomaly dicts. Does not modify core logic.
        """
        if not sections:
            return []

        # Prepare features
        section_names: List[str] = list(sections.keys())
        lengths: List[int] = []
        digit_ratios: List[float] = []
        upper_ratios: List[float] = []

        for name in section_names:
            content = sections[name].get('content', '') if isinstance(sections[name], dict) else str(sections[name])
            text = str(content)
            L = len(text)
            lengths.append(L)
            if L > 0:
                digits = sum(c.isdigit() for c in text)
                uppers = sum(c.isupper() for c in text)
                digit_ratios.append(digits / L)
                upper_ratios.append(uppers / L)
            else:
                digit_ratios.append(0.0)
                upper_ratios.append(0.0)

        lengths_arr = np.array(lengths, dtype=float)
        if lengths_arr.size == 0 or np.nanstd(lengths_arr) == 0:
            return []

        # Z-score anomalies (length based)
        zscores = (lengths_arr - np.nanmean(lengths_arr)) / (np.nanstd(lengths_arr) + 1e-9)
        extra: List[Dict] = []
        for idx, z in enumerate(zscores):
            if abs(z) >= zscore_threshold:
                name = section_names[idx]
                section_data = sections[name] if isinstance(sections[name], dict) else {}
                extra.append({
                    'type': 'statistical_anomaly',
                    'subtype': 'length_zscore',
                    'section': name,
                    'severity': 'medium' if abs(z) < zscore_threshold * 1.5 else 'high',
                    'description': f"Section length z-score {z:.2f} exceeds threshold {zscore_threshold}",
                    'zscore': float(z),
                    'length': int(lengths[idx]),
                    'page_number': section_data.get('page_number', 'Unknown'),
                    'section_start_line': section_data.get('start_line', 'Unknown'),
                    'section_end_line': section_data.get('end_line', 'Unknown')
                })

        # Ratio bounds checks
        bounds = ratio_bounds or {'digit_ratio': (0.0, 0.6), 'uppercase_ratio': (0.0, 0.5)}
        for idx, name in enumerate(section_names):
            section_data = sections[name] if isinstance(sections[name], dict) else {}
            dr = digit_ratios[idx]
            ur = upper_ratios[idx]
            if 'digit_ratio' in bounds:
                lo, hi = bounds['digit_ratio']
                if not (lo <= dr <= hi):
                    extra.append({
                        'type': 'statistical_anomaly',
                        'subtype': 'digit_ratio_out_of_bounds',
                        'section': name,
                        'severity': 'low',
                        'description': f"Digit ratio {dr:.2f} outside [{lo:.2f}, {hi:.2f}]",
                        'digit_ratio': float(dr),
                        'page_number': section_data.get('page_number', 'Unknown')
                    })
            if 'uppercase_ratio' in bounds:
                lo, hi = bounds['uppercase_ratio']
                if not (lo <= ur <= hi):
                    extra.append({
                        'type': 'statistical_anomaly',
                        'subtype': 'uppercase_ratio_out_of_bounds',
                        'section': name,
                        'severity': 'low',
                        'description': f"Uppercase ratio {ur:.2f} outside [{lo:.2f}, {hi:.2f}]",
                        'uppercase_ratio': float(ur),
                        'page_number': section_data.get('page_number', 'Unknown')
                    })

        return extra

    def detect_anomalies_with_ml(self,
                                  sections: Dict[str, Dict],
                                  contamination: float = 0.1,
                                  random_state: int = 42) -> List[Dict]:
        """
        Optional ML-based outlier detection using IsolationForest.
        Extracts simple, robust features per section and flags outliers.
        Does not alter the default detect_anomalies() behavior.

        Args:
            sections: Section mapping
            contamination: Expected proportion of outliers
            random_state: RNG seed for reproducibility

        Returns:
            A list of anomaly dicts flagged by the ML model.
        """
        if not sections:
            return []

        names: List[str] = list(sections.keys())
        feats: List[List[float]] = []
        meta: List[Dict[str, Any]] = []
        for name in names:
            data = sections[name] if isinstance(sections[name], dict) else {'content': str(sections[name])}
            content = data.get('content', '')
            length = float(len(content))
            words = float(len(content.split())) if content else 0.0
            digits = float(sum(c.isdigit() for c in content))
            uppers = float(sum(c.isupper() for c in content))
            digit_ratio = digits / max(length, 1.0)
            upper_ratio = uppers / max(length, 1.0)
            feats.append([length, words, digit_ratio, upper_ratio])
            meta.append({'name': name, 'page_number': data.get('page_number', 'Unknown')})

        X = np.array(feats, dtype=float)
        if X.size == 0:
            return []

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = IsolationForest(contamination=min(max(contamination, 0.01), 0.5), random_state=random_state)
        preds = clf.fit_predict(Xs)  # -1 outlier, 1 inlier
        scores = clf.decision_function(Xs)

        results: List[Dict] = []
        for i, y in enumerate(preds):
            if y == -1:
                name = meta[i]['name']
                results.append({
                    'type': 'ml_outlier',
                    'subtype': 'isolation_forest',
                    'section': name,
                    'severity': 'medium',
                    'description': f"Section flagged as outlier by IsolationForest (score {scores[i]:.3f})",
                    'model_score': float(scores[i]),
                    'features': {
                        'length': float(X[i, 0]),
                        'words': float(X[i, 1]),
                        'digit_ratio': float(X[i, 2]),
                        'upper_ratio': float(X[i, 3])
                    },
                    'page_number': meta[i]['page_number']
                })

        return results
    
    def _detect_financial_anomalies(self, sections: Dict[str, Dict]) -> List[Dict]:
        """
        Detect financial data anomalies with specific location mapping
        """
        anomalies = []
        
        # Extract financial data from financial statements section
        financial_section_data = sections.get('financial_statements', {})
        if not financial_section_data:
            return anomalies
        
        financial_section = financial_section_data.get('content', '')
        if not financial_section:
            return anomalies
        
        # Extract financial ratios with line numbers
        financial_data = self._extract_financial_ratios_with_location(financial_section)
        
        # Check for unrealistic ratios
        for ratio_name, data in financial_data.items():
            if data['value'] is not None:
                anomaly = self._check_financial_ratio_anomaly(ratio_name, data['value'])
                if anomaly:
                    anomaly['line_number'] = data['line_number']
                    anomaly['context'] = data['context']
                    anomaly['page_number'] = financial_section_data.get('page_number', 'Unknown')
                    anomaly['section_start_line'] = financial_section_data.get('start_line', 'Unknown')
                    anomaly['section_end_line'] = financial_section_data.get('end_line', 'Unknown')
                    anomaly['exact_location'] = f"Page {anomaly['page_number']}, Line {data['line_number']}: {data['context'][:100]}..."
                    anomalies.append(anomaly)
        
        # Check for missing financial data
        missing_data = self._check_missing_financial_data_with_location(financial_section, financial_section_data)
        anomalies.extend(missing_data)
        
        return anomalies
    
    def _extract_financial_ratios(self, text: str) -> Dict[str, float]:
        """
        Extract financial ratios from text
        """
        ratios = {}
        
        for ratio_name, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the first match and convert to float
                    ratios[ratio_name] = float(matches[0])
                except ValueError:
                    ratios[ratio_name] = None
        
        return ratios
    
    def _extract_financial_ratios_with_location(self, text: str) -> Dict[str, Dict]:
        """
        Extract financial ratios with line numbers and context
        """
        ratios = {}
        lines = text.split('\n')
        
        for ratio_name, pattern in self.financial_patterns.items():
            for line_num, line in enumerate(lines, 1):
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    try:
                        value = float(matches[0])
                        ratios[ratio_name] = {
                            'value': value,
                            'line_number': line_num,
                            'context': line.strip(),
                            'section': 'financial_statements'
                        }
                        break  # Take first occurrence
                    except ValueError:
                        continue
        
        return ratios
    
    def _check_financial_ratio_anomaly(self, ratio_name: str, value: float) -> Dict:
        """
        Check if a financial ratio is anomalous
        """
        anomaly_thresholds = {
            'revenue_growth': {'min': -50, 'max': 500},  # -50% to 500%
            'profit_margin': {'min': -100, 'max': 100},  # -100% to 100%
            'debt_equity': {'min': 0, 'max': 10},        # 0 to 10
            'current_ratio': {'min': 0.1, 'max': 10},   # 0.1 to 10
            'roe': {'min': -100, 'max': 100},           # -100% to 100%
            'roa': {'min': -50, 'max': 50}              # -50% to 50%
        }
        
        if ratio_name in anomaly_thresholds:
            thresholds = anomaly_thresholds[ratio_name]
            if value < thresholds['min'] or value > thresholds['max']:
                return {
                    'type': 'financial_anomaly',
                    'subtype': 'unrealistic_ratio',
                    'ratio_name': ratio_name,
                    'value': value,
                    'threshold_min': thresholds['min'],
                    'threshold_max': thresholds['max'],
                    'severity': 'high' if abs(value) > thresholds['max'] * 2 else 'medium',
                    'description': f"Unrealistic {ratio_name}: {value}",
                    'section': 'financial_statements'
                }
        
        return None
    
    def _check_missing_financial_data(self, text: str) -> List[Dict]:
        """
        Check for missing critical financial data
        """
        anomalies = []
        
        required_financial_items = [
            'revenue', 'profit', 'assets', 'liabilities',
            'cash flow', 'working capital', 'debt'
        ]
        
        for item in required_financial_items:
            if not re.search(rf'\b{item}\b', text, re.IGNORECASE):
                anomalies.append({
                    'type': 'missing_data',
                    'subtype': 'financial_data',
                    'missing_item': item,
                    'severity': 'medium',
                    'description': f"Missing critical financial data: {item}",
                    'section': 'financial_statements'
                })
        
        return anomalies
    
    def _check_missing_financial_data_with_location(self, text: str, section_data: Dict) -> List[Dict]:
        """
        Check for missing critical financial data with location information
        """
        anomalies = []
        
        required_financial_items = [
            'revenue', 'profit', 'assets', 'liabilities',
            'cash flow', 'working capital', 'debt'
        ]
        
        for item in required_financial_items:
            if not re.search(rf'\b{item}\b', text, re.IGNORECASE):
                anomaly = {
                    'type': 'missing_data',
                    'subtype': 'financial_data',
                    'missing_item': item,
                    'severity': 'medium',
                    'description': f"Missing critical financial data: {item}",
                    'section': 'financial_statements',
                    'page_number': section_data.get('page_number', 'Unknown'),
                    'section_start_line': section_data.get('start_line', 'Unknown'),
                    'section_end_line': section_data.get('end_line', 'Unknown'),
                    'exact_location': f"Page {section_data.get('page_number', 'Unknown')}, Section: {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')}",
                    'context': f"Financial statements section (lines {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')})"
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_risk_anomalies(self, sections: Dict[str, Dict]) -> List[Dict]:
        """
        Detect risk-related anomalies
        """
        anomalies = []
        
        risk_section_data = sections.get('risk_factors', {})
        if not risk_section_data:
            return anomalies
        
        risk_section = risk_section_data.get('content', '')
        if not risk_section:
            return anomalies
        
        # Check for high-risk keywords
        high_risk_count = 0
        for keyword in self.risk_patterns['high_risk_keywords']:
            if re.search(rf'\b{keyword}\b', risk_section, re.IGNORECASE):
                high_risk_count += 1
        
        if high_risk_count > 5:  # Threshold for high risk
            anomalies.append({
                'type': 'risk_anomaly',
                'subtype': 'high_risk_keywords',
                'risk_count': high_risk_count,
                'severity': 'high',
                'description': f"High number of risk keywords detected: {high_risk_count}",
                'section': 'risk_factors',
                'page_number': risk_section_data.get('page_number', 'Unknown'),
                'section_start_line': risk_section_data.get('start_line', 'Unknown'),
                'section_end_line': risk_section_data.get('end_line', 'Unknown'),
                'exact_location': f"Page {risk_section_data.get('page_number', 'Unknown')}, Section: {risk_section_data.get('start_line', 'Unknown')}-{risk_section_data.get('end_line', 'Unknown')}",
                'context': f"Risk factors section (lines {risk_section_data.get('start_line', 'Unknown')}-{risk_section_data.get('end_line', 'Unknown')})"
            })
        
        # Check for missing risk categories
        missing_risk_categories = self._check_missing_risk_categories_with_location(risk_section, risk_section_data)
        anomalies.extend(missing_risk_categories)
        
        return anomalies
    
    def _check_missing_risk_categories(self, text: str) -> List[Dict]:
        """
        Check for missing standard risk categories
        """
        anomalies = []
        
        standard_risk_categories = [
            'market risk', 'credit risk', 'operational risk',
            'regulatory risk', 'technology risk', 'competition risk'
        ]
        
        for category in standard_risk_categories:
            if not re.search(rf'\b{category}\b', text, re.IGNORECASE):
                anomalies.append({
                    'type': 'missing_data',
                    'subtype': 'risk_category',
                    'missing_category': category,
                    'severity': 'low',
                    'description': f"Missing standard risk category: {category}",
                    'section': 'risk_factors'
                })
        
        return anomalies
    
    def _check_missing_risk_categories_with_location(self, text: str, section_data: Dict) -> List[Dict]:
        """
        Check for missing standard risk categories with location information
        """
        anomalies = []
        
        standard_risk_categories = [
            'market risk', 'credit risk', 'operational risk',
            'regulatory risk', 'technology risk', 'competition risk'
        ]
        
        for category in standard_risk_categories:
            if not re.search(rf'\b{category}\b', text, re.IGNORECASE):
                anomaly = {
                    'type': 'missing_data',
                    'subtype': 'risk_category',
                    'missing_category': category,
                    'severity': 'low',
                    'description': f"Missing standard risk category: {category}",
                    'section': 'risk_factors',
                    'page_number': section_data.get('page_number', 'Unknown'),
                    'section_start_line': section_data.get('start_line', 'Unknown'),
                    'section_end_line': section_data.get('end_line', 'Unknown'),
                    'exact_location': f"Page {section_data.get('page_number', 'Unknown')}, Section: {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')}",
                    'context': f"Risk factors section (lines {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')})"
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_structural_anomalies(self, sections: Dict[str, Dict]) -> List[Dict]:
        """
        Detect structural anomalies in document
        """
        anomalies = []
        
        # Check for missing critical sections
        critical_sections = [
            'business_overview', 'financial_statements', 'risk_factors',
            'management_discussion', 'use_of_proceeds'
        ]
        
        for section in critical_sections:
            if section not in sections or not sections[section].get('content', '').strip():
                anomalies.append({
                    'type': 'structural_anomaly',
                    'subtype': 'missing_section',
                    'missing_section': section,
                    'severity': 'high',
                    'description': f"Missing critical section: {section}",
                    'section': 'document_structure',
                    'exact_location': f"Document structure - Section '{section}' not found",
                    'context': f"Required section '{section}' is missing from the document"
                })
        
        # Check for unusually short sections
        for section_name, section_data in sections.items():
            content = section_data.get('content', '')
            if len(content.strip()) < 100:  # Less than 100 characters
                anomalies.append({
                    'type': 'structural_anomaly',
                    'subtype': 'insufficient_content',
                    'section_name': section_name,
                    'content_length': len(content.strip()),
                    'severity': 'medium',
                    'description': f"Section '{section_name}' has insufficient content: {len(content.strip())} characters",
                    'section': section_name,
                    'page_number': section_data.get('page_number', 'Unknown'),
                    'section_start_line': section_data.get('start_line', 'Unknown'),
                    'section_end_line': section_data.get('end_line', 'Unknown'),
                    'exact_location': f"Page {section_data.get('page_number', 'Unknown')}, Section: {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')}",
                    'context': f"Section '{section_name}' (lines {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')}) has only {len(content.strip())} characters"
                })
        
        return anomalies
    
    def _detect_content_anomalies(self, sections: Dict[str, Dict]) -> List[Dict]:
        """
        Detect content-based anomalies
        """
        anomalies = []
        
        for section_name, section_data in sections.items():
            content = section_data.get('content', '')
            
            # Check for excessive repetition
            repetition_anomaly = self._check_repetition(content)
            if repetition_anomaly:
                repetition_anomaly['section'] = section_name
                repetition_anomaly['page_number'] = section_data.get('page_number', 'Unknown')
                repetition_anomaly['section_start_line'] = section_data.get('start_line', 'Unknown')
                repetition_anomaly['section_end_line'] = section_data.get('end_line', 'Unknown')
                repetition_anomaly['exact_location'] = f"Page {section_data.get('page_number', 'Unknown')}, Section: {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')}"
                repetition_anomaly['context'] = f"Section '{section_name}' (lines {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')})"
                anomalies.append(repetition_anomaly)
            
            # Check for formatting issues
            formatting_anomaly = self._check_formatting_issues(content)
            if formatting_anomaly:
                formatting_anomaly['section'] = section_name
                formatting_anomaly['page_number'] = section_data.get('page_number', 'Unknown')
                formatting_anomaly['section_start_line'] = section_data.get('start_line', 'Unknown')
                formatting_anomaly['section_end_line'] = section_data.get('end_line', 'Unknown')
                formatting_anomaly['exact_location'] = f"Page {section_data.get('page_number', 'Unknown')}, Section: {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')}"
                formatting_anomaly['context'] = f"Section '{section_name}' (lines {section_data.get('start_line', 'Unknown')}-{section_data.get('end_line', 'Unknown')})"
                anomalies.append(formatting_anomaly)
        
        return anomalies
    
    def _check_repetition(self, text: str) -> Dict:
        """
        Check for excessive repetition in text
        """
        words = text.lower().split()
        word_counts = {}
        
        for word in words:
            if len(word) > 3:  # Only count words longer than 3 characters
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find words that appear more than 5% of the time
        total_words = len(words)
        threshold = max(5, total_words * 0.05)
        
        for word, count in word_counts.items():
            if count > threshold:
                return {
                    'type': 'content_anomaly',
                    'subtype': 'excessive_repetition',
                    'repeated_word': word,
                    'count': count,
                    'percentage': (count / total_words) * 100,
                    'severity': 'medium',
                    'description': f"Excessive repetition of word '{word}': {count} times ({(count/total_words)*100:.1f}%)"
                }
        
        return None
    
    def _check_formatting_issues(self, text: str) -> Dict:
        """
        Check for formatting issues
        """
        # Check for excessive whitespace
        if re.search(r'\s{5,}', text):
            return {
                'type': 'content_anomaly',
                'subtype': 'formatting_issue',
                'issue': 'excessive_whitespace',
                'severity': 'low',
                'description': 'Excessive whitespace detected'
            }
        
        # Check for missing punctuation
        sentences = re.split(r'[.!?]', text)
        long_sentences = [s for s in sentences if len(s.strip()) > 200]
        
        if len(long_sentences) > 3:
            return {
                'type': 'content_anomaly',
                'subtype': 'formatting_issue',
                'issue': 'long_sentences',
                'count': len(long_sentences),
                'severity': 'low',
                'description': f"Multiple long sentences detected: {len(long_sentences)}"
            }
        
        return None
    
    def _detect_inconsistencies(self, sections: Dict[str, Dict]) -> List[Dict]:
        """
        Detect inconsistencies across sections
        """
        anomalies = []
        
        # Extract dates and check for consistency
        date_inconsistencies = self._check_date_consistency_with_location(sections)
        anomalies.extend(date_inconsistencies)
        
        # Extract numbers and check for consistency
        number_inconsistencies = self._check_number_consistency_with_location(sections)
        anomalies.extend(number_inconsistencies)
        
        return anomalies
    
    def _check_date_consistency_with_location(self, sections: Dict[str, Dict]) -> List[Dict]:
        """
        Check for date inconsistencies across sections with location data
        """
        anomalies = []
        all_dates = []
        
        for section_name, section_data in sections.items():
            content = section_data.get('content', '')
            dates = re.findall(self.inconsistency_patterns['date_mismatch'], content)
            for date in dates:
                all_dates.append({
                    'date': date,
                    'section': section_name,
                    'page_number': section_data.get('page_number', 'Unknown'),
                    'section_start_line': section_data.get('start_line', 'Unknown'),
                    'section_end_line': section_data.get('end_line', 'Unknown')
                })
        
        # Check for date conflicts
        if len(set([d['date'] for d in all_dates])) < len(all_dates):
            anomalies.append({
                'type': 'inconsistency',
                'subtype': 'date_conflict',
                'severity': 'medium',
                'description': 'Date inconsistencies detected across sections',
                'sections': list(set([d['section'] for d in all_dates])),
                'exact_location': f"Multiple sections: {', '.join(set([d['section'] for d in all_dates]))}",
                'context': f"Date conflicts found in {len(set([d['section'] for d in all_dates]))} sections"
            })
        
        return anomalies
    
    def _check_number_consistency_with_location(self, sections: Dict[str, Dict]) -> List[Dict]:
        """
        Check for number inconsistencies across sections with location data
        """
        anomalies = []
        
        # Extract financial numbers with location
        financial_numbers = {}
        for section_name, section_data in sections.items():
            content = section_data.get('content', '')
            numbers = re.findall(self.inconsistency_patterns['number_inconsistency'], content)
            financial_numbers[section_name] = {
                'numbers': numbers,
                'page_number': section_data.get('page_number', 'Unknown'),
                'section_start_line': section_data.get('start_line', 'Unknown'),
                'section_end_line': section_data.get('end_line', 'Unknown')
            }
        
        # Check for significant discrepancies (simplified)
        # This would need more sophisticated logic for real implementation
        
        return anomalies
    
    def _check_date_consistency(self, sections: Dict[str, str]) -> List[Dict]:
        """
        Check for date inconsistencies across sections
        """
        anomalies = []
        all_dates = []
        
        for section_name, content in sections.items():
            dates = re.findall(self.inconsistency_patterns['date_mismatch'], content)
            for date in dates:
                all_dates.append((date, section_name))
        
        # Check for date conflicts (simplified check)
        if len(set(all_dates)) < len(all_dates):
            anomalies.append({
                'type': 'inconsistency',
                'subtype': 'date_conflict',
                'severity': 'medium',
                'description': 'Date inconsistencies detected across sections',
                'sections': list(set([section for _, section in all_dates]))
            })
        
        return anomalies
    
    def _check_number_consistency(self, sections: Dict[str, str]) -> List[Dict]:
        """
        Check for number inconsistencies across sections
        """
        anomalies = []
        
        # Extract financial numbers
        financial_numbers = {}
        for section_name, content in sections.items():
            numbers = re.findall(self.inconsistency_patterns['number_inconsistency'], content)
            financial_numbers[section_name] = numbers
        
        # Check for significant discrepancies (simplified)
        # This would need more sophisticated logic for real implementation
        
        return anomalies
    
    def generate_anomaly_report(self, anomalies: List[Dict]) -> Dict:
        """
        Generate comprehensive anomaly report
        """
        if not anomalies:
            return {
                'total_anomalies': 0,
                'severity_breakdown': {},
                'section_breakdown': {},
                'recommendations': []
            }
        
        # Count by severity
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by section
        section_counts = {}
        for anomaly in anomalies:
            section = anomaly.get('section', 'unknown')
            section_counts[section] = section_counts.get(section, 0) + 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies)
        
        return {
            'total_anomalies': len(anomalies),
            'severity_breakdown': severity_counts,
            'section_breakdown': section_counts,
            'anomalies': anomalies,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, anomalies: List[Dict]) -> List[str]:
        """
        Generate recommendations based on detected anomalies
        """
        recommendations = []
        
        high_severity_count = sum(1 for a in anomalies if a.get('severity') == 'high')
        if high_severity_count > 0:
            recommendations.append(f"Address {high_severity_count} high-severity anomalies immediately")
        
        missing_sections = [a for a in anomalies if a.get('subtype') == 'missing_section']
        if missing_sections:
            recommendations.append("Add missing critical sections to improve document completeness")
        
        financial_anomalies = [a for a in anomalies if a.get('type') == 'financial_anomaly']
        if financial_anomalies:
            recommendations.append("Review and verify all financial data for accuracy")
        
        risk_anomalies = [a for a in anomalies if a.get('type') == 'risk_anomaly']
        if risk_anomalies:
            recommendations.append("Conduct thorough risk assessment and update risk factors")
        
        return recommendations
    
    def detect_verification_anomalies(self, sections: Dict[str, str], external_data: Dict = None) -> List[Dict]:
        """
        Detect anomalies by cross-checking DRHP claims against external data
        """
        verification_anomalies = []
        
        if not external_data:
            return verification_anomalies
        
        # Extract financial claims from DRHP
        financial_claims = self._extract_financial_claims(sections)
        
        # Cross-check with external news data
        if external_data.get('news_articles'):
            news_anomalies = self._cross_check_with_news(financial_claims, external_data['news_articles'])
            verification_anomalies.extend(news_anomalies)
        
        # Cross-check with company website data
        if external_data.get('company_website'):
            website_anomalies = self._cross_check_with_website(financial_claims, external_data['company_website'])
            verification_anomalies.extend(website_anomalies)
        
        return verification_anomalies
    
    def _extract_financial_claims(self, sections: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Extract specific financial claims from DRHP sections
        """
        claims = {
            'revenue_claims': [],
            'profit_claims': [],
            'growth_claims': [],
            'debt_claims': [],
            'asset_claims': []
        }
        
        # Patterns for extracting financial claims
        claim_patterns = {
            'revenue_claims': [
                r'revenue.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'total.*?revenue.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'sales.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)'
            ],
            'profit_claims': [
                r'profit.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'net.*?profit.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'earnings.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)'
            ],
            'growth_claims': [
                r'growth.*?(\d+(?:\.\d+)?)\s*%',
                r'increase.*?(\d+(?:\.\d+)?)\s*%',
                r'rise.*?(\d+(?:\.\d+)?)\s*%'
            ],
            'debt_claims': [
                r'debt.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'borrowing.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'liability.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)'
            ],
            'asset_claims': [
                r'assets.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'total.*?assets.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)'
            ]
        }
        
        # Extract claims from all sections
        for section_name, section_data in sections.items():
            if isinstance(section_data, dict):
                content = section_data.get('content', '')
            else:
                content = str(section_data)
            
            for claim_type, patterns in claim_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        try:
                            # Convert to numeric value
                            if '%' in pattern:
                                value = float(match)
                            else:
                                # Remove commas and convert
                                value = float(match.replace(',', ''))
                            
                            claims[claim_type].append({
                                'value': value,
                                'section': section_name,
                                'context': self._extract_context(content, match),
                                'claim_type': claim_type
                            })
                        except ValueError:
                            continue
        
        return claims
    
    def _extract_context(self, text: str, match: str, context_length: int = 100) -> str:
        """
        Extract context around a match
        """
        match_pos = text.find(match)
        if match_pos == -1:
            return ""
        
        start = max(0, match_pos - context_length // 2)
        end = min(len(text), match_pos + len(match) + context_length // 2)
        
        return text[start:end].strip()
    
    def _cross_check_with_news(self, financial_claims: Dict, news_articles: List[Dict]) -> List[Dict]:
        """
        Cross-check financial claims with news articles
        """
        anomalies = []
        
        # Extract financial data from news articles
        news_financial_data = self._extract_financial_data_from_news(news_articles)
        
        # Compare DRHP claims with news data
        for claim_type, claims in financial_claims.items():
            if not claims:
                continue
            
            for claim in claims:
                # Find matching news data
                news_matches = self._find_matching_news_data(claim, news_financial_data)
                
                for news_match in news_matches:
                    # Check for discrepancies
                    discrepancy = self._calculate_discrepancy(claim['value'], news_match['value'])
                    
                    if discrepancy > 0.2:  # 20% discrepancy threshold
                        anomalies.append({
                            'type': 'verification_anomaly',
                            'subtype': 'news_discrepancy',
                            'severity': 'high' if discrepancy > 0.5 else 'medium',
                            'description': f"DRHP claims {claim['value']} but news reports {news_match['value']} ({discrepancy:.1%} discrepancy)",
                            'drhp_claim': claim,
                            'news_data': news_match,
                            'discrepancy_percentage': discrepancy,
                            'section': claim['section'],
                            'recommendation': 'Verify financial data accuracy with external sources'
                        })
        
        return anomalies
    
    def _extract_financial_data_from_news(self, news_articles: List[Dict]) -> List[Dict]:
        """
        Extract financial data from news articles
        """
        financial_data = []
        
        # Patterns for extracting financial data from news
        news_patterns = {
            'revenue': [
                r'revenue.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'sales.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)'
            ],
            'profit': [
                r'profit.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'earnings.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)'
            ],
            'loss': [
                r'loss.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)',
                r'deficit.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)'
            ]
        }
        
        for article in news_articles:
            content = article.get('content', '')
            title = article.get('title', '')
            full_text = f"{title} {content}"
            
            for data_type, patterns in news_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, full_text, re.IGNORECASE)
                    for match in matches:
                        try:
                            value = float(match.replace(',', ''))
                            financial_data.append({
                                'type': data_type,
                                'value': value,
                                'source': article.get('source', 'Unknown'),
                                'title': title,
                                'url': article.get('url', ''),
                                'published_date': article.get('publish_date', ''),
                                'context': self._extract_context(full_text, match)
                            })
                        except ValueError:
                            continue
        
        return financial_data
    
    def _find_matching_news_data(self, claim: Dict, news_data: List[Dict]) -> List[Dict]:
        """
        Find matching news data for a claim
        """
        matches = []
        
        # Map claim types to news data types
        type_mapping = {
            'revenue_claims': ['revenue'],
            'profit_claims': ['profit', 'loss'],
            'debt_claims': ['debt'],
            'asset_claims': ['assets']
        }
        
        claim_type = claim['claim_type']
        if claim_type in type_mapping:
            target_types = type_mapping[claim_type]
            
            for news_item in news_data:
                if news_item['type'] in target_types:
                    # Check if values are in similar range (within 50% difference)
                    if abs(claim['value'] - news_item['value']) / max(claim['value'], news_item['value']) < 0.5:
                        matches.append(news_item)
        
        return matches
    
    def _calculate_discrepancy(self, claim_value: float, news_value: float) -> float:
        """
        Calculate percentage discrepancy between claim and news value
        """
        if claim_value == 0 and news_value == 0:
            return 0.0
        
        if claim_value == 0 or news_value == 0:
            return 1.0  # 100% discrepancy
        
        return abs(claim_value - news_value) / max(claim_value, news_value)
    
    def _cross_check_with_website(self, financial_claims: Dict, website_data: Dict) -> List[Dict]:
        """
        Cross-check financial claims with company website data
        """
        anomalies = []
        
        # Extract financial data from website
        website_financial_data = self._extract_financial_data_from_website(website_data)
        
        # Compare with DRHP claims
        for claim_type, claims in financial_claims.items():
            if not claims:
                continue
            
            for claim in claims:
                # Find matching website data
                website_matches = self._find_matching_website_data(claim, website_financial_data)
                
                for website_match in website_matches:
                    discrepancy = self._calculate_discrepancy(claim['value'], website_match['value'])
                    
                    if discrepancy > 0.15:  # 15% discrepancy threshold for website
                        anomalies.append({
                            'type': 'verification_anomaly',
                            'subtype': 'website_discrepancy',
                            'severity': 'medium',
                            'description': f"DRHP claims {claim['value']} but website shows {website_match['value']} ({discrepancy:.1%} discrepancy)",
                            'drhp_claim': claim,
                            'website_data': website_match,
                            'discrepancy_percentage': discrepancy,
                            'section': claim['section'],
                            'recommendation': 'Verify consistency between DRHP and company website'
                        })
        
        return anomalies
    
    def _extract_financial_data_from_website(self, website_data: Dict) -> List[Dict]:
        """
        Extract financial data from company website
        """
        financial_data = []
        
        # Extract from about text
        about_text = website_data.get('about_text', '')
        if about_text:
            # Use same patterns as news extraction
            website_patterns = {
                'revenue': [r'revenue.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)'],
                'profit': [r'profit.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|cr|million|billion|lakh)']
            }
            
            for data_type, patterns in website_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, about_text, re.IGNORECASE)
                    for match in matches:
                        try:
                            value = float(match.replace(',', ''))
                            financial_data.append({
                                'type': data_type,
                                'value': value,
                                'source': 'company_website',
                                'context': self._extract_context(about_text, match)
                            })
                        except ValueError:
                            continue
        
        return financial_data
    
    def _find_matching_website_data(self, claim: Dict, website_data: List[Dict]) -> List[Dict]:
        """
        Find matching website data for a claim
        """
        matches = []
        
        type_mapping = {
            'revenue_claims': ['revenue'],
            'profit_claims': ['profit']
        }
        
        claim_type = claim['claim_type']
        if claim_type in type_mapping:
            target_types = type_mapping[claim_type]
            
            for website_item in website_data:
                if website_item['type'] in target_types:
                    if abs(claim['value'] - website_item['value']) / max(claim['value'], website_item['value']) < 0.3:
                        matches.append(website_item)
        
        return matches
