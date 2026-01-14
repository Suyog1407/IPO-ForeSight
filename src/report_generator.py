import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Optional
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import base64
from io import BytesIO

class ReportGenerator:
    """
    Comprehensive report generation system for DRHP analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """
        Setup custom styles for reports
        """
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Body style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
    
    def generate_comprehensive_report(self, analysis_results: Dict, output_path: str = None) -> str:
        """
        Generate comprehensive PDF report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/drhp_analysis_report_{timestamp}.pdf"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Add title page
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        # Add executive summary
        story.extend(self._create_executive_summary(analysis_results))
        story.append(PageBreak())
        
        # Add document overview
        story.extend(self._create_document_overview(analysis_results))
        story.append(PageBreak())
        
        # Add anomaly analysis
        if analysis_results.get('anomalies'):
            story.extend(self._create_anomaly_analysis(analysis_results['anomalies']))
            story.append(PageBreak())
        
        # Add plagiarism analysis
        if analysis_results.get('plagiarism'):
            story.extend(self._create_plagiarism_analysis(analysis_results['plagiarism']))
            story.append(PageBreak())
        
        # Add recommendations
        story.extend(self._create_recommendations(analysis_results))
        story.append(PageBreak())
        
        # Add appendices
        story.extend(self._create_appendices(analysis_results))
        
        # Build PDF
        doc.build(story)
        
        self.logger.info(f"Comprehensive report generated: {output_path}")
        return output_path
    
    def _create_title_page(self) -> List:
        """
        Create title page content
        """
        content = []
        
        # Title
        title = Paragraph("DRHP Analysis Report", self.title_style)
        content.append(title)
        content.append(Spacer(1, 20))
        
        # Subtitle
        subtitle = Paragraph("Draft Red Herring Prospectus Analysis System", self.styles['Heading2'])
        content.append(subtitle)
        content.append(Spacer(1, 30))
        
        # Report details
        details = [
            ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Analysis Type:", "Comprehensive DRHP Analysis"],
            ["System Version:", "1.0.0"]
        ]
        
        details_table = Table(details, colWidths=[2*inch, 4*inch])
        details_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        content.append(details_table)
        content.append(Spacer(1, 30))
        
        # Disclaimer
        disclaimer = Paragraph(
            "This report is generated automatically by the DRHP Analysis System. "
            "The analysis is based on document processing algorithms and should be "
            "reviewed by qualified professionals before making any decisions.",
            self.styles['Normal']
        )
        content.append(disclaimer)
        
        return content
    
    def _create_executive_summary(self, analysis_results: Dict) -> List:
        """
        Create executive summary section
        """
        content = []
        
        # Section title
        title = Paragraph("Executive Summary", self.heading_style)
        content.append(title)
        
        # Summary statistics
        anomalies = analysis_results.get('anomalies', [])
        plagiarism_cases = analysis_results.get('plagiarism', [])
        sections = analysis_results.get('sections', {})
        
        summary_data = [
            ["Metric", "Count", "Status"],
            ["Document Sections", len(sections), "✓ Processed"],
            ["Anomalies Detected", len(anomalies), "⚠️" if anomalies else "✓ Clean"],
            ["Plagiarism Cases", len(plagiarism_cases), "⚠️" if plagiarism_cases else "✓ Original"],
            ["External Sources", len(analysis_results.get('external_data', [])), "✓ Analyzed"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(summary_table)
        content.append(Spacer(1, 20))
        
        # Key findings
        key_findings = self._generate_key_findings(analysis_results)
        findings_para = Paragraph(key_findings, self.body_style)
        content.append(findings_para)
        
        return content
    
    def _create_document_overview(self, analysis_results: Dict) -> List:
        """
        Create document overview section
        """
        content = []
        
        # Section title
        title = Paragraph("Document Overview", self.heading_style)
        content.append(title)
        
        # Document sections
        sections = analysis_results.get('sections', {})
        if sections:
            section_data = [["Section", "Word Count", "Status"]]
            for section_name, section_content in sections.items():
                word_count = len(section_content.split())
                status = "✓ Complete" if word_count > 100 else "⚠️ Insufficient"
                section_data.append([
                    section_name.replace('_', ' ').title(),
                    str(word_count),
                    status
                ])
            
            section_table = Table(section_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            section_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(section_table)
            content.append(Spacer(1, 20))
        
        # Document statistics
        total_words = sum(len(content.split()) for content in sections.values())
        stats_text = f"""
        <b>Document Statistics:</b><br/>
        • Total Sections: {len(sections)}<br/>
        • Total Word Count: {total_words:,}<br/>
        • Average Section Length: {total_words // max(len(sections), 1):,} words<br/>
        • Analysis Date: {datetime.now().strftime("%Y-%m-%d")}
        """
        
        stats_para = Paragraph(stats_text, self.body_style)
        content.append(stats_para)
        
        return content
    
    def _create_anomaly_analysis(self, anomalies: List[Dict]) -> List:
        """
        Create anomaly analysis section
        """
        content = []
        
        # Section title
        title = Paragraph("Anomaly Analysis", self.heading_style)
        content.append(title)
        
        # Anomaly summary
        severity_counts = {}
        type_counts = {}
        
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'unknown')
            anomaly_type = anomaly.get('type', 'unknown')
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
        
        # Summary table
        summary_data = [["Severity", "Count", "Percentage"]]
        total_anomalies = len(anomalies)
        
        for severity, count in severity_counts.items():
            percentage = (count / total_anomalies) * 100
            summary_data.append([severity.title(), str(count), f"{percentage:.1f}%"])
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(summary_table)
        content.append(Spacer(1, 20))
        
        # Detailed anomalies (top 10)
        if anomalies:
            detailed_data = [["Type", "Severity", "Description"]]
            for anomaly in anomalies[:10]:  # Top 10
                detailed_data.append([
                    anomaly.get('type', 'Unknown'),
                    anomaly.get('severity', 'Unknown'),
                    anomaly.get('description', 'No description')[:50] + "..."
                ])
            
            detailed_table = Table(detailed_data, colWidths=[1.5*inch, 1*inch, 3*inch])
            detailed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(detailed_table)
        
        return content
    
    def _create_plagiarism_analysis(self, plagiarism_cases: List[Dict]) -> List:
        """
        Create plagiarism analysis section
        """
        content = []
        
        # Section title
        title = Paragraph("Plagiarism Analysis", self.heading_style)
        content.append(title)
        
        # Plagiarism summary
        severity_counts = {}
        type_counts = {}
        
        for case in plagiarism_cases:
            severity = case.get('severity', 'unknown')
            case_type = case.get('type', 'unknown')
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[case_type] = type_counts.get(case_type, 0) + 1
        
        # Summary table
        summary_data = [["Similarity Level", "Count", "Percentage"]]
        total_cases = len(plagiarism_cases)
        
        for severity, count in severity_counts.items():
            percentage = (count / total_cases) * 100
            summary_data.append([severity.title(), str(count), f"{percentage:.1f}%"])
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(summary_table)
        content.append(Spacer(1, 20))
        
        # Detailed cases (top 10)
        if plagiarism_cases:
            detailed_data = [["Type", "Similarity", "Description"]]
            for case in plagiarism_cases[:10]:  # Top 10
                similarity = case.get('similarity_score', 0)
                detailed_data.append([
                    case.get('type', 'Unknown'),
                    f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "N/A",
                    case.get('description', 'No description')[:50] + "..."
                ])
            
            detailed_table = Table(detailed_data, colWidths=[1.5*inch, 1*inch, 3*inch])
            detailed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(detailed_table)
        
        return content
    
    def _create_recommendations(self, analysis_results: Dict) -> List:
        """
        Create recommendations section
        """
        content = []
        
        # Section title
        title = Paragraph("Recommendations", self.heading_style)
        content.append(title)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis_results)
        
        for i, rec in enumerate(recommendations, 1):
            rec_para = Paragraph(f"{i}. {rec}", self.body_style)
            content.append(rec_para)
            content.append(Spacer(1, 6))
        
        return content
    
    def _create_appendices(self, analysis_results: Dict) -> List:
        """
        Create appendices section
        """
        content = []
        
        # Section title
        title = Paragraph("Appendices", self.heading_style)
        content.append(title)
        
        # Appendix A: Raw Data
        appendix_a = Paragraph("Appendix A: Raw Analysis Data", self.styles['Heading3'])
        content.append(appendix_a)
        
        # Raw data summary
        raw_data_text = f"""
        <b>Analysis Timestamp:</b> {datetime.now().isoformat()}<br/>
        <b>Total Sections Analyzed:</b> {len(analysis_results.get('sections', {}))}<br/>
        <b>Anomalies Detected:</b> {len(analysis_results.get('anomalies', []))}<br/>
        <b>Plagiarism Cases:</b> {len(analysis_results.get('plagiarism', []))}<br/>
        <b>External Sources:</b> {len(analysis_results.get('external_data', []))}
        """
        
        raw_data_para = Paragraph(raw_data_text, self.body_style)
        content.append(raw_data_para)
        
        return content
    
    def _generate_key_findings(self, analysis_results: Dict) -> str:
        """
        Generate key findings summary
        """
        anomalies = analysis_results.get('anomalies', [])
        plagiarism_cases = analysis_results.get('plagiarism', [])
        
        findings = []
        
        if not anomalies:
            findings.append("✓ No anomalies detected - document appears clean")
        else:
            high_severity = len([a for a in anomalies if a.get('severity') == 'high'])
            if high_severity > 0:
                findings.append(f"⚠️ {high_severity} high-severity anomalies require immediate attention")
            else:
                findings.append("✓ No high-severity anomalies detected")
        
        if not plagiarism_cases:
            findings.append("✓ No plagiarism issues detected - content appears original")
        else:
            high_similarity = len([p for p in plagiarism_cases if p.get('severity') == 'high'])
            if high_similarity > 0:
                findings.append(f"⚠️ {high_similarity} high-similarity plagiarism cases detected")
            else:
                findings.append("✓ No high-similarity plagiarism cases detected")
        
        return "<br/>".join(findings)
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """
        Generate recommendations based on analysis results
        """
        recommendations = []
        
        anomalies = analysis_results.get('anomalies', [])
        plagiarism_cases = analysis_results.get('plagiarism', [])
        
        # Anomaly-based recommendations
        if anomalies:
            high_severity = [a for a in anomalies if a.get('severity') == 'high']
            if high_severity:
                recommendations.append(f"Address {len(high_severity)} high-severity anomalies immediately")
            
            missing_sections = [a for a in anomalies if a.get('subtype') == 'missing_section']
            if missing_sections:
                recommendations.append("Add missing critical sections to improve document completeness")
            
            financial_anomalies = [a for a in anomalies if a.get('type') == 'financial_anomaly']
            if financial_anomalies:
                recommendations.append("Review and verify all financial data for accuracy")
        
        # Plagiarism-based recommendations
        if plagiarism_cases:
            high_similarity = [p for p in plagiarism_cases if p.get('severity') == 'high']
            if high_similarity:
                recommendations.append(f"Review {len(high_similarity)} high-similarity plagiarism cases")
            
            external_plagiarism = [p for p in plagiarism_cases if p.get('type') == 'external_plagiarism']
            if external_plagiarism:
                recommendations.append("Verify originality of content and add proper citations if needed")
        
        # General recommendations
        if not anomalies and not plagiarism_cases:
            recommendations.append("Document appears to be in good condition - proceed with confidence")
        else:
            recommendations.append("Review all flagged issues before finalizing the document")
        
        return recommendations
    
    def generate_excel_report(self, analysis_results: Dict, output_path: str = None) -> str:
        """
        Generate Excel report with detailed data
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/drhp_analysis_{timestamp}.xlsx"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            self._create_summary_sheet(writer, analysis_results)
            
            # Anomalies sheet
            if analysis_results.get('anomalies'):
                self._create_anomalies_sheet(writer, analysis_results['anomalies'])
            
            # Plagiarism sheet
            if analysis_results.get('plagiarism'):
                self._create_plagiarism_sheet(writer, analysis_results['plagiarism'])
            
            # Sections sheet
            if analysis_results.get('sections'):
                self._create_sections_sheet(writer, analysis_results['sections'])
        
        self.logger.info(f"Excel report generated: {output_path}")
        return output_path
    
    def _create_summary_sheet(self, writer, analysis_results: Dict):
        """
        Create summary sheet in Excel
        """
        summary_data = {
            'Metric': [
                'Total Sections',
                'Total Anomalies',
                'High Severity Anomalies',
                'Total Plagiarism Cases',
                'High Similarity Cases',
                'External Sources',
                'Analysis Date'
            ],
            'Value': [
                len(analysis_results.get('sections', {})),
                len(analysis_results.get('anomalies', [])),
                len([a for a in analysis_results.get('anomalies', []) if a.get('severity') == 'high']),
                len(analysis_results.get('plagiarism', [])),
                len([p for p in analysis_results.get('plagiarism', []) if p.get('severity') == 'high']),
                len(analysis_results.get('external_data', [])),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Summary', index=False)
    
    def _create_anomalies_sheet(self, writer, anomalies: List[Dict]):
        """
        Create anomalies sheet in Excel
        """
        if not anomalies:
            return
        
        # Convert anomalies to DataFrame
        df = pd.DataFrame(anomalies)
        df.to_excel(writer, sheet_name='Anomalies', index=False)
    
    def _create_plagiarism_sheet(self, writer, plagiarism_cases: List[Dict]):
        """
        Create plagiarism sheet in Excel
        """
        if not plagiarism_cases:
            return
        
        # Convert plagiarism cases to DataFrame
        df = pd.DataFrame(plagiarism_cases)
        df.to_excel(writer, sheet_name='Plagiarism', index=False)
    
    def _create_sections_sheet(self, writer, sections: Dict[str, str]):
        """
        Create sections sheet in Excel
        """
        if not sections:
            return
        
        section_data = []
        for section_name, content in sections.items():
            section_data.append({
                'Section': section_name,
                'Word Count': len(content.split()),
                'Character Count': len(content),
                'Content Preview': content[:200] + "..." if len(content) > 200 else content
            })
        
        df = pd.DataFrame(section_data)
        df.to_excel(writer, sheet_name='Sections', index=False)
    
    def generate_json_report(self, analysis_results: Dict, output_path: str = None) -> str:
        """
        Generate JSON report for API consumption
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/drhp_analysis_{timestamp}.json"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add metadata
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_drhp_analysis',
                'version': '1.0.0'
            },
            'analysis_results': analysis_results
        }
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report generated: {output_path}")
        return output_path
