import os
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
import io
import base64
import re

# Load environment variables
load_dotenv()

class ReportComparator:
    def __init__(self):
        # Initialize the LLM with a more powerful model for deeper analysis
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.2,
            max_tokens=4000  # Ensure we have room for comprehensive responses
        )
        
    def _read_pdf(self, url, max_retries=3):
        """Download and extract text from PDF with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"Downloading PDF from: {url} (Attempt {attempt+1}/{max_retries})")
                response = requests.get(url, timeout=60)  # 60 second timeout
                
                if response.status_code != 200:
                    print(f"Failed with status code {response.status_code}, retrying...")
                    time.sleep(2)  # Wait before retrying
                    continue
                
                pdf = PdfReader(BytesIO(response.content))
                total_pages = len(pdf.pages)
                print(f"PDF has {total_pages} pages - reading all pages")
                
                text = ""
                for i in range(total_pages):
                    if i % 5 == 0:
                        print(f"Processing page {i+1}/{total_pages}...")
                    try:
                        page_text = pdf.pages[i].extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                        else:
                            print(f"Warning: Page {i+1} appears to be empty or contains non-extractable content")
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {i+1}: {str(e)}")
                
                print(f"Successfully extracted content from {total_pages} pages")
                return text
                
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to process PDF after {max_retries} attempts: {str(e)}")
                time.sleep(3)  # Wait before retrying
    
    def _extract_key_sections(self, text):
        """Extract important sections from the report to reduce noise"""
        # Look for common financial section headers
        key_phrases = [
            "financial highlights", "financial statements", "statement of financial position",
            "income statement", "balance sheet", "cash flow", "statement of comprehensive income",
            "key performance indicators", "financial review", "risk management", 
            "capital adequacy", "liquidity", "shariah compliance", "board of directors report"
        ]
        
        # Extract paragraphs containing these phrases (case insensitive)
        important_content = ""
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            lower_para = paragraph.lower()
            if any(phrase in lower_para for phrase in key_phrases):
                important_content += paragraph + "\n\n"
                
                # Also include the next paragraph for context
                idx = paragraphs.index(paragraph)
                if idx + 1 < len(paragraphs):
                    important_content += paragraphs[idx + 1] + "\n\n"
        
        # If we didn't find enough content, return the original
        if len(important_content) < len(text) * 0.2:
            return text
            
        return important_content
    
    def _chunk_text(self, text, chunk_size=8000):
        """Split text into manageable chunks"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    def _generate_comparative_charts(self, metrics_data):
        """Generate visualization charts for comparative analysis"""
        charts = []
        
        try:
            # Example: ROE comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            banks = list(metrics_data.keys())
            roe_values = [metrics_data[bank].get('ROE', 0) for bank in banks]
            
            ax.bar(banks, roe_values)
            ax.set_ylabel('ROE (%)')
            ax.set_title('Return on Equity Comparison')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Convert to base64 for embedding in markdown
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            charts.append(f"![ROE Comparison](data:image/png;base64,{img_str})")
            plt.close(fig)
            
            # Add more charts as needed...
            
        except Exception as e:
            print(f"Error generating charts: {str(e)}")
        
        return charts
    
    def _extract_metrics_from_text(self, result_text, bank_names):
        """Extract actual metrics from analysis text"""
        metrics_data = {}
        
        # Define patterns to look for metrics
        patterns = {
            'ROE (%)': r'(?:ROE|Return on Equity)[^\d]*(\d+\.?\d*)\s*%',
            'ROA (%)': r'(?:ROA|Return on Assets)[^\d]*(\d+\.?\d*)\s*%',
            'Cost-to-Income Ratio (%)': r'(?:Cost[- ]to[- ]Income|Cost Income)[^\d]*(\d+\.?\d*)\s*%',
            'NPL Ratio (%)': r'(?:NPL|Non[- ]Performing Loan)[^\d]*(\d+\.?\d*)\s*%',
            'Coverage Ratio (%)': r'(?:Coverage Ratio|Provision Coverage)[^\d]*(\d+\.?\d*)\s*%',
            'CET1 Ratio (%)': r'(?:CET1|Common Equity Tier 1)[^\d]*(\d+\.?\d*)\s*%',
            'CAR (%)': r'(?:CAR|Capital Adequacy Ratio)[^\d]*(\d+\.?\d*)\s*%',
            'Asset Growth (%)': r'(?:Asset Growth)[^\d]*(\d+\.?\d*)\s*%',
            'Loan Growth (%)': r'(?:Loan Growth|Financing Growth)[^\d]*(\d+\.?\d*)\s*%'
        }
        
        # Initialize metrics data structure
        for metric in patterns.keys():
            metrics_data[metric] = {}
            for bank in bank_names:
                metrics_data[metric][bank] = "N/A"
        
        # Try to extract metrics for each bank
        for bank in bank_names:
            # Find paragraphs mentioning the bank
            bank_paragraphs = []
            for i in range(len(result_text) - len(bank)):
                if result_text[i:i+len(bank)].lower() == bank.lower():
                    # Extract a window around the mention
                    start = max(0, i - 200)
                    end = min(len(result_text), i + 400)
                    bank_paragraphs.append(result_text[start:end])
            
            # Look for metrics in these paragraphs
            for paragraph in bank_paragraphs:
                for metric, pattern in patterns.items():
                    match = re.search(pattern, paragraph, re.IGNORECASE)
                    if match:
                        metrics_data[metric][bank] = match.group(1)
        
        return metrics_data
    
    def _generate_metrics_table(self, client_name, competitor_names, metrics_data=None):
        """Generate a metrics comparison table based on the analysis"""
        if not metrics_data:
            # Default metrics if none were extracted
            metrics = [
                'ROE (%)', 'ROA (%)', 'Cost-to-Income Ratio (%)', 
                'NPL Ratio (%)', 'Coverage Ratio (%)',
                'CET1 Ratio (%)', 'CAR (%)',
                'Asset Growth (%)', 'Loan Growth (%)'
            ]
            
            metrics_data = {}
            for metric in metrics:
                metrics_data[metric] = {client_name: "N/A"}
                for competitor in competitor_names:
                    metrics_data[metric][competitor] = "N/A"
        
        # Check if we have any actual values
        has_values = False
        for metric, values in metrics_data.items():
            for bank, value in values.items():
                if value != "N/A":
                    has_values = True
                    break
            if has_values:
                break
        
        # If no values were found, don't include the table
        if not has_values:
            return ""
            
        table = "## Key Financial Metrics Comparison\n\n"
        table += "| Metric | " + client_name + " | " + " | ".join(competitor_names) + " |\n"
        table += "|--------|" + "--------|" * (len(competitor_names) + 1) + "\n"
        
        for metric, values in metrics_data.items():
            row = f"| {metric} | {values.get(client_name, 'N/A')} | "
            row += " | ".join([values.get(comp, "N/A") for comp in competitor_names]) + " |\n"
            table += row
        
        return table
    
    def compare_reports(self, client_name, client_url, competitor_names, competitor_urls):
        """Compare financial reports using CrewAI"""
        # Process PDFs
        print(f"Processing {client_name} report...")
        try:
            client_text = self._read_pdf(client_url)
            client_key_content = self._extract_key_sections(client_text)
            print(f"Extracted key sections from {client_name} report")
        except Exception as e:
            print(f"Error processing {client_name} report: {str(e)}")
            raise
        
        competitor_texts = {}
        competitor_key_contents = {}
        for name, url in zip(competitor_names, competitor_urls):
            try:
                print(f"Processing {name} report...")
                competitor_texts[name] = self._read_pdf(url)
                competitor_key_contents[name] = self._extract_key_sections(competitor_texts[name])
                print(f"Extracted key sections from {name} report")
            except Exception as e:
                print(f"Error processing {name} report: {str(e)}")
                raise
        
        # Create AI agents with improved prompts
        financial_analyst = Agent(
            role="Senior Banking Financial Analyst",
            goal="Extract and compare key financial metrics and ratios with deep second-level interpretation",
            backstory="""Expert in Islamic and conventional banking with 25+ years experience analyzing financial statements.
            You specialize in extracting and comparing financial metrics across banks, with deep knowledge of UAE banking sector.
            You are known for identifying underlying patterns and second-order effects that others miss.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )
        
        strategic_advisor = Agent(
            role="Chief Strategy Officer",
            goal=f"Provide strategic insights and recommendations for {client_name} based on comprehensive financial analysis",
            backstory=f"""Former McKinsey banking practice lead with expertise in UAE banking sector.
            You provide strategic analysis and actionable recommendations to help {client_name} outperform competitors.
            You excel at identifying structural advantages and hidden vulnerabilities in business models.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        risk_analyst = Agent(
            role="Risk Management Specialist",
            goal="Analyze risk metrics and compliance aspects in detail",
            backstory="""Former central bank regulator with deep expertise in Basel frameworks and Islamic banking risk structures.
            You specialize in identifying risk patterns and compliance issues in banking reports.
            You've helped banks navigate regulatory challenges during multiple financial crises.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        trend_analyst = Agent(
            role="Financial Trend Analyst",
            goal="Identify underlying patterns and second-order effects in financial data",
            backstory="Former chief economist with 25 years of experience in identifying hidden correlations and causal relationships in banking data.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        # Create tasks with improved instructions
        data_extraction_task = Task(
            description=f"""
            Analyze the annual reports of {client_name} and other banks ({', '.join(competitor_names)}).
            
            Extract the following key metrics for all banks and present them in a clear table format:
            1. Profitability: ROE, ROA, Net Profit Margin, Cost-to-Income Ratio
            2. Asset Quality: NPL Ratio, Coverage Ratio
            3. Capital Adequacy: CET1 Ratio, CAR
            4. Liquidity: LCR, NSFR, Loan-to-Deposit Ratio
            5. Growth: Asset Growth, Loan Growth, Deposit Growth
            6. Islamic Banking Specific (if applicable): Profit Rate on Islamic Accounts, Shariah-compliant assets %
            
            Be extremely precise with numbers and include exact figures from the reports.
            Also identify trends and patterns in these metrics across the banks.
            """,
            agent=financial_analyst,
            expected_output="Detailed financial metrics comparison with second-level insights"
        )
        
        strategic_analysis_task = Task(
            description=f"""
            Perform a comprehensive second-level strategic analysis for {client_name} compared to {', '.join(competitor_names)}.
            
            Your analysis must go beyond surface metrics to identify:
            1. Underlying drivers of performance differences
            2. Second-order effects of strategic decisions
            3. Hidden vulnerabilities and opportunities not explicitly stated
            4. Potential future scenarios based on current trajectories
            5. Structural advantages or disadvantages in business models
            
            For each insight, provide:
            - The primary data points supporting it
            - The deeper interpretation of what this means
            - Strategic implications for the client
            - Specific, actionable recommendations
            
            Your analysis should be sophisticated enough to identify patterns and implications that would not be obvious from a first-level reading of the financial statements.
            """,
            agent=strategic_advisor,
            expected_output="Deep strategic analysis with second-level insights",
            context=[data_extraction_task]
        )
        
        risk_assessment_task = Task(
            description=f"""
            Analyze the risk profile of {client_name} compared to {', '.join(competitor_names)}.
            
            Focus on:
            1. Credit risk exposure and management
            2. Market risk sensitivity
            3. Operational risk indicators
            4. Regulatory compliance positioning
            5. Islamic banking specific risks (if applicable)
            
            Provide a comprehensive risk assessment with specific mitigation strategies.
            
            IMPORTANT: Identify both current and emerging risks. Prioritize risks based on potential impact and likelihood.
            For each significant risk, suggest a specific mitigation strategy.
            """,
            agent=risk_analyst,
            expected_output="Risk profile comparison with mitigation strategies",
            context=[data_extraction_task, strategic_analysis_task]
        )
        
        trend_analysis_task = Task(
            description=f"""
            Analyze the financial trends and patterns across {client_name} and {', '.join(competitor_names)}.
            
            Focus on:
            1. Multi-year performance trajectories
            2. Correlation between different financial metrics
            3. Seasonal patterns or cyclical behaviors
            4. Anomalies or deviations from expected patterns
            5. Leading indicators of future performance
            
            Provide insights on how these trends might evolve in the future and what they reveal about each bank's strategic positioning.
            """,
            agent=trend_analyst,
            expected_output="Comprehensive trend analysis with future projections",
            context=[data_extraction_task, strategic_analysis_task, risk_assessment_task]
        )
        
        # Prepare context for the crew - using key sections for more focused analysis
        client_chunks = self._chunk_text(client_key_content)
        competitor_chunks = {name: self._chunk_text(content) for name, content in competitor_key_contents.items()}
        
        # Create and run the crew
        crew = Crew(
            agents=[financial_analyst, strategic_advisor, risk_analyst, trend_analyst],
            tasks=[data_extraction_task, strategic_analysis_task, risk_assessment_task, trend_analysis_task],
            verbose=True,
            process=Process.sequential
        )

        # Update task contexts using proper dictionary format
        context_dict = {
            "client_name": client_name,
            "client_report": {
                "key_sections": client_key_content[:30000],
                "full_text": client_text[:20000]
            },
            "competitor_reports": {
                name: {
                    "key_sections": content[:30000],
                    "full_text": competitor_texts[name][:20000]
                } for name, content in competitor_key_contents.items()
            }
        }

        # Set context for each task properly
        # for task in crew.tasks:
        #     task.context = context_dict

        # Run the analysis
        print("Starting AI analysis...")
        raw_result = crew.kickoff(inputs=context_dict)
        print("Analysis complete!")
        
        # Convert the raw result to a string for processing
        result_text = str(raw_result)
        
        # Extract metrics from the analysis text
        all_bank_names = [client_name] + competitor_names
        metrics_data = self._extract_metrics_from_text(result_text, all_bank_names)
        
        # Create a consolidated report instead of separate sections
        metrics_table = self._generate_metrics_table(client_name, competitor_names, metrics_data)
        
        # Structure the results as a single consolidated report
        consolidated_report = f"""
        # Comprehensive Banking Analysis: {client_name} vs {', '.join(competitor_names)}
        
        {metrics_table}
        
        ## Executive Summary
        
        {result_text}
        """
        
        # Return both the consolidated report and individual sections for flexibility
        structured_results = {
            "consolidated_report": consolidated_report,
            "raw_result": result_text
        }
        
        return structured_results

