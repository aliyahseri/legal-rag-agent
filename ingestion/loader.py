"""
Legal document loader for PDF and text files.
Phase 3: Document loading with metadata preservation.
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from rich.console import Console

console = Console()


class LegalDocumentLoader:
    """Load legal documents from PDF and text files with metadata."""
    
    def __init__(self):
        """Initialize the document loader."""
        self.console = console
    
    def load_case_law(self, directory: str) -> List[Document]:
        """
        Load case law documents from a directory.
        
        Args:
            directory: Path to directory containing case law PDFs.
            
        Returns:
            List of Document objects with metadata.
        """
        docs = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            console.print(f"[yellow]Directory {directory} does not exist.[/yellow]")
            return docs
        
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {directory}[/yellow]")
            return docs
        
        console.print(f"[blue]Loading {len(pdf_files)} case law documents...[/blue]")
        
        for pdf_file in pdf_files:
            file_docs = self.load_single_pdf(str(pdf_file), doc_type="case_law")
            docs.extend(file_docs)
        
        console.print(f"[green]Loaded {len(docs)} pages from case law documents.[/green]")
        return docs
    
    def load_contracts(self, directory: str) -> List[Document]:
        """
        Load contract documents from a directory.
        
        Args:
            directory: Path to directory containing contract PDFs.
            
        Returns:
            List of Document objects with metadata.
        """
        docs = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            console.print(f"[yellow]Directory {directory} does not exist.[/yellow]")
            return docs
        
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {directory}[/yellow]")
            return docs
        
        console.print(f"[blue]Loading {len(pdf_files)} contract documents...[/blue]")
        
        for pdf_file in pdf_files:
            file_docs = self.load_single_pdf(str(pdf_file), doc_type="contract")
            docs.extend(file_docs)
        
        console.print(f"[green]Loaded {len(docs)} pages from contract documents.[/green]")
        return docs
    
    def load_single_pdf(self, path: str, doc_type: str = "case_law") -> List[Document]:
        """
        Load a single PDF file with metadata.
        
        Args:
            path: Path to the PDF file.
            doc_type: Type of document ("case_law" or "contract").
            
        Returns:
            List of Document objects with metadata.
        """
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        
        # Add metadata to each document
        filename = Path(path).name
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source": filename,
                "doc_type": doc_type,
                "page": i + 1
            })
        
        return docs
    
    def generate_synthetic_documents(self, output_dir: str) -> List[Document]:
        """
        Generate synthetic legal documents as text files if no PDFs exist.
        
        Args:
            output_dir: Directory to save synthetic documents.
            
        Returns:
            List of Document objects from synthetic files.
        """
        console.print("[yellow]No PDFs found. Generating synthetic legal documents...[/yellow]")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        synthetic_docs = {
            "force_majeure.txt": self._get_force_majeure_text(),
            "non_compete.txt": self._get_non_compete_text(),
            "gdpr_compliance.txt": self._get_gdpr_text(),
            "ip_assignment.txt": self._get_ip_assignment_text(),
            "liquidated_damages.txt": self._get_liquidated_damages_text()
        }
        
        docs = []
        for filename, content in synthetic_docs.items():
            file_path = output_path / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "doc_type": "synthetic",
                    "page": 1
                }
            )
            docs.append(doc)
        
        console.print(f"[green]Generated {len(docs)} synthetic documents.[/green]")
        return docs
    
    def _get_force_majeure_text(self) -> str:
        """Generate synthetic force majeure clause text."""
        return """
FORCE MAJEURE CLAUSE

This Force Majeure Clause is included in this Agreement to address events beyond the reasonable control of the parties that may affect performance under this Agreement.

1. DEFINITION OF FORCE MAJEURE
For purposes of this Agreement, "Force Majeure" shall mean any event or circumstance that is beyond the reasonable control of a party, including but not limited to: acts of God, natural disasters, war, terrorism, civil unrest, governmental actions, pandemics, epidemics, and other similar events.

2. COVID-19 PANDEMIC
The COVID-19 pandemic, declared by the World Health Organization in March 2020, constitutes a Force Majeure event under this Agreement. The widespread nature of the pandemic, government-mandated lockdowns, travel restrictions, and supply chain disruptions qualify as events beyond reasonable control.

3. NOTICE REQUIREMENTS
Upon occurrence of a Force Majeure event, the affected party must provide written notice to the other party within ten (10) business days, specifying the nature of the event and its expected impact on performance.

4. SUSPENSION OF OBLIGATIONS
Upon proper notice, the affected party's obligations under this Agreement shall be suspended for the duration of the Force Majeure event. Such suspension shall not constitute a breach of this Agreement.

5. MITIGATION EFFORTS
The affected party must use reasonable efforts to mitigate the effects of the Force Majeure event and resume performance as soon as practicable.

This clause is based on common law principles and has been upheld in various jurisdictions including New York, English law, and civil law systems.
"""
    
    def _get_non_compete_text(self) -> str:
        """Generate synthetic non-compete clause text."""
        return """
NON-COMPETITION AGREEMENT

This Non-Competition Agreement is entered into to protect the legitimate business interests of the Company.

1. RESTRICTION ON COMPETITION
During the term of employment and for a period of twelve (12) months following termination, the Employee agrees not to engage in any business that competes directly with the Company within the geographic area of fifty (50) miles from any Company office.

2. DEFINITION OF COMPETITION
Competition includes, but is not limited to: working for a competitor, starting a competing business, soliciting Company clients, or providing similar services to those offered by the Company.

3. CONSIDERATION
In consideration for these restrictions, the Company agrees to provide the Employee with: (a) continued employment, (b) access to confidential information, and (c) specialized training.

4. REASONABLENESS TEST
Courts evaluate non-compete agreements based on: (1) protection of legitimate business interests, (2) reasonable time restriction, (3) reasonable geographic scope, and (4) not unduly restricting public interest.

5. CALIFORNIA BUSINESS AND PROFESSIONS CODE SECTION 16600
Under California Business and Professions Code Section 16600, non-compete agreements are generally void except in limited circumstances involving the sale of a business or dissolution of partnership.

6. ENFORCEABILITY
Enforceability varies by jurisdiction. States like California generally prohibit non-competes, while states like New York enforce them if reasonable in scope and duration.
"""
    
    def _get_gdpr_text(self) -> str:
        """Generate synthetic GDPR compliance text."""
        return """
GDPR COMPLIANCE PROVISIONS

These provisions implement the requirements of the General Data Protection Regulation (GDPR) Regulation (EU) 2016/679.

1. DATA PROCESSING PRINCIPLES
The Data Processor shall process personal data in accordance with Article 5 of the GDPR, ensuring: (a) lawfulness, fairness, and transparency, (b) purpose limitation, (c) data minimization, (d) accuracy, (e) storage limitation, (f) integrity and confidentiality, and (g) accountability.

2. DATA SUBJECT RIGHTS
Data subjects have the right to: (a) access their personal data (Article 15), (b) rectification of inaccurate data (Article 16), (c) erasure ("right to be forgotten") (Article 17), (d) restriction of processing (Article 18), (e) data portability (Article 20), and (f) object to processing (Article 21).

3. DATA SECURITY MEASURES
Pursuant to Article 32 of the GDPR, the Data Processor shall implement appropriate technical and organizational measures to ensure a level of security appropriate to the risk, including: (a) pseudonymization and encryption, (b) confidentiality, integrity, availability, and resilience of processing systems, (c) ability to restore availability and access to personal data, and (d) regular testing of security measures.

4. DATA BREACH NOTIFICATION
In the event of a personal data breach under Article 33, the Data Processor shall notify the Data Controller without undue delay and, where feasible, not later than 72 hours after becoming aware of the breach.

5. INTERNATIONAL DATA TRANSFERS
Transfers of personal data to third countries require appropriate safeguards under Article 46, including: (a) binding corporate rules, (b) standard contractual clauses approved by the European Commission, or (c) adequacy decisions by the European Commission.

6. PENALTIES
Non-compliance with GDPR may result in fines up to €20 million or 4% of total global annual turnover, whichever is higher (Article 83).
"""
    
    def _get_ip_assignment_text(self) -> str:
        """Generate synthetic IP assignment clause text."""
        return """
INTELLECTUAL PROPERTY ASSIGNMENT AGREEMENT

This Agreement assigns all intellectual property rights created by the Employee to the Company.

1. ASSIGNMENT OF RIGHTS
The Employee hereby assigns to the Company all right, title, and interest in any and all intellectual property created, developed, or conceived by the Employee during the term of employment, including: (a) patents, (b) copyrights, (c) trademarks, (d) trade secrets, and (e) know-how.

2. DEFINITION OF WORKS FOR HIRE
All works created by the Employee within the scope of employment shall be considered "works made for hire" under the U.S. Copyright Act (17 U.S.C. § 101). If any work does not qualify as a work made for hire, the Employee hereby assigns all rights to the Company.

3. DEFENSE TRADE SECRETS ACT (DTSA)
Under 18 U.S.C. § 1836 of the Defend Trade Secrets Act, the Employee acknowledges that trade secrets are protected under federal law. Misappropriation of trade secrets may result in: (a) injunctive relief, (b) damages up to 2x actual loss, and (c) exemplary damages for willful misconduct.

4. PATENT ASSIGNMENT
The Employee agrees to execute all documents necessary to perfect the Company's ownership of patent rights, including: (a) patent applications, (b) assignments, (c) powers of attorney, and (d) declarations.

5. MORAL RIGHTS
To the extent moral rights cannot be waived under applicable law, the Employee irrevocably agrees not to assert such rights against the Company.

6. PRIOR INVENTIONS
The Employee discloses all prior inventions that are excluded from this assignment. The Company claims no rights to such prior inventions unless they are incorporated into Company products or services.

7. CONSIDERATION
The Employee's assignment of rights is part of the consideration for employment and does not require additional compensation beyond the Employee's salary and benefits.
"""
    
    def _get_liquidated_damages_text(self) -> str:
        """Generate synthetic liquidated damages clause text."""
        return """
LIQUIDATED DAMAGES CLAUSE

This Liquidated Damages Clause establishes predetermined damages for breach of contract.

1. PURPOSE OF CLAUSE
The parties agree that actual damages for breach of this Agreement would be difficult or impossible to determine with precision. Therefore, the parties agree to liquidated damages as a reasonable estimate of anticipated losses.

2. CALCULATION OF DAMAGES
In the event of material breach by either party, the non-breaching party shall be entitled to liquidated damages in the amount of: (a) $50,000 for breach of confidentiality, (b) $100,000 for breach of non-compete, or (c) $25,000 per day for delay in performance.

3. REASONABLENESS TEST
Courts enforce liquidated damages clauses only if: (a) damages are difficult to ascertain at contract formation, (b) the amount is reasonable in light of anticipated or actual harm, and (c) the amount is not a penalty.

4. DISTINCTION FROM PENALTIES
A penalty is designed to punish the breaching party, while liquidated damages are a genuine pre-estimate of loss. Courts will not enforce penalty clauses.

5. UCC SECTION 2-718
Under Uniform Commercial Code Section 2-718, liquidated damages are enforceable if reasonable in light of the anticipated or actual harm and the difficulties of proof of loss.

6. COMMON LAW PRINCIPLES
Common law jurisdictions generally follow similar principles. English law (as established in Dunlop Pneumatic Tyre Co Ltd v New Garage & Motor Co Ltd) requires that the liquidated damages be a genuine pre-estimate of loss, not extravagant or unconscionable.

7. MITIGATION
The non-breaching party must still take reasonable steps to mitigate damages, as required by common law principles.
"""
