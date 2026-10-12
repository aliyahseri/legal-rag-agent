"""
Generate synthetic legal documents as PDF files.
Phase 10: Create 10 realistic legal documents using reportlab.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY
from rich.console import Console

console = Console()


class SyntheticDocumentGenerator:
    """Generate synthetic legal documents as PDFs."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the document generator.
        
        Args:
            output_dir: Base directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.case_law_dir = self.output_dir / "case_law"
        self.contracts_dir = self.output_dir / "contracts"
        
        self.case_law_dir.mkdir(parents=True, exist_ok=True)
        self.contracts_dir.mkdir(parents=True, exist_ok=True)
        
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            alignment=1  # Center
        ))
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=8
        ))
    
    def _create_pdf(self, filepath: Path, title: str, content: str):
        """
        Create a PDF file with the given content.
        
        Args:
            filepath: Path to save the PDF.
            title: Document title.
            content: Document content.
        """
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []
        
        # Add title
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Add content paragraphs
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Check if it's a heading
                if para.strip().endswith(':'):
                    story.append(Paragraph(para.strip(), self.styles['CustomHeading']))
                else:
                    story.append(Paragraph(para.strip(), self.styles['CustomBody']))
                story.append(Spacer(1, 0.1 * inch))
        
        doc.build(story)
    
    def generate_all(self):
        """Generate all 10 synthetic legal documents."""
        console.print("[blue]Generating synthetic legal documents...[/blue]\n")
        
        # Case law documents (cases 1, 3, 7, 9, 10)
        self._generate_case_1_force_majeure()
        self._generate_case_3_gdpr()
        self._generate_case_7_whistleblower()
        self._generate_case_9_trade_secrets()
        self._generate_case_10_arbitration()
        
        # Contract documents (cases 2, 4, 5, 6, 8)
        self._generate_contract_2_non_compete()
        self._generate_contract_4_ip_assignment()
        self._generate_contract_5_liquidated_damages()
        self._generate_contract_6_auto_renewal()
        self._generate_contract_8_construction()
        
        console.print(f"\n[green]Generated 10 synthetic legal documents.[/green]")
        console.print(f"[green]Case law: {self.case_law_dir}[/green]")
        console.print(f"[green]Contracts: {self.contracts_dir}[/green]")
    
    def _generate_case_1_force_majeure(self):
        """Generate Case 1: Force Majeure and COVID-19."""
        content = """
IN THE SUPERIOR COURT OF THE STATE OF CALIFORNIA
COUNTY OF SAN FRANCISCO

SMITH CORPORATION, Plaintiff,
v.
JONES ENTERPRISES, Defendant.

CASE NO. CV-2024-12345

FORCE MAJEURE AND COVID-19 PANDEMIC

This case concerns the interpretation of a force majeure clause in a commercial contract during the COVID-19 pandemic.

BACKGROUND:

The parties entered into a supply agreement on January 15, 2020, containing a force majeure clause that defined force majeure events to include "acts of God, natural disasters, pandemics, epidemics, and other events beyond the reasonable control of the parties."

On March 11, 2020, the World Health Organization declared COVID-19 a pandemic. Subsequently, government-imposed lockdowns and travel restrictions prevented Defendant from performing its obligations under the agreement.

LEGAL ISSUES:

1. Whether the COVID-19 pandemic constitutes a force majeure event under the contract.
2. Whether government-mandated lockdowns qualify as events beyond reasonable control.
3. Whether Defendant provided proper notice of the force majeure event.

ANALYSIS:

Under California law, force majeure clauses are construed according to their plain meaning. The contract explicitly includes "pandemics" and "epidemics" as force majeure events. The WHO declaration of COVID-19 as a pandemic on March 11, 2020, clearly falls within this definition.

The California Supreme Court has held that government actions that prevent performance may qualify as force majeure events when they are beyond the parties' control. The statewide lockdown orders issued by Governor Newsom were clearly beyond Defendant's control.

Regarding notice, the contract required written notice within ten business days. Defendant provided notice on March 20, 2020, which was within the required timeframe.

HOLDING:

The court finds that COVID-19 constitutes a force majeure event under the contract. Defendant's performance obligations were properly suspended during the pandemic period. Plaintiff's claims for breach of contract are dismissed.

CONCLUSION:

This case establishes that pandemics declared by the WHO, accompanied by government-mandated restrictions, qualify as force majeure events under standard commercial contract language. Parties should ensure their force majeure clauses specifically address pandemics and include appropriate notice provisions.
"""
        self._create_pdf(
            self.case_law_dir / "case_1_force_majeure.pdf",
            "Case 1: Force Majeure and COVID-19 Pandemic",
            content
        )
        console.print("[cyan]Generated: case_1_force_majeure.pdf[/cyan]")
    
    def _generate_case_3_gdpr(self):
        """Generate Case 3: GDPR Compliance."""
        content = """
EUROPEAN UNION GENERAL DATA PROTECTION REGULATION
GUIDANCE ON DATA SUBJECT RIGHTS

This document provides guidance on the implementation of GDPR data subject rights under Regulation (EU) 2016/679.

ARTICLE 15 - RIGHT OF ACCESS:

Data subjects have the right to obtain confirmation from the data controller as to whether personal data concerning them are being processed, and, where that is the case, access to the personal data and the following information:

(a) the purposes of the processing;
(b) the categories of personal data concerned;
(c) the recipients or categories of recipient to whom the personal data have been or will be disclosed;
(d) where possible, the envisaged period for which the personal data will be stored;
(e) the existence of the right to request rectification or erasure of personal data.

ARTICLE 16 - RIGHT TO RECTIFICATION:

Data subjects shall have the right to obtain from the controller without undue delay the rectification of inaccurate personal data concerning them. Taking into account the purposes of the processing, the data subject shall have the right to have incomplete personal data completed, including by means of a supplementary statement.

ARTICLE 17 - RIGHT TO ERASURE ('RIGHT TO BE FORGOTTEN'):

The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay where one of the following grounds applies:

(a) the personal data are no longer necessary in relation to the purposes for which they were processed;
(b) the data subject withdraws consent on which the processing is based;
(c) the data subject objects to the processing;
(d) the personal data have been unlawfully processed.

ARTICLE 32 - DATA SECURITY:

Taking into account the state of the art, the cost of implementation and the nature, scope, context and purposes of processing as well as the risk of varying likelihood and severity for the rights and freedoms of natural persons, the controller and the processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.

PENALTIES:

Under Article 83, non-compliance with GDPR may result in administrative fines up to €20 million or 4% of total global annual turnover, whichever is higher.
"""
        self._create_pdf(
            self.case_law_dir / "case_3_gdpr_compliance.pdf",
            "Case 3: GDPR Data Subject Rights",
            content
        )
        console.print("[cyan]Generated: case_3_gdpr_compliance.pdf[/cyan]")
    
    def _generate_case_7_whistleblower(self):
        """Generate Case 7: Whistleblower Protection."""
        content = """
WHISTLEBLOWER PROTECTION UNDER FEDERAL LAW

This document summarizes whistleblower protections under key federal statutes.

SARBANES-OXLEY ACT OF 2002 (SOX):

Section 806 of SOX provides protection for employees of publicly traded companies who report securities fraud. Protected disclosures include:

- Violations of federal securities laws
- Fraud against shareholders
- Any provision of SOX itself

Employees must report the violation to a federal regulatory or law enforcement agency, Congress, or a person with supervisory authority over the employee. Remedies include reinstatement, back pay, and compensatory damages.

DODD-FRANK WALL STREET REFORM ACT:

Section 922 of Dodd-Frank created a comprehensive whistleblower program for the Securities and Exchange Commission. Key provisions:

- Anonymous reporting to the SEC
- Monetary awards of 10-30% of monetary sanctions exceeding $1 million
- Anti-retaliation protection for employees who report violations

The statute of limitations is six years from the violation or three years from when the violation was discovered.

DEFENSE CONTRACTOR WHISTLEBLOWER PROTECTION ACT:

Protects employees of defense contractors who report:

- Mismanagement of a Department of Defense contract
- Gross waste of federal funds
- Abuse of authority
- Violations of law, rule, or regulation
- Specific dangers to public health or safety

FALSE CLAIMS ACT (FCA):

The FCA allows private citizens (qui tam relators) to sue on behalf of the government for fraud against federal programs. Successful relators receive 15-30% of the recovery. The FCA includes anti-retaliation provisions protecting employees who investigate or report FCA violations.

RECENT DEVELOPMENTS:

The U.S. Supreme Court in Digital Realty Trust, Inc. v. Somers (2018) held that SOX's anti-retaliation provisions only protect employees who report to the SEC, not those who report internally. This decision highlights the importance of understanding the specific reporting requirements of each whistleblower statute.
"""
        self._create_pdf(
            self.case_law_dir / "case_7_whistleblower_protection.pdf",
            "Case 7: Whistleblower Protection Under Federal Law",
            content
        )
        console.print("[cyan]Generated: case_7_whistleblower_protection.pdf[/cyan]")
    
    def _generate_case_9_trade_secrets(self):
        """Generate Case 9: Trade Secrets and DTSA."""
        content = """
DEFEND TRADE SECRETS ACT (DTSA)
18 U.S.C. § 1836 et seq.

This document outlines the federal framework for trade secret protection under the Defend Trade Secrets Act of 2016.

DEFINITION OF TRADE SECRET:

Under 18 U.S.C. § 1839(3), a trade secret is defined as information that:

(a) Derives independent economic value, actual or potential, from not being generally known to, and not being readily ascertainable by proper means by, another person who can obtain economic value from its disclosure or use; and
(b) Is the subject of efforts that are reasonable under the circumstances to maintain its secrecy.

MISAPPROPRIATION:

Under 18 U.S.C. § 1839(5), misappropriation includes:

- Acquisition of a trade secret by improper means
- Disclosure or use of a trade secret without consent when acquired by improper means or in circumstances giving rise to a duty to maintain secrecy
- Disclosure or use of a trade secret without consent when known to be acquired through improper means

CIVIL REMEDIES:

18 U.S.C. § 1836(b)(2) provides for civil remedies including:

(a) Injunctive relief to prevent actual or threatened misappropriation
(b) Damages for actual loss and unjust enrichment
(c) Exemplary damages up to twice the amount of actual damages for willful or malicious misconduct
(d) Reasonable attorney's fees in cases of bad faith

EX PARTE SEIZURE:

Under 18 U.S.C. § 1836(b)(2)(A)(i), courts may issue ex parte seizure orders to prevent dissemination of trade secrets when:

- The applicant shows that immediate and irreparable injury will occur
- The applicant is likely to succeed on the merits
- The harm to the applicant outweighs the harm to the defendant
- The applicant is likely to succeed in identifying misappropriated information
- The seizure would not cause substantial harm to legitimate interests

CRIMINAL PROSECUTION:

18 U.S.C. § 1832 provides criminal penalties for trade secret theft, including fines up to $5 million and imprisonment up to 10 years for individuals, or $10 million for organizations.

RELATIONSHIP TO STATE LAW:

The DTSA does not preempt state trade secret laws. It provides a federal cause of action that coexists with state statutes such as the Uniform Trade Secrets Act (UTSA), which has been adopted in 48 states plus the District of Columbia.
"""
        self._create_pdf(
            self.case_law_dir / "case_9_trade_secrets_dtsa.pdf",
            "Case 9: Trade Secrets and the Defend Trade Secrets Act",
            content
        )
        console.print("[cyan]Generated: case_9_trade_secrets_dtsa.pdf[/cyan]")
    
    def _generate_case_10_arbitration(self):
        """Generate Case 10: Arbitration Clause Enforceability."""
        content = """
FEDERAL ARBITRATION ACT
9 U.S.C. § 1 et seq.

This document addresses the enforceability requirements for arbitration clauses under federal law.

SECTION 2 - VALIDITY, IRREVOCABILITY, AND ENFORCEMENT:

A written provision in any maritime transaction or a contract evidencing a transaction involving commerce to settle by arbitration a controversy thereafter arising out of such contract or transaction, or the refusal to perform the whole or any part thereof, or an agreement in writing to submit to arbitration an existing controversy arising out of such a contract, transaction, or refusal, shall be valid, irrevocable, and enforceable, save upon such grounds as exist at law or in equity for the revocation of any contract.

REQUIREMENTS FOR ENFORCEABILITY:

1. WRITTEN AGREEMENT: The arbitration agreement must be in writing. Electronic signatures and agreements formed through clickwrap or browsewrap methods generally satisfy this requirement.

2. MUTUAL ASSENT: Both parties must manifest assent to the arbitration agreement. Courts examine the totality of circumstances to determine whether mutual assent exists.

3. SCOPE OF AGREEMENT: The agreement must clearly define the scope of disputes subject to arbitration. Vague or ambiguous language may render the agreement unenforceable as to certain claims.

4. CONSPICUOUSNESS: For consumer agreements, the arbitration clause must be reasonably conspicuous. Buried clauses in fine print may be deemed unconscionable.

SUPREME COURT PRECEDENT:

AT&T MOBILITY LLC v. CONCEPCION (2011): The Court held that the Federal Arbitration Act preempts state laws that prohibit arbitration agreements from waiving class action procedures.

AMERICAN EXPRESS CO. v. ITALIAN COLORS RESTAURANT (2013): The Court found that the FAA does not permit courts to invalidate arbitration agreements on the ground that the costs of arbitrating a statutory claim exceed the potential recovery.

EPIC SYSTEMS CORP. v. LEWIS (2018): The Court held that arbitration agreements requiring individual arbitration of employment disputes are enforceable under the FAA, notwithstanding the National Labor Relations Act.

EXCEPTIONS:

Arbitration agreements may be unenforceable if:

- They are the product of fraud or duress
- They are unconscionable
- They violate a strong public policy
- The dispute involves a non-arbitrable subject matter (e.g., criminal cases, family law)

CONCLUSION:

Arbitration clauses are generally enforceable under the Federal Arbitration Act when they are in writing, supported by mutual assent, and not unconscionable. Courts strongly favor arbitration as a means of resolving disputes efficiently.
"""
        self._create_pdf(
            self.case_law_dir / "case_10_arbitration_enforceability.pdf",
            "Case 10: Arbitration Clause Enforceability",
            content
        )
        console.print("[cyan]Generated: case_10_arbitration_enforceability.pdf[/cyan]")
    
    def _generate_contract_2_non_compete(self):
        """Generate Contract 2: Non-Compete Agreement."""
        content = """
NON-COMPETITION AND CONFIDENTIALITY AGREEMENT

THIS AGREEMENT is made and entered into as of January 1, 2024, by and between TECH INNOVATIONS CORP. ("Company") and JOHN DOE ("Employee").

SECTION 1. NON-COMPETITION COVENANT:

1.1 During the term of employment and for a period of twelve (12) months following termination of employment for any reason, Employee agrees not to engage in any business that competes directly with the Company within the geographic area of fifty (50) miles from any Company office.

1.2 "Competition" includes, but is not limited to: (a) working for a competitor of the Company; (b) starting a business that offers similar products or services; (c) soliciting Company clients or customers; (d) soliciting Company employees to leave their employment.

1.3 This restriction applies regardless of whether Employee's termination is voluntary or involuntary, with or without cause.

SECTION 2. CONSIDERATION:

2.1 In consideration for Employee's obligations under this Agreement, the Company agrees to provide Employee with: (a) continued employment; (b) access to confidential information and trade secrets; (c) specialized training and development; and (d) other consideration as may be agreed upon.

SECTION 3. REASONABLENESS:

3.1 The parties acknowledge that the restrictions contained in this Agreement are reasonable in scope, duration, and geographic area, and are necessary to protect the Company's legitimate business interests.

3.2 The Company's legitimate business interests include: (a) protection of trade secrets and confidential information; (b) protection of customer relationships; (c) protection of investment in Employee training; and (d) prevention of unfair competition.

SECTION 4. CALIFORNIA RESIDENTS:

4.1 If Employee is a resident of California or performs work primarily in California, this Section 1 shall not apply, as non-compete agreements are generally void under California Business and Professions Code Section 16600.

4.2 For California residents, Employee agrees to the confidentiality and non-solicitation provisions of this Agreement, which remain enforceable under California law.

SECTION 5. REMEDIES:

5.1 Employee acknowledges that a breach of this Agreement would cause irreparable harm to the Company for which monetary damages would be inadequate. The Company shall be entitled to injunctive relief and specific performance to enforce this Agreement, in addition to any other remedies available at law or in equity.

SECTION 6. SEVERABILITY:

6.1 If any provision of this Agreement is held to be invalid or unenforceable, the remaining provisions shall continue in full force and effect. Any invalid provision shall be reformed to the minimum extent necessary to make it enforceable while preserving its original intent.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.
"""
        self._create_pdf(
            self.contracts_dir / "contract_2_non_compete.pdf",
            "Contract 2: Non-Competition Agreement",
            content
        )
        console.print("[cyan]Generated: contract_2_non_compete.pdf[/cyan]")
    
    def _generate_contract_4_ip_assignment(self):
        """Generate Contract 4: IP Assignment Agreement."""
        content = """
INTELLECTUAL PROPERTY ASSIGNMENT AGREEMENT

THIS AGREEMENT is made and entered into as of January 1, 2024, by and between SOFTWARE SOLUTIONS INC. ("Assignee") and JANE SMITH ("Assignor").

SECTION 1. ASSIGNMENT OF INTELLECTUAL PROPERTY:

1.1 Assignor hereby assigns, transfers, and conveys to Assignee all of Assignor's right, title, and interest in and to the following intellectual property:

(a) All inventions, discoveries, improvements, and innovations conceived, developed, or reduced to practice by Assignor during the term of employment;
(b) All patents, patent applications, and patent disclosures related to such inventions;
(c) All copyrights, copyright registrations, and copyrightable works created by Assignor;
(d) All trademarks, service marks, and trade names developed by Assignor;
(e) All trade secrets, know-how, and confidential information;
(f) All other intellectual property rights related to Assignor's work.

1.2 This assignment includes all rights of recovery for past, present, and future infringement of the assigned intellectual property.

SECTION 2. WORKS FOR HIRE:

2.1 To the extent any work created by Assignor qualifies as a "work made for hire" under the U.S. Copyright Act (17 U.S.C. § 101), Assignor acknowledges and agrees that such work is a work made for hire owned by Assignee.

2.2 To the extent any work does not qualify as a work made for hire, Assignor hereby assigns all rights in such work to Assignee as provided in Section 1.

SECTION 3. PATENT ASSIGNMENT:

3.1 Assignor agrees to execute all documents necessary to perfect Assignee's ownership of patent rights, including: (a) patent applications; (b) assignments; (c) powers of attorney; (d) declarations; and (e) any other documents required by patent offices.

3.2 Assignor agrees to cooperate with Assignee in the prosecution and maintenance of patent applications and patents.

SECTION 4. DEFENSE TRADE SECRETS ACT (DTSA):

4.1 Assignor acknowledges that the assigned intellectual property may include trade secrets protected under the Defend Trade Secrets Act, 18 U.S.C. § 1836 et seq.

4.2 Assignor agrees not to disclose or use any trade secrets of Assignee, except as authorized by Assignee.

4.3 Assignor understands that misappropriation of trade secrets may result in civil and criminal penalties under federal law.

SECTION 5. MORAL RIGHTS:

5.1 To the extent moral rights cannot be waived under applicable law, Assignor irrevocably agrees not to assert any moral rights against Assignee or its licensees.

SECTION 6. CONSIDERATION:

6.1 Assignee agrees to pay Assignor the sum of One Dollar ($1.00) and other good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged.

SECTION 7. GOVERNING LAW:

7.1 This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.

IN WITNESS WHEREOF, the parties have executed this Agreement.
"""
        self._create_pdf(
            self.contracts_dir / "contract_4_ip_assignment.pdf",
            "Contract 4: Intellectual Property Assignment Agreement",
            content
        )
        console.print("[cyan]Generated: contract_4_ip_assignment.pdf[/cyan]")
    
    def _generate_contract_5_liquidated_damages(self):
        """Generate Contract 5: Liquidated Damages Clause."""
        content = """
LIQUIDATED DAMAGES PROVISION

THIS LIQUIDATED DAMAGES PROVISION is incorporated into the Service Agreement between ABC SERVICES LLC ("Provider") and XYZ CORPORATION ("Client").

SECTION 1. PURPOSE:

1.1 The parties acknowledge that actual damages for breach of this Agreement would be difficult or impossible to determine with precision. Therefore, the parties agree to the liquidated damages set forth in this Provision as a reasonable pre-estimate of anticipated losses.

SECTION 2. LIQUIDATED DAMAGES FOR CONFIDENTIALITY BREACH:

2.1 In the event of a material breach of the confidentiality provisions of this Agreement, Client shall be entitled to liquidated damages in the amount of Fifty Thousand Dollars ($50,000) per breach.

2.2 This amount represents a reasonable estimate of the harm caused by disclosure of confidential information, including loss of competitive advantage, damage to reputation, and costs of remediation.

SECTION 3. LIQUIDATED DAMAGES FOR NON-COMPETE BREACH:

3.1 In the event of a material breach of the non-competition provisions of this Agreement, Client shall be entitled to liquidated damages in the amount of One Hundred Thousand Dollars ($100,000) per breach.

3.2 This amount represents a reasonable estimate of the harm caused by competitive activities, including loss of customers, damage to business relationships, and costs of enforcing rights.

SECTION 4. LIQUIDATED DAMAGES FOR DELAY:

4.1 In the event of delay in performance by Provider, Client shall be entitled to liquidated damages in the amount of Twenty-Five Thousand Dollars ($25,000) per day of delay.

4.2 This amount represents a reasonable estimate of the harm caused by delay, including additional costs, lost opportunities, and project disruption.

SECTION 5. REASONABLENESS:

5.1 The parties agree that the liquidated damages set forth in this Provision are reasonable in light of the anticipated or actual harm caused by breach and the difficulties of proving actual damages.

5.2 The parties acknowledge that these amounts are not penalties and are intended to compensate, not punish, the non-breaching party.

SECTION 6. MITIGATION:

6.1 The non-breaching party agrees to use reasonable efforts to mitigate damages, as required by applicable law. Liquidated damages shall be reduced by any amounts actually recovered through mitigation efforts.

SECTION 7. LEGAL PRINCIPLES:

7.1 Under Uniform Commercial Code Section 2-718, liquidated damages are enforceable if reasonable in light of the anticipated or actual harm and the difficulties of proof of loss.

7.2 Under common law principles, liquidated damages are enforceable if they are a genuine pre-estimate of loss and not extravagant or unconscionable in relation to the anticipated loss.

SECTION 8. SEVERABILITY:

8.1 If any liquidated damages provision is held to be unenforceable as a penalty, the parties agree that the court may reform the provision to the maximum enforceable amount.
"""
        self._create_pdf(
            self.contracts_dir / "contract_5_liquidated_damages.pdf",
            "Contract 5: Liquidated Damages Provision",
            content
        )
        console.print("[cyan]Generated: contract_5_liquidated_damages.pdf[/cyan]")
    
    def _generate_contract_6_auto_renewal(self):
        """Generate Contract 6: Auto-Renewal Clause."""
        content = """
AUTO-RENEWAL AND TERMINATION PROVISION

THIS PROVISION is incorporated into the Subscription Agreement between CLOUD SERVICES INC. ("Provider") and RETAIL COMPANY ("Customer").

SECTION 1. INITIAL TERM:

1.1 The initial term of this Agreement shall commence on January 1, 2024 and continue for a period of twelve (12) months (the "Initial Term").

SECTION 2. AUTOMATIC RENEWAL:

2.1 Unless either party provides written notice of non-renewal at least thirty (30) days prior to the end of the Initial Term or any Renewal Term, this Agreement shall automatically renew for successive twelve (12) month periods (each a "Renewal Term").

2.2 The renewal terms shall be on the same terms and conditions as the then-current Agreement, except that the fees may be increased by Provider with at least sixty (60) days prior written notice.

SECTION 3. NOTICE REQUIREMENTS:

3.1 Notice of non-renewal must be in writing and delivered to the other party at the address specified in this Agreement.

3.2 Notice may be delivered by: (a) personal delivery; (b) certified or registered mail; (c) overnight courier; or (d) email to the specified email address.

3.3 Notice is effective upon receipt.

SECTION 4. TERMINATION FOR CAUSE:

4.1 Either party may terminate this Agreement immediately upon written notice if the other party: (a) breaches a material term of this Agreement and fails to cure such breach within thirty (30) days after written notice; or (b) becomes insolvent or files for bankruptcy.

SECTION 5. TERMINATION WITHOUT CAUSE:

5.1 Customer may terminate this Agreement without cause by providing at least sixty (60) days prior written notice. In such event, Customer shall pay a termination fee equal to fifty percent (50%) of the remaining fees for the then-current term.

5.2 Provider may terminate this Agreement without cause by providing at least ninety (90) days prior written notice.

SECTION 6. EFFECT OF TERMINATION:

6.1 Upon termination, Customer shall pay all fees accrued through the effective date of termination.

6.2 Provider shall provide Customer with access to its data for a period of thirty (30) days following termination, after which such data may be deleted.

6.3 Each party shall return or destroy the other party's confidential information upon termination.

SECTION 7. SURVIVAL:

7.1 The following provisions shall survive termination: (a) confidentiality; (b) intellectual property rights; (c) limitation of liability; (d) indemnification; and (e) governing law.

SECTION 8. STATE LAW COMPLIANCE:

8.1 The parties acknowledge that certain states (e.g., California, New York) have specific requirements for auto-renewal clauses. This Provision is intended to comply with applicable state laws.

8.2 If any provision of this Provision is invalid under applicable state law, the parties agree to reform such provision to the minimum extent necessary to comply with such law while preserving its original intent.
"""
        self._create_pdf(
            self.contracts_dir / "contract_6_auto_renewal.pdf",
            "Contract 6: Auto-Renewal and Termination Provision",
            content
        )
        console.print("[cyan]Generated: contract_6_auto_renewal.pdf[/cyan]")
    
    def _generate_contract_8_construction(self):
        """Generate Contract 8: Construction Contract."""
        content = """
CONSTRUCTION SERVICES AGREEMENT

THIS AGREEMENT is made and entered into as of January 1, 2024, by and between BUILDERS CORP. ("Contractor") and PROPERTY DEVELOPMENT LLC ("Owner").

SECTION 1. SCOPE OF WORK:

1.1 Contractor shall provide construction services for the commercial building project located at 123 Main Street, Anytown, State (the "Project").

1.2 The scope of work includes: (a) site preparation and grading; (b) foundation work; (c) structural framing; (d) electrical and plumbing installation; (e) interior finishing; and (f) final inspection and cleanup.

1.3 Detailed specifications are set forth in Exhibit A attached hereto and incorporated by reference.

SECTION 2. CONTRACT PRICE:

2.1 The total contract price shall be Two Million Five Hundred Thousand Dollars ($2,500,000), payable as follows:

(a) $250,000 upon execution of this Agreement;
(b) $500,000 upon completion of foundation;
(c) $750,000 upon completion of structural framing;
(d) $750,000 upon completion of interior finishing;
(e) $250,000 upon final inspection and acceptance.

2.2 Price adjustments may be made for changes in scope approved by Owner in writing.

SECTION 3. COMPLETION DATE:

3.1 Contractor shall substantially complete the Project by December 31, 2024 (the "Completion Date").

3.2 Time is of the essence. Contractor shall use commercially reasonable efforts to complete the Project by the Completion Date.

3.3 Owner may grant extensions for delays caused by: (a) changes in scope; (b) force majeure events; (c) delays caused by Owner; or (d) other causes beyond Contractor's control.

SECTION 4. CHANGE ORDERS:

4.1 Any changes to the scope of work must be approved by Owner in writing via a change order.

4.2 Change orders shall specify: (a) the nature of the change; (b) any adjustment to the contract price; and (c) any adjustment to the completion date.

4.3 No work shall be performed outside the scope of work without an approved change order.

SECTION 5. WARRANTIES:

5.1 Contractor warrants that all work shall be performed in a workmanlike manner and in accordance with applicable building codes and industry standards.

5.2 Contractor provides a one-year warranty on all workmanship and materials from the date of substantial completion.

5.3 Manufacturer warranties on materials and equipment shall be assigned to Owner.

SECTION 6. INSURANCE:

6.1 Contractor shall maintain: (a) general liability insurance with minimum coverage of $2,000,000; (b) workers' compensation insurance as required by law; and (c) builder's risk insurance.

6.2 Certificates of insurance shall be provided to Owner prior to commencement of work.

SECTION 7. INDEMNIFICATION:

7.1 Contractor shall indemnify, defend, and hold harmless Owner from claims arising from Contractor's negligence or willful misconduct.

7.2 Owner shall indemnify, defend, and hold harmless Contractor from claims arising from Owner's negligence or willful misconduct.

SECTION 8. TERMINATION:

8.1 Owner may terminate this Agreement for cause if Contractor materially breaches this Agreement and fails to cure such breach within fifteen (15) days after written notice.

8.2 Contractor may terminate this Agreement if Owner fails to make payments when due and such failure continues for ten (10) days after written notice.

SECTION 9. DISPUTE RESOLUTION:

9.1 Disputes shall be resolved through good faith negotiations. If negotiations fail, disputes shall be submitted to mediation.

9.2 If mediation fails, disputes shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association.

SECTION 10. GOVERNING LAW:

10.1 This Agreement shall be governed by and construed in accordance with the laws of the State where the Project is located.

IN WITNESS WHEREOF, the parties have executed this Agreement.
"""
        self._create_pdf(
            self.contracts_dir / "contract_8_construction.pdf",
            "Contract 8: Construction Services Agreement",
            content
        )
        console.print("[cyan]Generated: contract_8_construction.pdf[/cyan]")


if __name__ == "__main__":
    generator = SyntheticDocumentGenerator(".")
    generator.generate_all()
