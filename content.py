import re


def create_gradio_anchor_id(text: str, validation) -> str:
    """
    Replicates the ID format created by gr.Markdown(header_links=True).
    Example: "Paper Finder Validation" -> "h-paper-finder-validation"
    """
    text = text.lower()
    text = re.sub(r'\s+', '-', text) # Replace spaces with hyphens
    text = re.sub(r'[^\w-]', '', text) # Remove non-word characters
    if validation:
        return f"h-{text}-leaderboard-1"
    return f"h-{text}-leaderboard"


TITLE = """<h1 align="left" id="space-title">OpenHands Index</h1>"""

INTRO_PARAGRAPH = """
<p>
    The <strong>OpenHands Index</strong> is a comprehensive benchmark for evaluating AI coding agents across real-world software engineering tasks. As agents become more capable, we need ways to measure their performance across diverse challenges, from fixing bugs to building applications.
</p>


<p>
    The OpenHands Index assesses models across five categories: <strong>Issue Resolution</strong> (fixing bugs), <strong>Greenfield</strong> (building new apps), <strong>Frontend</strong> (UI development), <strong>Testing</strong> (test generation), and <strong>Information Gathering</strong>. All models are currently evaluated using the <a href="https://github.com/OpenHands/software-agent-sdk">OpenHands Software Agent SDK</a>. This provides a single view of both <strong>performance</strong> and <strong>cost efficiency</strong>, enabling fair comparisons between agents.
</p>


<p>
    For methodology details, see the <a href="/about" class="intro-link">About</a> page.
</p>
"""
SCATTER_DISCLAIMER = """
**Note:** Agents without cost data are displayed to the right of the vertical divider line.
"""
PARETO_DISCLAIMER = """
Agents names that are green are Pareto optimal, meaning they achieve the best performance for their cost. 
"""
BUG_FIXING_DESCRIPTION = """
The **Issue Resolution** category evaluates how well agents can diagnose and fix bugs in real-world codebases. This tests their ability to understand GitHub issues, navigate repositories, identify root causes, and implement correct fixes.
<br><br>
The scores shown below reflect performance aggregated across two distinct benchmarks: SWE-bench (text-based bug reports) and SWE-bench-multimodal (issues with visual context like screenshots or diagrams). 
<br><br>
For detailed results, use the links above to explore individual benchmarks.
<br>
"""
APP_CREATION_DESCRIPTION = """
The **Greenfield** category in OpenHands Index evaluates an agent's ability to build complete applications from scratch based on natural language specifications. This tests whether agents can understand requirements, design architecture, write modular code, and create working applications.
<br><br>
This category currently includes Commit0, which challenges agents to implement complete features and applications by generating the initial commit for a project.
<br><br>
For detailed results, use the links above to explore individual benchmark pages.
<br>
"""
FRONTEND_DEVELOPMENT_DESCRIPTION = """
The **Frontend** category evaluates agents on their ability to build user interfaces and web applications. This tests skills in HTML, CSS, JavaScript frameworks, responsive design, and creating interactive user experiences.
<br><br>
This category uses the dev set of <a href="https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal" target="_blank" rel="noopener noreferrer">SWE-bench Multimodal (Verified)</a>, a version of SWE-bench Multimodal that includes only <a href="https://github.com/OpenHands/benchmarks/blob/main/benchmarks/swebenchmultimodal/ambiguity_annotations.json" target="_blank" rel="noopener noreferrer">problems verified as solveable</a> by human review.
<br>
"""
TEST_GENERATION_DESCRIPTION = """
The **Testing** category evaluates agents on their ability to create comprehensive test suites for existing code. This tests their understanding of code behavior, edge cases, and the ability to write effective unit tests, integration tests, and end-to-end tests.
<br><br>
This category includes SWT-bench (Software Testing Benchmark), which challenges agents to generate high-quality test cases that achieve good coverage and catch real bugs.
<br>
"""

INFORMATION_GATHERING_DESCRIPTION = """
The **Information Gathering** category tests whether agents can effectively search for information, synthesize knowledge from multiple sources, and answer complex questions that require tool use and reasoning.
<br><br>
This category includes GAIA (General AI Assistant benchmark), which evaluates agents on real-world assistant tasks that require web search, file manipulation, and multi-step reasoning to gather and process information.
<br>
"""
SUBMISSION_CONFIRMATION = """
**Your agent has been submitted to OpenHands Index for evaluation.**
<br><br>
🙏 Thanks for contributing!
<br><br>
You'll receive a confirmation email from our team within 2 business days with next steps. We will reach out to you directly if further information is needed.
<br><br>
We appreciate your support in advancing scientific AI.
"""

# External URLs for benchmark descriptions
SCHOLAR_QA_CS_URL = "https://www.semanticscholar.org/paper/OpenScholar%3A-Synthesizing-Scientific-Literature-LMs-Asai-He/b40df4b273f255b3cb5639e220c8ab7b1bdb313e"
LITQA2_URL = "https://www.semanticscholar.org/paper/Language-agents-achieve-superhuman-synthesis-of-Skarlinski-Cox/fa5f9aa1cb6f97654ca8e6d279ceee1427a87e68"
ARXIV_DIGESTABLES_URL = "https://www.semanticscholar.org/paper/ArxivDIGESTables%3A-Synthesizing-Scientific-into-Newman-Lee/c7face35e84f2cb04fb1600d54298799aa0ed189"
SUPER_URL = "https://www.semanticscholar.org/paper/SUPER%3A-Evaluating-Agents-on-Setting-Up-and-Tasks-Bogin-Yang/053ef8299988680d47df36224bfccffc817472f1"
CORE_BENCH_URL = "https://www.semanticscholar.org/paper/CORE-Bench%3A-Fostering-the-Credibility-of-Published-Siegel-Kapoor/4c913d59d150fe7581386b87dfd9f90448a9adee"
DS1000_URL = "https://arxiv.org/abs/2211.11501"
DISCOVERY_BENCH_URL = "https://www.semanticscholar.org/paper/DiscoveryBench%3A-Towards-Data-Driven-Discovery-with-Majumder-Surana/48c83799530dc523ee01e6c1c40ad577d5c10a16"

# Helper function to create external links
def external_link(url, text, is_s2_url=False):
    url = f"{url}?utm_source=openhands_index" if is_s2_url else url
    return f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{text}</a>"

def internal_leaderboard_link(text, validation):
    anchor_id = create_gradio_anchor_id(text, validation)
    return f"<a href='#{anchor_id}'>{text}</a>"

# Function to get benchmark descriptions with validation flag
def get_benchmark_description(benchmark_name, validation):
    descriptions = {
    'PaperFindingBench': (
        "PaperFindingBench assesses an agent's ability to locate sets of papers based on a natural language "
        "description that may involve both the papers' content and metadata, such as the author or publication year."
    ),
    'LitQA2-FullText-Search': (
        f"A version of {internal_leaderboard_link('LitQA2-FullText', validation)} that isolates the retrieval aspect of the task. "
        f"This benchmark features the same multi-choice questions as {internal_leaderboard_link('LitQA2-FullText', validation)}, but the agent is not evaluated on answering the actual question "
        "but rather on providing a ranked list of papers in which the answer is likely to be found."
    ),
    'ScholarQA-CS2': (
        "ScholarQA-CS2 assesses long-form model responses to literature review questions in the domain of computer science. "
        "Answers are expected to be comprehensive reports, such as those produced by deep research systems. "
        f"This benchmark advances on the previously released {external_link(SCHOLAR_QA_CS_URL, 'ScholarQA-CS', is_s2_url=True)} "
        "by using queries from real-world usage, and introducing new evaluation methods for coverage and precision "
        "of both the report text and its citations."
    ),
    'LitQA2-FullText': (
        f"{external_link(LITQA2_URL, 'LitQA2', is_s2_url=True)}, a benchmark introduced by FutureHouse, gauges a model's ability to answer questions that require document retrieval from the scientific literature. "
        "It consists of multiple-choice questions that necessitate finding a unique paper and analyzing its detailed full text to spot precise information; these questions cannot be answered from a paper’s abstract. "
        "While the original version of the benchmark provided for each question the title of the paper in which the answer can be found, it did not specify the overall collection to search over. In our version, "
        "we search over the index we provide as part of the standard toolset. The “-FullText” suffix indicates we consider only the subset of LitQA2 questions for which "
        "the full-text version of the answering paper is open source and available in our index."
    ),
    'ArxivDIGESTables-Clean': (
        f"{external_link(ARXIV_DIGESTABLES_URL, 'ArxivDIGESTables', is_s2_url=True)} assesses the ability of models to construct literature review tables, i.e., tables whose rows are papers and whose columns constitute a set of "
        "aspects used to compare and contrast the papers. The goal is to construct such tables given a set of related papers and a table caption describing the user's goal. Generated tables are evaluated by "
        "comparing them to actual tables published in ArXiv papers. The “-Clean” suffix indicates a curated subset of ArxivDIGESTables which drops tables that are either trivial or impossible to reconstruct from full-texts."
    ),
    'SUPER-Expert': (
        "SUPER-Expert evaluates the capability of models in setting up and executing tasks from low-resource "
        "research repositories—centralized databases containing research data and related materials. "
        f"The \"-Expert\" split indicates the name of the most challenging split in the {external_link(SUPER_URL, 'original SUPER benchmark', is_s2_url=True)} "
        "that involves solving reproduction tasks from scratch and without any intermediate hints or details "
        "about the important landmarks involved in each task."
    ),
    'CORE-Bench-Hard': (
        "Core-Bench-Hard tests computational reproducibility, a task involving reproducing the results of a study "
        "using provided code and data. It consists of both language-only and vision-language challenges across "
        "multiple difficulty levels. "
        f"The \"-Hard\" split refers to the name of the most challenging split in the original {external_link(CORE_BENCH_URL, 'Core-bench benchmark', is_s2_url=True)} "
        "where only a README file is provided with no instructions or an auxiliary Dockerfile."
    ),
    'DS-1000': (
        "DS-1000 is an established code generation benchmark containing Python data science coding questions "
        "originally sourced from StackOverflow. It's designed to reflect an array of diverse, realistic, and "
        "practical use cases and directly involves many of the Python libraries commonly used in data science "
        f"and machine learning research. We split the original {external_link(DS1000_URL, 'dataset')} "
        "into 100 validation and 900 test problems."
    ),
    'DiscoveryBench': (
        "DiscoveryBench is the first comprehensive benchmark to formalize the multi-step process of data-driven "
        "analysis and discovery (i.e., data loading, transformation, statistical analysis, and modeling). "
        f"Originally introduced {external_link(DISCOVERY_BENCH_URL, 'here', is_s2_url=True)}, it is designed to systematically "
        "evaluate how well current LLMs can replicate or reproduce published scientific findings across diverse "
        "domains, including social science, biology, history, and more."
    ),
    'E2E-Bench': (
        "E2E-Bench is the \"decathlon\" of AI-assisted research. It measures whether a system can run the entire "
        "research pipeline, starting with an initial task description, to designing and performing (software) "
        "experiments, to analyzing and writing up the results."
    ),
    'E2E-Bench-Hard': (
        f"E2E-Bench-Hard is a more challenging variant of {internal_leaderboard_link('E2E-Bench', validation)}. Tasks are generated using the HypER system, "
        "which identifies research trends and proposes new, underexplored problems. Unlike the regular version, "
        "these tasks are not simplified or curated for accessibility; they are reviewed only for feasibility. "
        "This version is intended to test whether systems can handle more complex and less-structured research "
        f"scenarios, following the same end-to-end process as {internal_leaderboard_link('E2E-Bench', validation)}."
    )
    }
    
    return descriptions.get(benchmark_name, "")

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""@article{openhands-index,
    title={OpenHands Index},
    author={OpenHands Team},
    year={2025},
    eprint={TBD.TBD},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    secondaryClass={cs.CL}
}"""

LEGAL_DISCLAIMER_TEXT = """
<h2>Terms and Conditions</h2>
<p>
    OpenHands maintains this repository for agent evaluation submissions to OpenHands Index. To keep OpenHands Index fair and auditable, all evaluation logs and associated submission files will be made publicly available. This includes your benchmark inputs, model output responses, and other data and information related to your submission as needed to verify the results.
</p>
<br>
<p>
    Your submissions to OpenHands Index will be posted, scored, and ranked on the leaderboard at <a href="https://huggingface.co/spaces/OpenHands/openhands-index" target="_blank" rel="noopener noreferrer">https://huggingface.co/spaces/OpenHands/openhands-index</a>. You agree you have the rights to the materials you submit and that you will not share any personal, sensitive, proprietary, or confidential information.
</p>
"""

def format_error(msg):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{msg}</p>"


def format_warning(msg):
    return f"<p style='color: orange; font-size: 20px; text-align: center;'>{msg}</p>"


def format_log(msg):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{msg}</p>"


def hyperlink(link_url: str, text: str = "🔗") -> str:
    if not link_url or not isinstance(link_url, str):
        return str(text) # Or simply "" if link_url is bad
    return f'<a target="_blank" href="{link_url}">{text}</a>'


def hf_uri_to_web_url(uri: str) -> str:
    """
    Convert a Hugging Face-style URI like:
        hf://datasets/{namespace}/{repo}/{path...}
    into a public web URL:
        https://huggingface.co/datasets/{namespace}/{repo}/tree/main/{path...}
    """
    prefix = "hf://datasets/"
    if not uri.startswith(prefix):
        raise ValueError("URI must start with 'hf://datasets/'")

    parts = uri[len(prefix) :].split("/", 2)
    if len(parts) < 3:
        raise ValueError("Expected format: hf://datasets/{namespace}/{repo}/{path...}")

    namespace, repo, path = parts
    return f"https://huggingface.co/datasets/{namespace}/{repo}/tree/main/{path}"


css = """
/* CSS Color Variables aligned with OpenHands brand (openhands-ui/tokens.css) */
:root {
    /* Primary - Yellow */
    --color-primary-accent: #FFE165;
    --color-primary-light: #FFF3C0;
    --color-primary-dark: #BBA54A;
    
    /* Secondary - Green */
    --color-secondary-accent: #BCFF8C;
    --color-secondary-dark: #577641;
    
    /* Neutral - Grey scale */
    --color-neutral-50: #F7F8FB;
    --color-neutral-100: #EBEDF3;
    --color-neutral-200: #D4D8E7;
    --color-neutral-300: #B1B9D3;
    --color-neutral-700: #2F3137;
    --color-neutral-800: #222328;
    --color-neutral-900: #18191C;
    --color-neutral-950: #0D0D0F;
    
    /* Semantic colors */
    --color-primary-link: #2F3137;
    --color-primary-link-dark: #B1B9D3;
    --color-background-light: #F7F8FB;
    --color-background-dark: #18191C;
    --color-text-dark: #0D0D0F;
    --color-text-light: #F7F8FB;
    --color-button-hover: #222328;
    
    /* Danger/Error - Red */
    --color-danger: #FF684E;
}

/* This makes space for the huggingface header bar which must shown on HF spaces. */
/* FIXME Media queries don't seem to survive rendering. */
/* @media (min-width: 768px) { ... } */
gradio-app {
    padding-top: 65px;
}

/* Global Styles */
h2 {
    overflow: hidden;
}

/* Global link color styles */
.dark a {
    color: var(--color-primary-link-dark);
}
.dark a:hover {
    color: #dddddd;
}
.dark a:visited {
    color: #bbbbbb;
}

#intro-paragraph {
    font-size: 18px;
    max-width: 90%;
    padding-left: 35px;
    margin-top: 20px;
}

#intro-paragraph p,
#intro-paragraph li {
    font-size: 16px; 
    line-height: 1.8; 
}

/* Links in intro paragraph */
.intro-link {
    color: var(--color-primary-link);
    text-decoration: underline;
}
.dark .intro-link {
    color: var(--color-primary-link-dark);
}

#intro-paragraph ul {
    margin-top: 20px;
    margin-bottom: 20px;
}

#diagram-image {
    height: 100%;
}

#diagram-image img {
    width: 100%;
    height: 100%;
    object-fit: cover; 
}
#intro-category-paragraph {
    font-size: 18px;
    max-width: 90%;
    margin-top: 20px;
}

#intro-category-paragraph p,
#intro-category-paragraph li {
    font-size: 16px; 
    line-height: 1.8; 
}

#intro-category-paragraph ul {
    margin-top: 20px;
    margin-bottom: 20px;
}

#about-content {
    font-size: 18px;
    max-width: 60%;
    padding-left: 25px;
}
#category-intro {
    font-size: 18px;
    max-width: 60%;
}
#logo-image { 
    margin: 0;
    margin-bottom: 30px; 
    justify-content: flex-start;        
    max-width: 250px;       
    height: auto;           
}
#page-content-wrapper{
    padding-left: 25px;
}
.table-component{
    height: auto !important;
    max-height: none !important;
}
.table-wrap {
    max-height: none !important;
    height: auto !important;
    overflow-y: visible !important;
}
/* --- New Rules for Table Density --- */
table.gr-table th, table.gr-table td {
    padding: 4px 4px !important; 
    width: 1%;
    white-space: nowrap;
}
table.svelte-1e98i6s td {
    vertical-align: top !important;
}
table.gr-table {
    font-size: 15px !important;
}
.html-container {
    padding-top: 0 !important;
}
#scatter-disclaimer {
        overflow: visible !important;
}
#pareto-disclaimer {
    color: var(--color-primary-accent) !important;
}
thead.svelte-1e98i6s th {
    background: white !important;
}
.dark thead.svelte-1e98i6s th {
    background: var(--color-background-dark) !important;
}
.cell-wrap.svelte-v1pjjd {
    font-family: Arial, sans-serif;
    }
nav.svelte-ti537g.svelte-ti537g {
    justify-content: flex-start;
}
.nav-holder {
    padding-left: 20px !important;
}
#legend-markdown span {
    margin-right: 15px !important; 
}
#leaderboard-accordion .label-wrap {
    font-size: 1.4rem !important; 
    z-index: 10 !important;
    position: relative !important;
}
.dark #leaderboard-accordion .label-wrap {
    color: var(--color-primary-accent) !important; 
}
.dark block.svelte-1svsvh2 {
    background: var(--color-background-dark) !important;
}
.padding.svelte-phx28p {
    padding: 0 !important;
}
.sub-nav-bar-container {
    display: flex !important;
    flex-wrap: wrap !important; 
    align-items: center !important; 
    gap: 10px !important;
}
.dark .primary-link-button {
    color: var(--color-primary-link-dark);
}
.primary-link-button {
    background: none;
    border: none;
    padding: 0;
    margin: 0;
    font-family: inherit;
    font-size: 16px;
    color: var(--color-primary-link);
    text-decoration: none;
    cursor: pointer;
    white-space: nowrap;
}
.primary-link-button:hover {
    text-decoration: underline;
}
.sub-nav-label {
    font-weight: bold;
    font-size: 16px;
    display: flex;
    align-items: center;
}
.wrap-header-df th span{
    white-space: normal !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
    line-height: 1.2 !important;
    vertical-align: top !important;
    font-size: 12px !important;
    font-family: Arial, sans-serif;
}
.wrap-header-df th {
    height: auto !important;
}
.wrap-header-df .cell-wrap img {
    width: 16px;
    height: 16px;
    vertical-align: middle;
}
#legend-markdown img {
    width: 16px;
    height: 16px;
    vertical-align: middle;
}
/*------ Global tooltip styles ------*/
.tooltip-icon {
    display: inline-block;
    cursor: help;
    position: relative;
}
.tooltip-icon::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 125%;
    background-color: var(--color-background-dark);
    color: var(--color-text-light);
    padding: 10px;
    border-radius: 4px;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.2s;
    white-space: pre-line;
    width: max-content;
    text-align: left;
    pointer-events: none;
    max-width: 300px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
}
@media (max-width: 768px) {
    .tooltip-icon::after {
        max-width: 250px;
    }
}
.tooltip-icon:hover::after {
    opacity: 1;
}
/*------ Openness label tooltip styles ------*/
.styler,
#openness-label-html,
#agent-tooling-label-html {
    overflow: visible !important;
}
/*------ Table cell tooltip styles ------*/
.wrap.default.full,
span.wrap[tabindex="0"][role="button"][data-editable="false"] {
  overflow: visible !important;
}

.cell-tooltip-icon::after {
    height: fit-content;
    top: 125%;
}
/*------ Table column description tooltip styles ------*/
#legend-markdown,
#leaderboard-accordion {
    overflow: visible !important;
}

/* --- inside table tooltips --- */
.native-tooltip-icon {
    cursor: help;
    text-decoration: underline dotted 1px;
}
/* Main Nav bar styling */
.nav-holder nav {
    display: grid !important;
    grid-template-columns: auto auto auto auto auto 1fr auto auto !important;
    gap: 10px 20px !important; /* Vertical and horizontal spacing */
    width: 100% !important;
    align-items: center;
}
.nav-holder nav a[href*="about"] {
    grid-row: 1 !important;
    grid-column: 7 !important;
}
.nav-holder nav a[href*="submit"] {
    grid-row: 1 !important;
    grid-column: 8 !important;
    white-space: nowrap !important;
}
/* Divider line between header and category nav */
.nav-holder nav::after {
    content: ''; /* Required for pseudo-elements to appear */
    background-color: #C9C9C3;
    height: 1px; 
    grid-row: 2 !important;
    grid-column: 1 / -1 !important;
}

/* Horizontal scrolling for navigation */
.nav-holder nav {
    overflow-x: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.nav-holder nav::-webkit-scrollbar {
    display: none;
}

/* Category navigation buttons in row 3 */
.nav-holder nav a[href*="literature-understanding"],
.nav-holder nav a[href*="code-execution"],
.nav-holder nav a[href*="data-analysis"],
.nav-holder nav a[href*="discovery"] {
    grid-row: 3 !important;
    justify-self: center !important;
    width: fit-content !important;
    white-space: nowrap;
    flex-shrink: 0;
}

.nav-holder nav a[href*="literature-understanding"] { grid-column: 1 !important; }
.nav-holder nav a[href*="code-execution"] { grid-column: 2 !important; }
.nav-holder nav a[href*="data-analysis"] { grid-column: 3 !important; }
.nav-holder nav a[href*="discovery"] { grid-column: 4 !important; }

/* Navigation hover styles */
.nav-holder nav a[href*="about"]:hover,
.nav-holder nav a[href*="submit"]:hover,
.nav-holder nav a[href*="literature-understanding"]:hover,
.nav-holder nav a[href*="code-execution"]:hover,
.nav-holder nav a[href*="data-analysis"]:hover,
.nav-holder nav a[href*="discovery"]:hover {
    background-color: #FDF9F4;
}

.dark .nav-holder nav a[href*="about"]:hover,
.dark .nav-holder nav a[href*="submit"]:hover,
.dark .nav-holder nav a[href*="literature-understanding"]:hover,
.dark .nav-holder nav a[href*="code-execution"]:hover,
.dark .nav-holder nav a[href*="data-analysis"]:hover,
.dark .nav-holder nav a[href*="discovery"]:hover {
    background-color: #1C3A3C;
}
.benchmark-main-subtitle{
    color: var(--color-primary-link);
    overflow: hidden;
    padding-top: 120px;
}
.dark .benchmark-main-subtitle{
    color: var(--color-primary-link-dark);
}
.benchmark-title{
    color: var(--color-primary-link);
    margin-top: 50px;
        font-size: 20px;
}
.dark .benchmark-title{
    color: var(--color-primary-accent);
}
.benchmark-description {
    margin: 20px 0;
    max-width: 800px;
}

.dark #main-header h2 {
    color: var(--color-primary-accent); 
}
#main-header h2 {
    color: var(--color-primary-link);
}

/* --- New HTML-Based Tooltip Styles --- */
.tooltip-icon-legend {
    position: relative;
    cursor: help;
    display: inline-block;
}

/* The HTML pop-up card tooltips.*/
.tooltip-card {
    /* Hiding mechanism */
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s;
    pointer-events: none;
    /* Card appearance */
    position: fixed;
    z-index: 1000;
    background-color: var(--color-background-dark);
    color: var(--color-text-light);
    border-radius: 12px;
    padding: 15px;
    width: max-content;
    max-width: 400px;
    text-align: left;
}
.tooltip-card.visible {
    opacity: 1;
    visibility: visible;
} 
.tooltip-card h3 {
    font-size: 18px; 
    color: #fff; 
    margin-top: 0; 
    margin-bottom: 12px;
}
.tooltip-card .tooltip-description {
    margin-bottom: 20px; 
    line-height: 1.3;
}
.tooltip-card .tooltip-items-container {
    display: flex; 
    flex-direction: column; 
    gap: 10px;
}
.tooltip-card .tooltip-legend-item {
    display: flex; 
    align-items: 
    flex-start; 
    gap: 10px;
}
.tooltip-card .tooltip-legend-item img {
    width: 20px; 
    height: 20px; 
    margin-top: 2px;
}
.tooltip-card .tooltip-legend-item div {
    display: flex; 
    flex-direction: column;
}
.tooltip-card .tooltip-legend-item strong {
    font-weight: 600; 
    color: #fff;
}
.tooltip-card .tooltip-legend-item span {
    font-size: 13px; 
    line-height: 1.3;
}
.tooltip-sub-list {
    list-style-type: '• '; 
    padding-left: 18px;         
    font-size: 13px;
    line-height: 1.3;  
    display: flex;
    flex-direction: column;   
} 
.table-legend-item {  
    display: flex; 
    align-items: center; 
    white-space: nowrap; 
    margin-top: 8px; 
    flex-wrap: wrap;
}

/* About Page CSS */
#about-page-content-wrapper {
    margin-left: auto;
    margin-right: auto;
    max-width: 800px; 
    padding: 0 24px;
    display: flex;
    flex-direction: column; 
    gap: 40px; 
    margin-top: 40px;
    opacity: 85%; 
    margin-bottom: 60px;
}
.link-buttons-container {
    display: flex;
    flex-wrap: wrap; /* Allows buttons to stack on very narrow screens */
    gap: 16px;     
    margin-top: 16px;
}
.link-button {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-grow: 1; 
    background-color: var(--color-neutral-800);
    padding: 16px 20px;
    font-weight: 600;
    border-radius: 12px;
    text-decoration: none; 
    transition: background-color 0.2s ease-in-out;
    color: var(--color-text-light);
}
.link-button:hover {
    background-color: var(--color-neutral-700);
}
.external-link-icon {
    font-size: 20px;
    line-height: 1;
    margin-left: 12px;
}

#leaderboard-accordion table {
    width: auto !important;
    margin-right: auto !important;
}
.info-list {
    padding-left: 20px;
}

/* Smooth scrolling for the entire page */
html {
    scroll-behavior: smooth;
}
/* Home Page Styling */
.diagram-placeholder {
    width: 100%;
    height: 100%; 
    min-height: 250px; 
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: var(--color-neutral-100);
    color: var(--color-primary-accent);
    border-radius: 8px;
    font-size: 14px;
    text-align: center;
}
/* 2. Responsive behavior for smaller screens */
@media (max-width: 900px) {
    #intro-row {
        flex-direction: column;
    }
}
/* Plot legend styles */
.plot-legend-container {
    min-height: 572px;
    background-color: #fff;
    padding: 24px 32px;
    border: 1px solid var(--color-neutral-300);
    border-radius: 4px;
}

.dark .plot-legend-container {
    background: rgba(247, 248, 251, 0.1);
    border-color: var(--color-neutral-700);
}

#plot-legend-logo {
    margin-bottom: 24px;
}

#plot-legend-logo img {
    height: 19px;
}

.plot-legend-category-heading {
    font-size: 16px;
    font-weight: 700;    
}

.plot-legend-item {
    display: flex;      
    margin-top: 8px;
}


.plot-legend-item-text .description {
    color: #888;
    font-size: 12px;
}

.plot-legend-item-svg {
    margin-top: 3px;
    width: 14px;
    height: 14px;
    margin-right: 8px;
}

.plot-legend-tooling-svg {
    height: 16px;
    width: 16px;
    margin-top: 2px;
}

#plot-legend-item-pareto-svg {
    width: 18px;
    height: 18px;
    margin-right: 2px;
}
h3 .header-link-icon {
    font-size: 12px;
    vertical-align: text-top;
    margin-left: 6px;
    text-decoration: none;
}

/* Targets all "overall stats" columns in the main leaderboard for each category */
#main-leaderboard td:nth-child(6) .prose,
#main-leaderboard td:nth-child(7) .prose {
    font-weight: 700 !important;
}

/* ====== Winners by Category Section ====== */
.winners-by-category-container {
    margin: 20px 0;
    overflow-x: auto;
}

.winners-unified-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: #fff;
    border: 1px solid var(--color-neutral-300);
    border-radius: 12px;
    overflow: hidden;
}

.dark .winners-unified-table {
    background: rgba(247, 248, 251, 0.05);
    border-color: var(--color-neutral-700);
}

.winners-unified-table thead tr {
    background: linear-gradient(to right, rgba(255, 225, 101, 0.15), rgba(220, 194, 87, 0.25));
}

.dark .winners-unified-table thead tr {
    background: linear-gradient(to right, rgba(255, 225, 101, 0.2), rgba(220, 194, 87, 0.3));
}

.winners-unified-table .category-header {
    padding: 12px 8px;
    text-align: center;
    font-weight: 700;
    font-size: 13px;
    color: var(--color-text-dark);
    border-bottom: 2px solid var(--color-primary-accent) !important;
    border-right: 2px solid #999 !important;
    white-space: nowrap;
}

.dark .winners-unified-table .category-header {
    color: #fff;
    border-bottom-color: var(--color-primary-accent) !important;
    border-right-color: var(--color-neutral-500) !important;
}

.winners-unified-table .category-header:last-child {
    border-right: none !important;
}

.winner-category-icon-small {
    width: 18px;
    height: 18px;
    vertical-align: middle;
    margin-right: 6px;
}

.winners-unified-table td {
    padding: 8px 6px;
    vertical-align: middle;
    border-bottom: 1px solid #eee;
}

.dark .winners-unified-table td {
    border-bottom-color: #2a3a3a;
}

.winners-unified-table tbody tr:last-child td {
    border-bottom: none;
}

.winners-unified-table tbody tr:hover {
    background: rgba(255, 225, 101, 0.1);
}

.dark .winners-unified-table tbody tr:hover {
    background: rgba(255, 225, 101, 0.15);
}

.winners-unified-table .score-cell {
    text-align: left;
    font-weight: 700;
    color: var(--color-primary-dark);
    padding-left: 12px;
    min-width: 50px;
    border-right: 1px solid #eee;
}

.dark .winners-unified-table .score-cell {
    color: var(--color-primary-accent);
    border-right-color: var(--color-neutral-700);
}

.winners-unified-table .model-cell {
    white-space: nowrap;
    color: var(--color-text-dark);
    font-weight: 500;
    padding-right: 12px;
    border-right: 2px solid #999;
}

.dark .winners-unified-table .model-cell {
    color: #fff;
    border-right-color: var(--color-neutral-500);
}

.winners-unified-table td:nth-last-child(1) {
    border-right: none;
}

.company-logo-tiny {
    width: 16px;
    height: 16px;
    vertical-align: middle;
    margin-right: 6px;
}

/* Responsive adjustments for winners section */
@media (max-width: 900px) {
    .winners-unified-table {
        font-size: 11px;
    }
    
    .winners-unified-table .category-header {
        font-size: 11px;
        padding: 8px 4px;
    }
    
    .winner-category-icon-small {
        width: 14px;
        height: 14px;
        margin-right: 4px;
    }
    
    .winners-unified-table td {
        padding: 6px 4px;
    }
    
    .company-logo-tiny {
        width: 14px;
        height: 14px;
        margin-right: 4px;
    }
    
    .model-name {
        font-size: 10px;
    }
}

/* ====== Dark Mode Plotly Charts ====== */
/* Invert chart background for dark mode, but leave logos untouched */
.dark .js-plotly-plot {
    filter: invert(0.88) hue-rotate(180deg);
}

/* Re-invert the modebar so icons display correctly, with transparent background */
.dark .js-plotly-plot .modebar {
    filter: invert(1) hue-rotate(180deg);
    background: transparent !important;
}

.dark .js-plotly-plot .modebar-container {
    background: transparent !important;
}

.dark .js-plotly-plot .modebar-group {
    background: transparent !important;
}

.dark .js-plotly-plot .modebar-btn {
    background: transparent !important;
}

/* Style modebar icons to match the light gray text color (after re-inversion) */
/* Since we apply invert(1), we need to use the inverted color to get the desired result */
/* Target: #82889B (neutral-400). After invert(1) hue-rotate(180deg), use #7d7764 */
.dark .js-plotly-plot .modebar-btn path {
    fill: #7d7764 !important;
}

.dark .js-plotly-plot .modebar-btn:hover path {
    fill: #4a4533 !important;
}

/* ====== Mobile Responsive Styles ====== */
/* Reduce left/right padding and margins on mobile devices */
@media (max-width: 768px) {
    /* Reduce padding on main content wrapper */
    #page-content-wrapper {
        padding-left: 8px !important;
        padding-right: 8px !important;
    }
    
    /* Reduce padding on intro paragraph */
    #intro-paragraph {
        padding-left: 8px !important;
        padding-right: 8px !important;
        max-width: 100% !important;
    }
    
    /* Reduce padding on category intro */
    #intro-category-paragraph {
        padding-left: 8px !important;
        padding-right: 8px !important;
        max-width: 100% !important;
    }
    
    /* Reduce navigation holder padding */
    .nav-holder {
        padding-left: 8px !important;
        padding-right: 8px !important;
    }
    
    /* Reduce about content padding */
    #about-content {
        padding-left: 8px !important;
        padding-right: 8px !important;
        max-width: 100% !important;
    }
    
    /* Reduce about page wrapper padding */
    #about-page-content-wrapper {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }
    
    /* Reduce gradio container padding */
    .gradio-container {
        padding-left: 8px !important;
        padding-right: 8px !important;
    }
    
    /* Reduce main block padding */
    .main, .wrap, .contain {
        padding-left: 4px !important;
        padding-right: 4px !important;
    }
    
    /* Make plots and tables full width on mobile */
    .plot-legend-container {
        padding: 16px 12px !important;
    }
    
    /* Reduce table cell padding on mobile */
    table.gr-table th, table.gr-table td {
        padding: 4px 2px !important;
    }
}
"""
