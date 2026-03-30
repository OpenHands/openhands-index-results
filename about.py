import gradio as gr


def build_page():
    with gr.Column(elem_id="about-page-content-wrapper"):
        # --- Section 1: About ---
        gr.HTML(
            """
            <h2>About</h2>
            <p>
                OpenHands Index tracks AI coding agent performance across software engineering benchmarks, providing a unified view of both accuracy and cost efficiency.
            </p>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 2: Benchmark Details ---
        gr.HTML(
            """
            <h2>Benchmark Details</h2>
            <p>We evaluate agents across five categories:</p>
            <ul class="info-list">
                <li><strong>Issue Resolution:</strong> <a href="https://www.swebench.com/" target="_blank">SWE-bench Verified</a> — 500 instances</li>
                <li><strong>Frontend:</strong> <a href="https://github.com/OpenHands/SWE-bench-multimodal" target="_blank">SWE-bench Multimodal</a> — 617 instances</li>
                <li><strong>Greenfield:</strong> <a href="https://github.com/commit-0/commit0" target="_blank">Commit0</a> — 16 libraries (lite split)</li>
                <li><strong>Testing:</strong> <a href="https://github.com/logic-star-ai/swt-bench" target="_blank">SWT-bench Verified</a> — 433 instances</li>
                <li><strong>Information Gathering:</strong> <a href="https://huggingface.co/gaia-benchmark" target="_blank">GAIA</a> — 165 questions (validation split)</li>
            </ul>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 3: Methodology ---
        gr.HTML(
            """
            <h2>Methodology</h2>
            <p><strong>Per-benchmark scores:</strong> Each benchmark reports a percentage metric (resolve rate, accuracy, or test pass rate), making scores comparable regardless of dataset size.</p>
            <p><strong>Average score:</strong> Macro-average across all five categories with equal weighting.</p>
            <p><strong>Cost &amp; Runtime:</strong> Average USD and seconds per task instance.</p>
            <p>All evaluations use the <a href="https://github.com/OpenHands/software-agent-sdk" target="_blank">OpenHands Agent SDK</a> with identical configurations per model.</p>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 4: API Access ---
        gr.HTML(
            """
            <h2>API Access</h2>
            <p>Access leaderboard data programmatically via our REST API:</p>
            <ul class="info-list">
                <li><a href="https://index.openhands.dev/api/docs" target="_blank">Interactive API Documentation</a> - Swagger UI with all endpoints</li>
                <li><a href="https://index.openhands.dev/api/leaderboard" target="_blank">/api/leaderboard</a> - Full leaderboard with scores and metadata</li>
                <li><a href="https://index.openhands.dev/api/categories" target="_blank">/api/categories</a> - List of benchmark categories</li>
            </ul>
            <p style="margin-top: 10px;"><strong>Example:</strong></p>
            <pre class="citation-block" style="font-size: 0.9em;">curl "https://index.openhands.dev/api/leaderboard?limit=5"</pre>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 5: Resources ---
        gr.HTML(
            """
            <h2>Resources</h2>
            <ul class="info-list">
                <li><a href="https://github.com/OpenHands/OpenHands" target="_blank">OpenHands</a> - The main OpenHands repository</li>
                <li><a href="https://github.com/OpenHands/software-agent-sdk" target="_blank">Software Agent SDK</a> - The agent code used for evaluation</li>
                <li><a href="https://github.com/OpenHands/benchmarks" target="_blank">Benchmarks</a> - The benchmarking code</li>
                <li><a href="https://github.com/OpenHands/openhands-index-results" target="_blank">Results</a> - Raw evaluation results</li>
            </ul>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 5: Contact ---
        gr.HTML(
            """
            <h2>Contact</h2>
            <p>
                Questions or feedback? Join us on <a href="https://dub.sh/openhands" target="_blank">Slack</a>.
            </p>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 6: Acknowledgements ---
        gr.HTML(
            """
            <h2>Acknowledgements</h2>
            <p>
                The leaderboard interface is adapted from the 
                <a href="https://huggingface.co/spaces/allenai/asta-bench-leaderboard" target="_blank">AstaBench Leaderboard</a> 
                by Allen Institute for AI.
            </p>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 7: Citation ---
        gr.HTML(
            """
            <h2>Citation</h2>
            <pre class="citation-block">
@misc{openhandsindex2025,
    title={OpenHands Index: A Comprehensive Leaderboard for AI Coding Agents},
    author={OpenHands Team},
    year={2025},
    howpublished={https://index.openhands.dev}
}</pre>
            """
        )
