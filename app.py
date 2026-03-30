# app.py
import logging
import sys
import os

from constants import FONT_FAMILY_SHORT

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Force rebuild to fetch latest data from GitHub repo
logger.info("Starting OpenHands Index application")

# Setup mock data before anything else
try:
    from setup_data import setup_mock_data, start_background_refresh, CACHE_TTL_SECONDS
    setup_mock_data()
    logger.info("Data setup completed successfully")
    
    # Start background refresh scheduler (checks for new data every hour)
    start_background_refresh()
    logger.info(f"Background refresh scheduler started (interval: {CACHE_TTL_SECONDS}s)")
except Exception as e:
    logger.error(f"Error during data setup: {e}", exc_info=True)
    logger.warning("Continuing with app startup despite error")

import gradio as gr
import urllib.parse
from huggingface_hub import HfApi
from config import LEADERBOARD_PATH, LOCAL_DEBUG
from content import css
from main_page import build_page as build_main_page
from bug_fixing import build_page as build_bug_fixing_page
from app_creation import build_page as build_app_creation_page
from frontend_development import build_page as build_frontend_page
from test_generation import build_page as build_test_generation_page
from information_gathering import build_page as build_information_gathering_page
from about import build_page as build_about_page

logger.info(f"All modules imported (LOCAL_DEBUG={LOCAL_DEBUG})")

api = HfApi()
LOGO_PATH = "assets/logo.svg"

# PostHog analytics (client-side)
POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY", "phc_ERBPfEE0gwNgkOBsxbHr1wh9mBsYcsw4zSLtvdA9RFg")
posthog_script = f"""
<script>
    !function(t,e){{var o,n,p,r;e.__SV||(window.posthog && window.posthog.__loaded)||(window.posthog=e,e._i=[],e.init=function(i,s,a){{function g(t,e){{var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){{t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){{var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e}},u.people.toString=function(){{return u.toString(1)+".people (stub)"}},o="init ss us bi os hs es ns capture Bi calculateEventProperties cs register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSurveysLoaded onSessionId getSurveys getActiveMatchingSurveys renderSurvey displaySurvey cancelPendingSurvey canRenderSurvey canRenderSurveyAsync identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException startExceptionAutocapture stopExceptionAutocapture loadToolbar get_property getSessionProperty ps vs createPersonProfile gs Zr ys opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing get_explicit_consent_status is_capturing clear_opt_in_out_capturing ds debug O fs getPageViewId captureTraceFeedback captureTraceMetric Yr".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])}},e.__SV=1)}}(document,window.posthog||[]);
    posthog.init('{POSTHOG_API_KEY}', {{
        api_host: 'https://us.i.posthog.com',
        defaults: '2025-11-30',
        person_profiles: 'identified_only',
    }})
</script>
"""

# JavaScripts
scroll_script = """
<script>
function scroll_to_element(id) {
    console.log("Global scroll_to_element called for ID:", id);
    const element = document.querySelector('#' + id);
    if (element) {
        console.log("Element found:", element);
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
        console.error("Error: Element with ID '" + id + "' not found in the document.");
    }
}
</script>
"""
redirect_script = """
<script>
    if (window.location.pathname === '/') { window.location.replace('/home'); }
</script>
"""

# JavaScript to fix navigation links to use relative paths (avoids domain mismatch when behind proxy)
fix_nav_links_script = """
<script>
(function() {
    function fixNavLinks() {
        // Find all navigation links in the nav-holder
        const navLinks = document.querySelectorAll('.nav-holder nav a');
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href) {
                // Extract the pathname from the href (works with both relative and absolute URLs)
                try {
                    const url = new URL(href, window.location.origin);
                    // Only update if the pathname starts with /
                    if (url.pathname.startsWith('/')) {
                        link.setAttribute('href', url.pathname);
                    }
                } catch (e) {
                    // If URL parsing fails, leave the href as-is
                }
            }
            // Remove target="_blank" from nav links to prevent opening in new tab
            link.removeAttribute('target');
        });
    }
    
    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fixNavLinks);
    } else {
        fixNavLinks();
    }
    
    // Also run periodically to catch dynamically added links
    setInterval(fixNavLinks, 1000);
})();
</script>
"""
tooltip_script = """
<script>
function initializeSmartTooltips() {
    // Find all tooltip trigger icons
    const tooltipIcons = document.querySelectorAll('.tooltip-icon-legend');

    tooltipIcons.forEach(icon => {
        // Find the tooltip card associated with this icon
        const tooltipCard = icon.querySelector('.tooltip-card');
        if (!tooltipCard) return;

        // Move the card to the end of the <body>. This is the KEY to escaping
        // any parent containers that might clip it.
        document.body.appendChild(tooltipCard);

        // --- MOUSE HOVER EVENT ---
        icon.addEventListener('mouseenter', () => {
            // Get the exact position of the icon on the screen
            const iconRect = icon.getBoundingClientRect();
            // Get the dimensions of the tooltip card
            const cardRect = tooltipCard.getBoundingClientRect();

            // Calculate the ideal top position (above the icon with a 10px gap)
            const top = iconRect.top - cardRect.height - 10;
            
            // --- Smart Centering Logic ---
            // Start by calculating the perfect center
            let left = iconRect.left + (iconRect.width / 2) - (cardRect.width / 2);

            // Check if it's going off the left edge of the screen
            if (left < 10) {
                left = 10; // Pin it to the left with a 10px margin
            }
            // Check if it's going off the right edge of the screen
            if (left + cardRect.width > window.innerWidth) {
                left = window.innerWidth - cardRect.width - 10; // Pin it to the right
            }

            // Apply the calculated position and show the card
            tooltipCard.style.top = `${top}px`;
            tooltipCard.style.left = `${left}px`;
            tooltipCard.classList.add('visible');
        });

        // --- MOUSE LEAVE EVENT ---
        icon.addEventListener('mouseleave', () => {
            // Hide the card
            tooltipCard.classList.remove('visible');
        });
    });
}

// Poll the page until the tooltips exist, then run the initialization.
const tooltipInterval = setInterval(() => {
    if (document.querySelector('.tooltip-icon-legend')) {
        clearInterval(tooltipInterval);
        initializeSmartTooltips();
    }
}, 200);
</script>
"""

# JavaScript to handle dark mode for Plotly charts and OpenHands logos
dark_mode_script = """
<script>
function updateChartsForDarkMode() {
    const isDark = document.body.classList.contains('dark');
    
    // Update Plotly chart backgrounds
    const plots = document.querySelectorAll('.js-plotly-plot');
    plots.forEach(plot => {
        if (plot._fullLayout) {
            Plotly.relayout(plot, {
                'paper_bgcolor': isDark ? '#1f1f1f' : 'white',
                'plot_bgcolor': isDark ? '#1f1f1f' : 'white',
                'font.color': isDark ? '#e0e0e0' : '#333',
                'xaxis.gridcolor': isDark ? '#444' : '#eee',
                'yaxis.gridcolor': isDark ? '#444' : '#eee'
            });
        }
    });
    
    // Swap OpenHands logos based on theme
    const images = document.querySelectorAll('.js-plotly-plot image');
    images.forEach(img => {
        const href = img.getAttribute('href') || img.getAttribute('xlink:href') || '';
        if (href.includes('openhands=lightlogo')) {
            img.style.display = isDark ? 'none' : '';
        } else if (href.includes('openhands=darklogo')) {
            img.style.display = isDark ? '' : 'none';
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(updateChartsForDarkMode, 500);
    
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class') {
                updateChartsForDarkMode();
            }
        });
    });
    observer.observe(document.body, { attributes: true });
    
    setInterval(updateChartsForDarkMode, 1000);
});
</script>
"""
# --- Theme Definition ---
# Color scheme aligned with OpenHands brand (from openhands-ui/tokens.css)
# Primary: Yellow (#FFE165), Neutral: Grey scale, Accents: Green (#BCFF8C), Red (#FF684E)
theme = gr.themes.Base(
    # Primary hue - Yellow (OpenHands brand color)
    primary_hue=gr.themes.Color(
        c50="#FFFCF0", c100="#FFF3C0", c200="#FFEEAA", c300="#FFEA92", c400="#FFE57B",
        c500="#FFE165", c600="#DCC257", c700="#BBA54A", c800="#99873D", c900="#76682F", c950="#534921"
    ),
    # Secondary hue - Green accent (from OpenHands palette)
    secondary_hue=gr.themes.Color(
        c50="#F8FFF4", c100="#E4FFD0", c200="#DAFFBF", c300="#CFFFAD", c400="#C6FF9D",
        c500="#BCFF8C", c600="#A2DC79", c700="#8ABB67", c800="#719954", c900="#577641", c950="#3D532E"
    ),
    # Neutral hue - Grey scale (OpenHands dark mode colors)
    neutral_hue=gr.themes.Color(
        c50="#F7F8FB", c100="#EBEDF3", c200="#D4D8E7", c300="#B1B9D3", c400="#82889B",
        c500="#525662", c600="#3A3C45", c700="#2F3137", c800="#222328", c900="#18191C", c950="#0D0D0F"
    ),
    font=[FONT_FAMILY_SHORT, 'sans-serif'],
    font_mono=['monospace'],
).set(
    body_text_color='*neutral_950',
    body_text_color_subdued='*neutral_700',
    body_text_color_subdued_dark='*neutral_300',
    body_text_color_dark='*neutral_50',
    background_fill_primary='*neutral_50',
    background_fill_primary_dark='*neutral_900',
    background_fill_secondary='*neutral_100',
    background_fill_secondary_dark='*neutral_800',
    border_color_accent='*primary_500',
    border_color_accent_subdued='*neutral_300',
    border_color_accent_subdued_dark='*neutral_600',
    color_accent='*primary_500',
    color_accent_soft='*neutral_200',
    color_accent_soft_dark='*neutral_800',
    link_text_color='*neutral_700',
    link_text_color_dark='*neutral_300',
    link_text_color_active_dark='*primary_500',
    link_text_color_hover_dark='*primary_400',
    link_text_color_visited_dark='*neutral_400',
    table_even_background_fill='*neutral_100',
    table_even_background_fill_dark='*neutral_800',
    button_primary_background_fill='*primary_500',
    button_primary_background_fill_dark='*primary_500',
    button_primary_background_fill_hover='*primary_400',
    button_primary_background_fill_hover_dark='*primary_400',
    button_secondary_background_fill='*secondary_500',
    button_secondary_background_fill_dark='*secondary_600',
    button_secondary_text_color='*neutral_900',
    button_secondary_text_color_dark='*neutral_900',
    block_title_text_color='*neutral_900',
    button_primary_text_color='*neutral_900',
    block_title_text_color_dark='*neutral_50',
    button_primary_text_color_dark='*neutral_900',
    block_border_color='*neutral_300',
    block_border_color_dark='*neutral_700',
    block_background_fill_dark='*neutral_900',
    block_background_fill='*neutral_50',
    checkbox_label_text_color='*neutral_900',
    checkbox_label_background_fill='*neutral_200',
    checkbox_label_background_fill_dark='*neutral_700',
    checkbox_background_color_selected='*primary_500',
    checkbox_background_color_selected_dark='*primary_500',
)
try:
    with open(LOGO_PATH, "r") as f:
        svg_content = f.read()
    encoded_svg = urllib.parse.quote(svg_content)
    home_icon_data_uri = f"data:image/svg+xml,{encoded_svg}"
except FileNotFoundError:
    logger.warning(f"Home icon file not found at {LOGO_PATH}")
    home_icon_data_uri = "none"

# Load dark mode logo (PNG)
LOGO_DARK_PATH = "assets/logo-dark.png"
try:
    import base64
    with open(LOGO_DARK_PATH, "rb") as f:
        dark_logo_content = f.read()
    encoded_dark_logo = base64.b64encode(dark_logo_content).decode('utf-8')
    home_icon_dark_data_uri = f"data:image/png;base64,{encoded_dark_logo}"
except FileNotFoundError:
    logger.warning(f"Dark mode logo file not found at {LOGO_DARK_PATH}")
    home_icon_dark_data_uri = home_icon_data_uri  # Fallback to light logo

# --- This is the final CSS ---
final_css = css + f"""
/* --- Find the "Home" button and replace its text with an icon --- */
.nav-holder nav a[href$="/"] {{
    display: none !important;
}}
.nav-holder nav a[href*="/home"] {{
    grid-row: 1 !important;
    grid-column: 1 !important;
    justify-self: start !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    /* 2. Hide the original "Home" text */
    font-size: 0 !important;
    text-indent: -9999px;

    /* 3. Apply the icon as the background (light mode) */
    background-image: url("{home_icon_data_uri}") !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center !important;

    width: 240px !important;    
    height: 50px !important;   
    padding: 0 !important;
    border: none !important;
    outline: none !important;
}}

/* Dark mode logo override */
.dark .nav-holder nav a[href*="/home"] {{
    background-image: url("{home_icon_dark_data_uri}") !important;
}}
"""
# --- Gradio App Definition ---
logger.info("Creating Gradio application")
demo = gr.Blocks(
    theme=theme,
    css=final_css,
    head=posthog_script + scroll_script + redirect_script + fix_nav_links_script + tooltip_script + dark_mode_script,
    title="OpenHands Index",
)

with demo.route("Home", "/home"):
    build_main_page()

with demo.route("Issue Resolution", "/issue-resolution"):
    build_bug_fixing_page()

with demo.route("Greenfield", "/greenfield"):
    build_app_creation_page()

with demo.route("Frontend", "/frontend"):
    build_frontend_page()

with demo.route("Testing", "/testing"):
    build_test_generation_page()

with demo.route("Information Gathering", "/information-gathering"):
    build_information_gathering_page()

with demo.route("About", "/about"):
    build_about_page()

logger.info("All routes configured")

# Mount the REST API on /api
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api import api_app


class RootRedirectMiddleware(BaseHTTPMiddleware):
    """Middleware to redirect root path "/" to "/home".
    
    This fixes the 307 trailing slash redirect issue (Gradio bug #11071) that 
    occurs when Gradio is mounted at "/" - FastAPI's default behavior redirects 
    "/" to "//", which breaks routing on HuggingFace Spaces.
    
    See: https://github.com/gradio-app/gradio/issues/11071
    """
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/":
            return RedirectResponse(url="/home", status_code=302)
        return await call_next(request)


# Create a parent FastAPI app with redirect_slashes=False to prevent
# automatic trailing slash redirects that cause issues with Gradio
root_app = FastAPI(redirect_slashes=False)

# Add middleware to handle root path redirect to /home
root_app.add_middleware(RootRedirectMiddleware)

root_app.mount("/api", api_app)

# Mount Gradio app at root path
app = gr.mount_gradio_app(root_app, demo, path="/")
logger.info("REST API mounted at /api, Gradio app mounted at /")


# Launch the app
if __name__ == "__main__":
    import uvicorn
    # Respect platform port/host if provided (e.g., OpenHands runtime)
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    host = os.environ.get("HOST", os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    logger.info(f"Launching app on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
    logger.info("App launched successfully")

