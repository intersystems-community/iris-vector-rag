"""
Header Component

Renders the main header for the RAG Templates demo application.
"""

from datetime import datetime

import streamlit as st
from utils.app_config import get_config


def render_header():
    """Render the main application header."""
    config = get_config()

    # Main header with logo and title
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.markdown("🤖", help="RAG Templates Demo")

    with col2:
        st.markdown(
            f"""
        <div style="text-align: center;">
            <h1 style="margin: 0; color: #1f77b4;">{config.app_title}</h1>
            <p style="margin: 0; color: #666; font-size: 1.1rem;">
                Interactive demonstration of cutting-edge RAG pipeline implementations
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        # Status indicator
        render_system_status()


def render_system_status():
    """Render system status indicator."""
    # This would normally check actual system health
    status = "🟢 Online"
    tooltip = "All systems operational"

    st.markdown(
        f"""
    <div style="text-align: right;">
        <span title="{tooltip}" style="font-size: 0.9rem;">{status}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, description: str = "", icon: str = ""):
    """
    Render a page-specific header.

    Args:
        title: Page title
        description: Optional page description
        icon: Optional page icon
    """
    if icon:
        display_title = f"{icon} {title}"
    else:
        display_title = title

    st.markdown(
        f"""
    <div style="padding: 1rem 0 2rem 0; border-bottom: 1px solid #eee; margin-bottom: 2rem;">
        <h2 style="margin: 0; color: #1f77b4;">{display_title}</h2>
        {f'<p style="margin: 0.5rem 0 0 0; color: #666; font-size: 1.1rem;">{description}</p>' if description else ''}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_breadcrumb(path: list):
    """
    Render a breadcrumb navigation.

    Args:
        path: List of (name, url) tuples for breadcrumb items
    """
    breadcrumb_html = []

    for i, (name, url) in enumerate(path):
        if i == len(path) - 1:  # Last item (current page)
            breadcrumb_html.append(f'<span style="color: #666;">{name}</span>')
        else:
            breadcrumb_html.append(
                f'<a href="{url}" style="color: #1f77b4; text-decoration: none;">{name}</a>'
            )

        if i < len(path) - 1:
            breadcrumb_html.append(
                '<span style="color: #ccc; margin: 0 0.5rem;"> / </span>'
            )

    st.markdown(
        f"""
    <div style="margin-bottom: 1rem; font-size: 0.9rem;">
        {''.join(breadcrumb_html)}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_notification_banner(
    message: str, type: str = "info", dismissible: bool = True
):
    """
    Render a notification banner.

    Args:
        message: Notification message
        type: Type of notification (info, success, warning, error)
        dismissible: Whether the banner can be dismissed
    """

    colors = {
        "info": {"bg": "#d1ecf1", "border": "#bee5eb", "text": "#0c5460"},
        "success": {"bg": "#d4edda", "border": "#c3e6cb", "text": "#155724"},
        "warning": {"bg": "#fff3cd", "border": "#ffeaa7", "text": "#856404"},
        "error": {"bg": "#f8d7da", "border": "#f5c6cb", "text": "#721c24"},
    }

    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}

    color_scheme = colors.get(type, colors["info"])
    icon = icons.get(type, icons["info"])

    dismiss_button = ""
    if dismissible:
        dismiss_button = """
        <button onclick="this.parentElement.style.display='none'" 
                style="background: none; border: none; float: right; font-size: 1.2rem; cursor: pointer;">
            ×
        </button>
        """

    st.markdown(
        f"""
    <div style="
        background-color: {color_scheme['bg']};
        border: 1px solid {color_scheme['border']};
        color: {color_scheme['text']};
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
        position: relative;
    ">
        {dismiss_button}
        <span style="margin-right: 0.5rem;">{icon}</span>
        {message}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_progress_bar(current: int, total: int, label: str = ""):
    """
    Render a progress bar.

    Args:
        current: Current progress value
        total: Total/maximum value
        label: Optional label for the progress bar
    """
    if total == 0:
        percentage = 0
    else:
        percentage = min(100, (current / total) * 100)

    st.markdown(
        f"""
    <div style="margin: 1rem 0;">
        {f'<div style="margin-bottom: 0.5rem; font-weight: 500;">{label}</div>' if label else ''}
        <div style="
            background-color: #e9ecef;
            border-radius: 0.375rem;
            overflow: hidden;
            height: 1rem;
        ">
            <div style="
                background-color: #1f77b4;
                height: 100%;
                width: {percentage}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <div style="text-align: right; font-size: 0.9rem; color: #666; margin-top: 0.25rem;">
            {current} / {total} ({percentage:.1f}%)
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_loading_spinner(message: str = "Loading..."):
    """
    Render a loading spinner with message.

    Args:
        message: Loading message to display
    """
    st.markdown(
        f"""
    <div style="text-align: center; padding: 2rem;">
        <div style="
            border: 4px solid #f3f3f3;
            border-top: 4px solid #1f77b4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem auto;
        "></div>
        <p style="color: #666; margin: 0;">{message}</p>
    </div>
    
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_error_message(error: str, details: str = "", show_details: bool = False):
    """
    Render an error message with optional details.

    Args:
        error: Main error message
        details: Detailed error information
        show_details: Whether to show details by default
    """
    st.error(f"❌ {error}")

    if details and show_details:
        with st.expander("Error Details"):
            st.code(details)
    elif details:
        if st.button("Show Error Details"):
            st.code(details)


def render_success_message(message: str, details: str = ""):
    """
    Render a success message with optional details.

    Args:
        message: Success message
        details: Optional additional details
    """
    st.success(f"✅ {message}")

    if details:
        st.info(details)


def render_info_box(title: str, content: str, icon: str = "ℹ️"):
    """
    Render an information box.

    Args:
        title: Box title
        content: Box content
        icon: Icon to display
    """
    st.markdown(
        f"""
    <div style="
        border: 1px solid #d1ecf1;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    ">
        <div style="font-weight: 600; margin-bottom: 0.5rem; color: #495057;">
            {icon} {title}
        </div>
        <div style="color: #6c757d;">
            {content}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, delta: str = "", icon: str = ""):
    """
    Render a metric card.

    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta/change value
        icon: Optional icon
    """
    delta_html = ""
    if delta:
        delta_color = (
            "#28a745"
            if delta.startswith("+")
            else "#dc3545" if delta.startswith("-") else "#6c757d"
        )
        delta_html = f'<div style="color: {delta_color}; font-size: 0.9rem; margin-top: 0.25rem;">{delta}</div>'

    icon_html = (
        f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>'
        if icon
        else ""
    )

    st.markdown(
        f"""
    <div style="
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        text-align: center;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    ">
        {icon_html}
        <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.25rem;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: #495057;">{value}</div>
        {delta_html}
    </div>
    """,
        unsafe_allow_html=True,
    )
