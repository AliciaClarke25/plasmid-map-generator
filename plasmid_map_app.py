#!/usr/bin/env python3
"""
Plasmid Map Generator - Web App
A simple web interface for generating plasmid maps
No installation needed for users - just visit the URL!
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib import colors as mcolors
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Plasmid Map Generator",
    page_icon="🧬",
    layout="wide"
)

# R color name mapping
R_COLOR_MAP = {
    'darkorchid4': 'darkorchid', 'darkorchid3': 'darkorchid',
    'darkorchid2': 'darkorchid', 'darkorchid1': 'orchid',
    'darkturquoise': 'darkturquoise', 'brown2': 'brown',
    'brown1': 'brown', 'brown3': 'brown', 'brown4': 'brown',
    'gold1': 'gold', 'gold2': 'gold', 'gold3': 'goldenrod',
    'gold4': 'goldenrod', 'aquamarine1': 'aquamarine',
    'aquamarine2': 'aquamarine', 'aquamarine3': 'mediumaquamarine',
    'aquamarine4': 'mediumaquamarine',
}

def convert_r_color(color_name):
    """Convert R color names to matplotlib-compatible colors"""
    color_lower = color_name.lower().strip()
    if color_lower in R_COLOR_MAP:
        return R_COLOR_MAP[color_lower]
    if mcolors.is_color_like(color_name):
        return color_name
    base_color = ''.join([c for c in color_lower if not c.isdigit()])
    if mcolors.is_color_like(base_color):
        return base_color
    return 'black'

def create_plasmid_map(data, font_size=11):
    """Create plasmid map from DataFrame"""
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Clean data
    data['box_position'] = data['box_position'].str.strip().str.lower()
    data['arrow_end_type'] = data['arrow_end_type'].str.strip().str.lower()
    data['colour'] = data['colour'].str.strip()
    
    # Calculate plasmid extent
    plasmid_start = min(data['start'].min(), data['end'].min()) - 50
    plasmid_end = max(data['start'].max(), data['end'].max()) + 50
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Set font
    plt.rcParams['font.size'] = font_size
    
    # Draw main plasmid line
    ax.plot([plasmid_start, plasmid_end], [0, 0], 'k-', linewidth=3, zorder=1)
    
    # Parameters
    box_height = 80
    text_distance = 250
    
    # Draw each element
    for idx, row in data.iterrows():
        # Handle point elements
        element_start = row['start']
        element_end = row['end']
        if element_start == element_end:
            min_width = 50
            element_start = element_start - min_width / 2
            element_end = element_end + min_width / 2
        
        midpoint = (element_start + element_end) / 2
        
        # Calculate positions
        if row['box_position'] == 'up':
            box_y = box_height / 2
            text_y = box_height + text_distance
            arrow_start_y = box_height
            arrow_end_y = text_y - 20
        else:
            box_y = -box_height / 2
            text_y = -box_height - text_distance
            arrow_start_y = -box_height
            arrow_end_y = text_y + 20
        
        # Draw box
        converted_color = convert_r_color(row['colour'])
        rect = patches.Rectangle(
            (element_start, box_y - box_height/2),
            element_end - element_start,
            box_height,
            linewidth=1.5,
            edgecolor='black',
            facecolor=converted_color,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Draw arrow
        if row['arrow_end_type'] == 'arrow':
            arrow = FancyArrowPatch(
                (midpoint, arrow_start_y),
                (midpoint, arrow_end_y),
                arrowstyle='-|>',
                mutation_scale=20,
                linewidth=1.5,
                color='black',
                zorder=1
            )
            ax.add_patch(arrow)
        else:
            ax.plot([midpoint, midpoint], [arrow_start_y, arrow_end_y],
                   'k-', linewidth=1.5, zorder=1)
        
        # Add label
        ax.text(midpoint, text_y, row['element'],
               ha='center', va='center', fontsize=font_size, zorder=3)
    
    # Set axis properties
    y_margin = 400
    ax.set_xlim(plasmid_start - 100, plasmid_end + 100)
    ax.set_ylim(-y_margin, y_margin)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def fig_to_download_link(fig, filename, format='pdf'):
    """Convert matplotlib figure to download link"""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    
    mime_types = {'pdf': 'application/pdf', 'svg': 'image/svg+xml', 'png': 'image/png'}
    mime = mime_types.get(format, 'application/octet-stream')
    
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}.{format}">Download {format.upper()}</a>'
    return href

# Main app
st.title("🧬 Plasmid Map Generator")
st.markdown("Create beautiful plasmid maps with no installation required!")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    font_size = st.slider("Label Font Size", min_value=8, max_value=20, value=11, step=1)
    st.markdown("---")
    st.header("📖 Quick Guide")
    st.markdown("""
    **Required Columns:**
    - Element
    - Start
    - End
    - Box position (Up/Down)
    - Colour
    - Arrow end type (arrow/flat)
    
    **Popular Colors:**
    - blue, green, red, gold
    - purple, orange, brown
    - darkturquoise, forestgreen
    """)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📊 Upload Data", "✏️ Manual Entry", "ℹ️ Help"])

with tab1:
    st.header("Upload CSV or Excel File")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with your plasmid data"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("✅ File uploaded successfully!")
            
            # Show data
            st.subheader("Your Data")
            st.dataframe(df, use_container_width=True)
            
            # Generate button
            if st.button("🎨 Generate Plasmid Map", type="primary", key="upload_generate"):
                with st.spinner("Creating your plasmid map..."):
                    try:
                        fig = create_plasmid_map(df, font_size=font_size)
                        st.pyplot(fig)
                        
                        # Download buttons
                        st.subheader("📥 Download Your Map")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(fig_to_download_link(fig, "plasmid_map", "pdf"), 
                                      unsafe_allow_html=True)
                        with col2:
                            st.markdown(fig_to_download_link(fig, "plasmid_map", "svg"), 
                                      unsafe_allow_html=True)
                        with col3:
                            st.markdown(fig_to_download_link(fig, "plasmid_map", "png"), 
                                      unsafe_allow_html=True)
                        
                        plt.close(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("Please check your data format matches the required columns.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("👆 Upload a file to get started, or try the Manual Entry tab")
        
        # Show example
        st.subheader("📋 Example Data Format")
        example_df = pd.DataFrame({
            'Element': ['Promoter', 'GeneX', 'Resistance'],
            'Start': [100, 400, 1300],
            'End': [300, 1200, 2100],
            'Box position': ['Up', 'Down', 'Up'],
            'Colour': ['dodgerblue', 'forestgreen', 'brown'],
            'Arrow end type': ['arrow', 'arrow', 'flat']
        })
        st.dataframe(example_df, use_container_width=True)

with tab2:
    st.header("Manual Data Entry")
    st.markdown("Enter your plasmid elements one by one")
    
    # Initialize session state for manual entries
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = []
    
    # Input form
    with st.form("element_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            element = st.text_input("Element Name", value="Gene1")
            start = st.number_input("Start Position (bp)", min_value=0, value=100, step=10)
            end = st.number_input("End Position (bp)", min_value=0, value=500, step=10)
        
        with col2:
            box_pos = st.selectbox("Box Position", ["Up", "Down"])
            colour = st.selectbox("Colour", [
                'blue', 'green', 'red', 'purple', 'gold', 'orange', 'brown',
                'dodgerblue', 'forestgreen', 'darkred', 'darkorchid',
                'darkturquoise', 'aquamarine', 'coral'
            ])
            arrow_type = st.selectbox("Arrow Type", ["arrow", "flat"])
        
        col_add, col_clear = st.columns([1, 1])
        with col_add:
            add_button = st.form_submit_button("➕ Add Element", type="primary")
        with col_clear:
            clear_button = st.form_submit_button("🗑️ Clear All")
    
    if add_button:
        st.session_state.manual_data.append({
            'Element': element,
            'Start': start,
            'End': end,
            'Box position': box_pos,
            'Colour': colour,
            'Arrow end type': arrow_type
        })
        st.success(f"✅ Added {element}")
    
    if clear_button:
        st.session_state.manual_data = []
        st.info("🗑️ Cleared all elements")
    
    # Show current data
    if st.session_state.manual_data:
        st.subheader("Current Elements")
        manual_df = pd.DataFrame(st.session_state.manual_data)
        st.dataframe(manual_df, use_container_width=True)
        
        # Generate button
        if st.button("🎨 Generate Plasmid Map", type="primary", key="manual_generate"):
            with st.spinner("Creating your plasmid map..."):
                try:
                    fig = create_plasmid_map(manual_df, font_size=font_size)
                    st.pyplot(fig)
                    
                    # Download buttons
                    st.subheader("📥 Download Your Map")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(fig_to_download_link(fig, "plasmid_map", "pdf"), 
                                  unsafe_allow_html=True)
                    with col2:
                        st.markdown(fig_to_download_link(fig, "plasmid_map", "svg"), 
                                  unsafe_allow_html=True)
                    with col3:
                        st.markdown(fig_to_download_link(fig, "plasmid_map", "png"), 
                                  unsafe_allow_html=True)
                    
                    plt.close(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Add elements using the form above")

with tab3:
    st.header("📖 How to Use")
    
    st.markdown("""
    ### Option 1: Upload File
    1. Prepare a CSV or Excel file with your data
    2. Upload it in the "Upload Data" tab
    3. Click "Generate Plasmid Map"
    4. Download PDF, SVG, or PNG
    
    ### Option 2: Manual Entry
    1. Go to "Manual Entry" tab
    2. Fill in the form for each element
    3. Click "Add Element" for each one
    4. Click "Generate Plasmid Map"
    
    ### Required Columns
    
    | Column | Description | Example |
    |--------|-------------|---------|
    | Element | Name of genetic element | "Promoter", "GeneX" |
    | Start | Start position in bp | 100, 1500 |
    | End | End position in bp | 300, 2000 |
    | Box position | "Up" or "Down" | Up, Down |
    | Colour | Color name | blue, green, red |
    | Arrow end type | "arrow" or "flat" | arrow, flat |
    
    ### Popular Colors
    
    **For Promoters:** dodgerblue, steelblue, royalblue  
    **For Genes:** green, forestgreen, limegreen  
    **For Resistance:** red, brown, firebrick  
    **For Tags:** purple, darkorchid, plum  
    **For Reporters:** gold, yellow  
    **For Origins:** orange, coral  
    
    ### Tips
    
    - **Point elements:** If Start = End, a minimum 50bp width is used
    - **Overlapping:** Alternate Up/Down positions to avoid overlap
    - **Font size:** Adjust in the sidebar on the left
    - **File formats:** PDF for publications, SVG for editing, PNG for presentations
    """)
    
    st.markdown("---")
    st.markdown("### 🎨 Example")
    
    example_code = """
Element,Start,End,Box position,Colour,Arrow end type
Promoter,100,300,Up,dodgerblue,arrow
GeneX,400,1200,Down,forestgreen,arrow
Resistance,1300,2100,Up,brown,flat
"""
    st.code(example_code, language='csv')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧬 Plasmid Map Generator | Made for researchers by researchers</p>
    <p><small>No installation required • Works in any browser • Free to use</small></p>
</div>
""", unsafe_allow_html=True)
