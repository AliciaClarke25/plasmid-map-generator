#!/usr/bin/env python3
"""
Plasmid Map Generator - Advanced Version with GenBank Support
Supports .gb files, region selection, text orientation, position display, and more!
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib import colors as mcolors
import io
import base64
import random
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature

# Page configuration
st.set_page_config(
    page_title="Plasmid Map Generator - Advanced",
    page_icon="🧬",
    layout="wide"
)

# R color mapping
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

# Pastel colors pool for random assignment
PASTEL_COLORS = [
    'lightblue', 'lightcoral', 'lightgreen', 'lightpink', 
    'lightsalmon', 'lightyellow', 'lavender', 'mistyrose',
    'peachpuff', 'powderblue', 'paleturquoise', 'thistle',
    'plum', 'wheat', 'lightcyan', 'honeydew', 'azure'
]

ALL_COLORS = [
    # Pastel colors
    'lightblue', 'lightcoral', 'lightgreen', 'lightpink', 
    'lightsalmon', 'lightyellow', 'lavender', 'mistyrose',
    'peachpuff', 'powderblue', 'paleturquoise', 'thistle',
    'plum', 'wheat',
    # Bright colors
    'blue', 'green', 'red', 'purple', 'gold', 'orange', 
    'yellow', 'cyan', 'magenta', 'lime', 'hotpink',
    # Standard colors
    'brown', 'dodgerblue', 'forestgreen', 'darkred', 
    'darkorchid', 'darkturquoise', 'aquamarine', 'coral',
    'teal', 'olive'
]

# Create a mapping of colors to display with color indicators
def get_color_display_name(color):
    """Add a colored square indicator before the color name"""
    # Unicode colored square
    return f"🟦 {color}" if color else color

def get_color_options_with_swatches():
    """Create color options list with visual indicators"""
    color_map = {}
    for color in ALL_COLORS:
        # Map display name back to actual color
        color_map[get_color_display_name(color)] = color
    return color_map

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

def parse_genbank_file(file_content):
    """Parse GenBank file and extract features"""
    features_list = []
    
    try:
        # Parse GenBank file
        record = SeqIO.read(io.StringIO(file_content.decode('utf-8')), "genbank")
        
        for feature in record.features:
            # Skip 'source' features
            if feature.type.lower() == 'source':
                continue
            
            # Extract feature information
            start = int(feature.location.start) + 1  # Convert to 1-based
            end = int(feature.location.end)
            feature_type = feature.type
            
            # Detect strand (1 = forward, -1 = reverse/complement)
            strand = feature.location.strand if hasattr(feature.location, 'strand') else 1
            
            # Extract standard_name or label
            name = None
            if 'standard_name' in feature.qualifiers:
                name = feature.qualifiers['standard_name'][0]
            elif 'label' in feature.qualifiers:
                name = feature.qualifiers['label'][0]
            elif 'gene' in feature.qualifiers:
                name = feature.qualifiers['gene'][0]
            elif 'product' in feature.qualifiers:
                name = feature.qualifiers['product'][0]
            else:
                name = f"{feature_type}_{start}_{end}"
            
            # Check if this is a promoter (regulatory + "promoter" in qualifiers)
            is_promoter = False
            if feature_type.lower() == 'regulatory':
                # Check note, standard_name, or regulatory_class for "promoter"
                for qual_key in ['note', 'standard_name', 'regulatory_class', 'label']:
                    if qual_key in feature.qualifiers:
                        qual_value = str(feature.qualifiers[qual_key]).lower()
                        if 'promoter' in qual_value:
                            is_promoter = True
                            break
            
            # Default position: misc_feature = Down, others = Up
            box_position = 'Down' if feature_type.lower() == 'misc_feature' else 'Up'
            
            # Assign random pastel color
            colour = random.choice(PASTEL_COLORS)
            
            # Default arrow type
            arrow_type = 'arrow'
            
            features_list.append({
                'Element': name,
                'Start': start,
                'End': end,
                'Box position': box_position,
                'Colour': colour,
                'Arrow end type': arrow_type,
                'Feature type': feature_type,
                'Strand': strand,
                'Is promoter': is_promoter
            })
        
        return pd.DataFrame(features_list)
    
    except Exception as e:
        st.error(f"Error parsing GenBank file: {str(e)}")
        return None

def create_plasmid_map(data, font_size=11, show_positions=False, text_orientation='horizontal',
                       region_start=None, region_end=None, visible_elements=None):
    """Create plasmid map from DataFrame with advanced options"""
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Clean data
    data['box_position'] = data['box_position'].str.strip().str.lower()
    data['arrow_end_type'] = data['arrow_end_type'].str.strip().str.lower()
    data['colour'] = data['colour'].str.strip()
    
    # Add default values for strand and is_promoter if not present
    if 'strand' not in data.columns:
        data['strand'] = 1
    if 'is_promoter' not in data.columns:
        data['is_promoter'] = False
    
    # Filter by visibility if specified
    if visible_elements is not None:
        data = data[data['element'].isin(visible_elements)].copy()
    
    if len(data) == 0:
        st.warning("No elements to display!")
        return None
    
    # Filter by region if specified
    if region_start is not None and region_end is not None:
        data = data[
            ((data['start'] >= region_start) & (data['start'] <= region_end)) |
            ((data['end'] >= region_start) & (data['end'] <= region_end)) |
            ((data['start'] <= region_start) & (data['end'] >= region_end))
        ].copy()
        
        if len(data) == 0:
            st.warning("No elements in selected region!")
            return None
        
        plasmid_start = region_start
        plasmid_end = region_end
    else:
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
    position_offset = 30  # Offset for position labels from boxes
    arrow_point_width = 40  # Width of arrow point for promoters
    
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
        element_width = element_end - element_start
        
        # Calculate positions
        if row['box_position'] == 'up':
            box_y = box_height / 2
            text_y = box_height + text_distance
            arrow_start_y = box_height
            arrow_end_y = text_y - 20
            position_y = box_y + box_height/2 + position_offset  # Above box
        else:
            box_y = -box_height / 2
            text_y = -box_height - text_distance
            arrow_start_y = -box_height
            arrow_end_y = text_y + 20
            position_y = box_y - box_height/2 - position_offset  # Below box
        
        converted_color = convert_r_color(row['colour'])
        
        # Draw box or promoter arrow
        if row['is_promoter']:
            # Draw promoter as arrow-shaped box
            strand = row['strand']
            arrow_point = min(arrow_point_width, element_width * 0.3)  # Limit point size
            
            if strand == -1:  # Complement/reverse strand - arrow points LEFT
                # Create left-pointing arrow polygon
                vertices = [
                    (element_start, box_y - box_height/2),      # Bottom-left (point)
                    (element_start + arrow_point, box_y - box_height/2),  # Bottom after point
                    (element_end, box_y - box_height/2),        # Bottom-right
                    (element_end, box_y + box_height/2),        # Top-right
                    (element_start + arrow_point, box_y + box_height/2),  # Top after point
                    (element_start, box_y)                      # Top-left (point tip)
                ]
            else:  # Forward strand - arrow points RIGHT
                # Create right-pointing arrow polygon
                vertices = [
                    (element_start, box_y - box_height/2),      # Bottom-left
                    (element_end - arrow_point, box_y - box_height/2),  # Bottom before point
                    (element_end, box_y),                       # Right point tip
                    (element_end - arrow_point, box_y + box_height/2),  # Top before point
                    (element_start, box_y + box_height/2)       # Top-left
                ]
            
            # Draw the promoter arrow
            promoter_arrow = patches.Polygon(
                vertices,
                closed=True,
                linewidth=1.5,
                edgecolor='black',
                facecolor=converted_color,
                zorder=2
            )
            ax.add_patch(promoter_arrow)
            
        else:
            # Draw regular rectangular box
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
        
        # Draw arrow to label
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
        
        # Add element name label
        if text_orientation == 'vertical':
            ax.text(midpoint, text_y, row['element'],
                   ha='center', va='center', fontsize=font_size, zorder=3,
                   rotation=90, rotation_mode='anchor')
        else:
            ax.text(midpoint, text_y, row['element'],
                   ha='center', va='center', fontsize=font_size, zorder=3)
        
        # Add size labels if enabled (CHANGED: show size instead of positions)
        if show_positions:
            element_size = int(row['end'] - row['start'])
            size_text = f"{element_size} bp"
            position_fontsize = max(6, font_size - 3)  # Smaller font
            ax.text(midpoint, position_y, size_text,
                   ha='center', va='center', 
                   fontsize=position_fontsize, 
                   color='gray', 
                   zorder=3)
    
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
st.title("🧬 Plasmid Map Generator - Advanced")
st.markdown("Create beautiful plasmid maps from GenBank files, CSV, Excel, or manual entry!")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Display Settings")
    
    font_size = st.slider("Label Font Size", min_value=8, max_value=20, value=11, step=1)
    
    text_orientation = st.radio(
        "Element Name Orientation",
        options=['horizontal', 'vertical'],
        index=0
    )
    
    show_positions = st.checkbox("Show Element Sizes", value=False, 
                                 help="Display the size (in bp) of each element")
    
    st.markdown("---")
    st.header("🎨 Color Palette")
    
    st.markdown("**✨ Pastel Colors:**")
    st.markdown("lightblue, lightcoral, lightgreen, lightpink, lightsalmon, lightyellow, lavender, mistyrose, peachpuff, powderblue, paleturquoise, thistle, plum, wheat")
    
    st.markdown("**🌈 Bright Colors:**")
    st.markdown("red, blue, green, yellow, orange, purple, cyan, magenta, lime, hotpink")
    
    st.markdown("**🎯 Standard Colors:**")
    st.markdown("forestgreen, dodgerblue, brown, gold, coral, darkorchid, darkturquoise, teal, olive")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 GenBank File", "📊 CSV/Excel", "✏️ Manual Entry", "ℹ️ Help"])

with tab1:
    st.header("Upload GenBank (.gb) File")
    
    gb_file = st.file_uploader(
        "Choose a GenBank file",
        type=['gb', 'gbk', 'genbank'],
        help="Upload a GenBank format file"
    )
    
    if gb_file is not None:
        try:
            # Parse GenBank file
            file_content = gb_file.read()
            df = parse_genbank_file(file_content)
            
            if df is not None and len(df) > 0:
                st.success(f"✅ Loaded {len(df)} features from GenBank file!")
                
                # Store in session state
                if 'gb_data' not in st.session_state or not df.equals(st.session_state.get('gb_data', pd.DataFrame())):
                    st.session_state.gb_data = df.copy()
                    # Initialize color preferences
                    st.session_state.color_prefs = {row['Element']: row['Colour'] for _, row in df.iterrows()}
                    st.session_state.position_prefs = {row['Element']: row['Box position'] for _, row in df.iterrows()}
                    # Initialize visibility preferences (all visible by default)
                    st.session_state.visibility_prefs = {row['Element']: True for _, row in df.iterrows()}
                
                # Region selection
                st.subheader("📍 Region Selection")
                
                min_pos = int(df['Start'].min())
                max_pos = int(df['End'].max())
                
                use_region = st.checkbox("Show only specific region", value=False)
                
                if use_region:
                    col1, col2 = st.columns(2)
                    with col1:
                        region_start = st.number_input("Region Start (bp)", 
                                                      min_value=min_pos, 
                                                      max_value=max_pos, 
                                                      value=min_pos)
                    with col2:
                        region_end = st.number_input("Region End (bp)", 
                                                    min_value=min_pos, 
                                                    max_value=max_pos, 
                                                    value=max_pos)
                else:
                    region_start = None
                    region_end = None
                
                # Element customization
                st.subheader("🎨 Customize Elements")
                
                with st.expander("Change Colors, Positions, and Visibility", expanded=False):
                    st.markdown("*Adjust colors, positions, and visibility for individual elements*")
                    st.markdown("*Uncheck to hide an element from the map*")
                    
                    for idx, row in df.iterrows():
                        element_name = row['Element']
                        is_promoter = row.get('Is promoter', False)
                        element_size = int(row['End'] - row['Start'])
                        
                        # Add indicator if it's a promoter
                        display_name = f"**{element_name}** ({element_size} bp)"
                        if is_promoter:
                            display_name = f"**{element_name}** ({element_size} bp) 🔷"
                        
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                        
                        with col1:
                            st.markdown(display_name)
                        
                        with col2:
                            # Get current color preference
                            current_color = st.session_state.color_prefs.get(element_name, row['Colour'])
                            
                            # Create color selector with visual swatches
                            new_color = st.selectbox(
                                f"Color",
                                options=ALL_COLORS,
                                index=ALL_COLORS.index(current_color),
                                key=f"color_{idx}",
                                label_visibility="collapsed"
                            )
                            st.session_state.color_prefs[element_name] = new_color
                            
                            # Show color swatch below the dropdown
                            st.markdown(
                                f'<div style="width: 100%; height: 20px; background-color: {new_color}; '
                                f'border: 1px solid #ccc; border-radius: 3px; margin-top: 5px;"></div>',
                                unsafe_allow_html=True
                            )
                        
                        with col3:
                            new_position = st.selectbox(
                                f"Position",
                                options=['Up', 'Down'],
                                index=0 if st.session_state.position_prefs.get(element_name, row['Box position']) == 'Up' else 1,
                                key=f"pos_{idx}",
                                label_visibility="collapsed"
                            )
                            st.session_state.position_prefs[element_name] = new_position
                        
                        with col4:
                            is_visible = st.checkbox(
                                "Show",
                                value=st.session_state.visibility_prefs.get(element_name, True),
                                key=f"vis_{idx}"
                            )
                            st.session_state.visibility_prefs[element_name] = is_visible
                
                # Apply customizations
                display_df = df.copy()
                for idx, row in display_df.iterrows():
                    element_name = row['Element']
                    if element_name in st.session_state.color_prefs:
                        display_df.at[idx, 'Colour'] = st.session_state.color_prefs[element_name]
                    if element_name in st.session_state.position_prefs:
                        display_df.at[idx, 'Box position'] = st.session_state.position_prefs[element_name]
                
                # Get visible elements
                visible_elements = [name for name, vis in st.session_state.visibility_prefs.items() if vis]
                
                # Show data table (only visible elements)
                st.subheader("Feature Table")
                visible_df = display_df[display_df['Element'].isin(visible_elements)].copy()
                if len(visible_df) > 0:
                    # Add size column for display
                    visible_df['Size (bp)'] = visible_df['End'] - visible_df['Start']
                    st.dataframe(visible_df[['Element', 'Start', 'End', 'Size (bp)', 'Feature type', 'Box position', 'Colour']], 
                               use_container_width=True)
                    st.info(f"Showing {len(visible_df)} of {len(df)} elements")
                else:
                    st.warning("No elements selected to display!")
                
                # Generate button
                if st.button("🎨 Generate Plasmid Map", type="primary", key="gb_generate"):
                    with st.spinner("Creating your plasmid map..."):
                        try:
                            fig = create_plasmid_map(
                                display_df, 
                                font_size=font_size,
                                show_positions=show_positions,
                                text_orientation=text_orientation,
                                region_start=region_start,
                                region_end=region_end,
                                visible_elements=visible_elements
                            )
                            
                            if fig is not None:
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
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("👆 Upload a GenBank file to get started")

with tab2:
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
                        fig = create_plasmid_map(
                            df, 
                            font_size=font_size,
                            show_positions=show_positions,
                            text_orientation=text_orientation
                        )
                        
                        if fig is not None:
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
        st.info("👆 Upload a file to get started")
        
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

with tab3:
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
            
            # Color selection with swatch display
            colour = st.selectbox("Colour", ALL_COLORS)
            # Show selected color swatch
            st.markdown(
                f'<div style="width: 100%; height: 30px; background-color: {colour}; '
                f'border: 1px solid #ccc; border-radius: 3px; margin-top: 5px;"></div>',
                unsafe_allow_html=True
            )
            
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
        # Add size column
        manual_df['Size (bp)'] = manual_df['End'] - manual_df['Start']
        st.dataframe(manual_df, use_container_width=True)
        
        # Generate button
        if st.button("🎨 Generate Plasmid Map", type="primary", key="manual_generate"):
            with st.spinner("Creating your plasmid map..."):
                try:
                    fig = create_plasmid_map(
                        manual_df, 
                        font_size=font_size,
                        show_positions=show_positions,
                        text_orientation=text_orientation
                    )
                    
                    if fig is not None:
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

with tab4:
    st.header("📖 How to Use")
    
    st.markdown("""
    ### 🆕 GenBank File Support
    
    Upload a `.gb` file and the app will automatically:
    - Parse all features (except 'source')
    - Extract feature names from `/standard_name=`
    - Assign random pastel colors (customizable!)
    - Set `misc_feature` to "Down" by default
    - **Detect promoters** and draw as arrow-shaped boxes 🔷
    - **Detect strand orientation** (forward/complement)
    - Allow you to customize colors, positions, and visibility
    
    ### 🔷 Promoter Arrows (NEW!)
    
    **Automatic Detection:**
    - Features marked as `regulatory` with "promoter" in notes
    - Drawn as arrow-shaped boxes instead of rectangles
    
    **Arrow Direction:**
    - **Forward strand** → Arrow points RIGHT
    - **Complement/reverse** → Arrow points LEFT
    - Direction matches GenBank strand annotation
    
    ### 🎯 Advanced Features
    
    **Element Visibility:**
    - Show/hide individual elements
    - Checkbox for each element in customization panel
    - Remove unwanted features from map
    
    **Color Swatches:**
    - Visual color preview below each dropdown
    - See your color choice immediately
    - Easy to match colors across elements
    
    **Region Selection:**
    - View entire plasmid OR specific region
    - Enter start and end positions
    
    **Text Orientation:**
    - Horizontal (default) or Vertical element names
    
    **Size Labels:**
    - Toggle to show/hide element sizes
    - Displayed in smaller grey font as "XXX bp"
    - Shows length of each element
    
    **Individual Element Control:**
    - Change color for each element separately
    - Change Up/Down position for each element
    - See element sizes in the customization panel
    
    ### 📁 Input Options
    
    1. **GenBank File** - Upload .gb file (automatic parsing)
    2. **CSV/Excel** - Upload traditional data file
    3. **Manual Entry** - Add elements one by one
    
    ### 📊 CSV/Excel Format
    
    | Column | Description | Example |
    |--------|-------------|---------|
    | Element | Name of genetic element | "Promoter", "GeneX" |
    | Start | Start position in bp | 100, 1500 |
    | End | End position in bp | 300, 2000 |
    | Box position | "Up" or "Down" | Up, Down |
    | Colour | Color name | blue, lightgreen, coral |
    | Arrow end type | "arrow" or "flat" | arrow, flat |
    
    ### 🎨 Tips
    
    - **GenBank files**: Colors assigned randomly, but you can change them
    - **Color swatches**: Check the color preview to match your scheme
    - **Region view**: Focus on specific gene clusters
    - **Vertical text**: Better for long element names
    - **Size labels**: Shows element length instead of coordinates
    - **Mix colors**: Use pastels for most, bright for key elements
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧬 Plasmid Map Generator  v1.0 | Dunkelmann Lab | Plant Synthetic Biology at MPI-MP | Created by Alicia Clarke </p>
    <p><small>Supports .gb, .csv, .xlsx • Region selection • Size display • Color swatches • Custom colors</small></p>
</div>
""", unsafe_allow_html=True)
