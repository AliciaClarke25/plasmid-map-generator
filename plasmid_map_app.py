import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
from io import BytesIO
import base64

# Try to import BioPython for GenBank parsing
try:
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# Define pastel colors
PASTEL_COLORS = [
    'lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 
    'lightpink', 'lightsalmon', 'lightcyan', 'lavender',
    'peachpuff', 'palegreen', 'mistyrose', 'wheat',
    'lightsteelblue', 'thistle', 'lightgoldenrodyellow',
    'powderblue', 'pink'
]

# Extended color palette for dropdowns (organized by category)
ALL_COLORS = [
    # Pastel colors
    'lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 
    'lightpink', 'lightsalmon', 'lightcyan', 'lavender',
    'peachpuff', 'palegreen', 'mistyrose', 'wheat',
    'lightsteelblue', 'thistle',
    # Bright colors
    'red', 'blue', 'green', 'yellow', 'orange', 'purple',
    'cyan', 'magenta', 'lime', 'pink',
    # Standard colors
    'brown', 'gray', 'olive', 'navy', 'teal', 'maroon',
    'coral', 'gold', 'indigo', 'crimson', 'darkgreen',
    'darkblue', 'darkred', 'darkorange', 'darkviolet',
    'deepskyblue', 'forestgreen', 'hotpink', 'khaki',
    'lightseagreen', 'mediumpurple', 'mediumseagreen',
    'orangered', 'orchid', 'palevioletred', 'peru',
    'plum', 'royalblue', 'salmon', 'sandybrown',
    'seagreen', 'sienna', 'skyblue', 'slateblue',
    'springgreen', 'steelblue', 'tan', 'tomato',
    'turquoise', 'violet', 'yellowgreen'
]

def convert_r_color(color_name):
    """Convert R color names to matplotlib equivalents"""
    color_map = {
        'lightyellow2': 'lightyellow',
        'lightblue2': 'lightblue',
        'lightpink2': 'lightpink'
    }
    return color_map.get(color_name, color_name)

def parse_genbank_file(uploaded_file):
    """
    Parse a GenBank file and extract features for plasmid mapping
    Returns a pandas DataFrame with columns: Element, Start, End, Color, Position, Strand, Note, IsPromoter
    """
    if not BIOPYTHON_AVAILABLE:
        st.error("BioPython is not installed. Please install it with: pip install biopython")
        return None
    
    try:
        # Convert Streamlit's UploadedFile to text mode for BioPython
        from io import StringIO
        
        # Read the file content and decode if necessary
        file_content = uploaded_file.read()
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8')
        
        # Create a StringIO object for BioPython
        text_file = StringIO(file_content)
        
        # Parse the GenBank file
        record = SeqIO.read(text_file, "genbank")
        plasmid_length = len(record.seq)
        
        elements_data = []
        
        for feature in record.features:
            # Skip source features
            if feature.type.lower() == 'source':
                continue
            
            # Extract feature location
            start = int(feature.location.start) + 1  # Convert to 1-based
            end = int(feature.location.end)
            
            # Get strand information (+1 = forward, -1 = reverse/complement)
            strand = feature.location.strand if hasattr(feature.location, 'strand') else 1
            
            # Extract feature name with priority order
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
                # Fallback to feature type and position
                name = f"{feature.type}_{start}_{end}"
            
            # Extract note qualifier
            note = ""
            if 'note' in feature.qualifiers:
                note = " ".join(feature.qualifiers['note']).lower()
            
            # Check if this is a promoter
            is_promoter = (feature.type.lower() == 'regulatory' and 'promoter' in note)
            
            # Assign random pastel color
            color = np.random.choice(PASTEL_COLORS)
            
            # Smart positioning based on strand direction:
            # Forward strand (+1) = Up
            # Reverse strand/complement (-1) = Down
            # If strand info missing, use old logic (misc_feature = Down, others = Up)
            if strand == -1:
                position = "Down"
            elif strand == 1:
                position = "Up"
            else:
                # Fallback for features without strand info
                position = "Down" if feature.type.lower() == 'misc_feature' else "Up"
            
            elements_data.append({
                'Element': name,
                'Start': start,
                'End': end,
                'Color': color,
                'Position': position,
                'Strand': strand,
                'Note': note,
                'IsPromoter': is_promoter
            })
        
        df = pd.DataFrame(elements_data)
        return df, plasmid_length
    
    except Exception as e:
        st.error(f"Error parsing GenBank file: {str(e)}")
        return None

def create_arrow_polygon(x_center, y_center, width, height, direction='right'):
    """
    Create an arrow-shaped polygon for promoters
    direction: 'right' for forward strand, 'left' for reverse strand
    """
    if direction == 'right':
        # Arrow pointing right
        points = [
            [x_center - width/2, y_center - height/2],  # Bottom left
            [x_center + width/3, y_center - height/2],  # Bottom middle
            [x_center + width/3, y_center - height/1.5], # Bottom arrow point
            [x_center + width/2, y_center],              # Right tip
            [x_center + width/3, y_center + height/1.5], # Top arrow point
            [x_center + width/3, y_center + height/2],   # Top middle
            [x_center - width/2, y_center + height/2],   # Top left
        ]
    else:  # left
        # Arrow pointing left
        points = [
            [x_center + width/2, y_center - height/2],  # Bottom right
            [x_center - width/3, y_center - height/2],  # Bottom middle
            [x_center - width/3, y_center - height/1.5], # Bottom arrow point
            [x_center - width/2, y_center],              # Left tip
            [x_center - width/3, y_center + height/1.5], # Top arrow point
            [x_center - width/3, y_center + height/2],   # Top middle
            [x_center + width/2, y_center + height/2],   # Top right
        ]
    
    return Polygon(points, closed=True)

def create_plasmid_map(df, plasmid_length, label_font=11, show_positions=False, 
                       text_orientation='horizontal', region_start=None, region_end=None):
    """Create plasmid map visualization with arrow-shaped promoters"""
    
    # Filter for region if specified
    if region_start is not None and region_end is not None:
        df = df[
            ((df['Start'] >= region_start) & (df['Start'] <= region_end)) |
            ((df['End'] >= region_start) & (df['End'] <= region_end)) |
            ((df['Start'] <= region_start) & (df['End'] >= region_end))
        ].copy()
        
        if df.empty:
            st.warning("No elements found in the specified region.")
            return None
    
    # Determine plot range
    if region_start is not None and region_end is not None:
        plot_start = region_start
        plot_end = region_end
    else:
        plot_start = 0
        plot_end = plasmid_length
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Draw the plasmid line
    ax.plot([plot_start, plot_end], [0, 0], 'k-', linewidth=3)
    
    box_height = 80
    text_distance = 120  # Distance from box to line endpoint
    text_gap = 30  # Gap between line endpoint and text start
    
    # For horizontal text, create staggered levels to avoid overlap
    if text_orientation == 'horizontal':
        # Sort elements by position to assign levels
        df_sorted = df.sort_values('Start').reset_index(drop=True)
        element_levels = {}
        num_levels = 3  # Use 3 different height levels
        
        for idx, row in df_sorted.iterrows():
            element = row['Element']
            position = row['Position']
            # Cycle through levels based on index
            level = idx % num_levels
            element_levels[element] = level
    
    for _, row in df.iterrows():
        element = row['Element']
        start = row['Start']
        end = row['End']
        color = convert_r_color(row['Color'])
        position = row['Position']
        is_promoter = row.get('IsPromoter', False)
        strand = row.get('Strand', 1)
        
        # Calculate center and width
        center = (start + end) / 2
        width = end - start
        
        # Calculate element size for label
        element_size = end - start + 1
        size_label = f" ({element_size} bp)" if show_positions else ""
        
        # Determine y position and line height
        if text_orientation == 'horizontal':
            # Use staggered levels for horizontal text
            level = element_levels.get(element, 0)
            level_multiplier = 1 + (level * 0.6)  # Stagger at 1x, 1.6x, 2.2x height (more spacing!)
            
            if position == "Up":
                box_y = box_height / 2
                line_end_y = box_height + (text_distance * level_multiplier)
                text_y = line_end_y + text_gap
            else:
                box_y = -box_height / 2
                line_end_y = -box_height - (text_distance * level_multiplier)
                text_y = line_end_y - text_gap
        else:
            # Vertical text - no staggering needed
            if position == "Up":
                box_y = box_height / 2
                line_end_y = box_height + text_distance
                text_y = line_end_y + text_gap
            else:
                box_y = -box_height / 2
                line_end_y = -box_height - text_distance
                text_y = line_end_y - text_gap
        
        # Draw arrow for promoters, rectangle for others
        if is_promoter:
            # Draw arrow shape
            arrow_direction = 'right' if strand >= 0 else 'left'
            arrow = create_arrow_polygon(center, box_y, width, box_height, arrow_direction)
            arrow_patch = patches.Polygon(arrow.get_xy(), closed=True, 
                                         edgecolor='black', facecolor=color, 
                                         linewidth=1.5, zorder=3)
            ax.add_patch(arrow_patch)
        else:
            # Draw rectangle
            rect = patches.Rectangle((start, box_y - box_height/2), width, box_height,
                                    linewidth=1.5, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
        
        # Draw vertical line connecting box to text (now stops before text)
        ax.plot([center, center], [box_y, line_end_y], 'k-', linewidth=1.5)
        
        # Add element name with size in brackets (if enabled)
        if text_orientation == 'vertical':
            # For vertical text (rotated 90°)
            if position == "Up":
                # Draw element name
                ax.text(center, text_y, element, ha='left', va='center',
                       fontsize=label_font, rotation=90, rotation_mode='anchor',
                       color='black')
                # Add size if enabled
                if show_positions:
                    offset_y = len(element) * label_font * 0.2 + 5
                    ax.text(center, text_y + offset_y, size_label, ha='left', va='center',
                           fontsize=label_font, rotation=90, rotation_mode='anchor',
                           color='grey')
            else:
                # Draw element name
                ax.text(center, text_y, element, ha='right', va='center',
                       fontsize=label_font, rotation=90, rotation_mode='anchor',
                       color='black')
                # Add size if enabled
                if show_positions:
                    offset_y = len(element) * label_font * 0.7 + 15
                    ax.text(center, text_y - offset_y, size_label, ha='right', va='center',
                           fontsize=label_font, rotation=90, rotation_mode='anchor',
                           color='grey')
        else:
            # Horizontal text
            va_align = 'bottom' if position == "Up" else 'top'
            
            if show_positions:
                # Draw element name in black, centered
                ax.text(center, text_y, element, ha='center', va=va_align,
                       fontsize=label_font, color='black')
                
                # Draw size in grey using offset from the element name position
                # Use textcoords='offset points' for pixel-based positioning
                ax.annotate(size_label, xy=(center, text_y), 
                           xytext=(len(element)*label_font*0.3 + 10, 0),  # Offset in points (pixels)
                           textcoords='offset points',
                           ha='left', va=va_align,
                           fontsize=label_font, color='grey')
            else:
                # Just element name, no size
                ax.text(center, text_y, element, ha='center', va=va_align,
                       fontsize=label_font, color='black')
    
    # Set axis properties - adjust y limits for staggered text
    ax.set_xlim(plot_start - 500, plot_end + 500)
    if text_orientation == 'horizontal':
        # Need more vertical space for staggered levels (up to 2.2x multiplier)
        y_max = max(box_height + text_distance * 4.5 + text_gap + 100, 700)
    else:
        y_max = max(box_height + text_distance + text_gap + 100, 400)
    ax.set_ylim(-y_max, y_max)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def fig_to_bytes(fig, format='png', dpi=500):
    """Convert matplotlib figure to bytes with specified DPI"""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf

def get_download_link(buf, filename, file_label):
    """Generate a download link for a file"""
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{file_label}</a>'
    return href

# Streamlit app
st.set_page_config(page_title="Plasmid Map Generator", layout="wide")
st.title("🧬 Plasmid Map Generator")
st.markdown("Generate professional plasmid maps from GenBank files, CSV/Excel, or manual entry")

# Sidebar controls
st.sidebar.header("⚙️ Map Settings")
label_font = st.sidebar.slider("Label Font Size", 8, 20, 11)
text_orientation = st.sidebar.radio("Element Name Orientation", 
                                    options=['horizontal', 'vertical'],
                                    index=0)
show_positions = st.sidebar.checkbox("Show Element Sizes (in brackets)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Color Palette Reference")
st.sidebar.markdown("""
**Pastel Colors:** lightblue, lightcoral, lightgreen, lightyellow, lightpink, lightsalmon, lightcyan, lavender, peachpuff, palegreen, mistyrose, wheat, lightsteelblue, thistle

**Bright Colors:** red, blue, green, yellow, orange, purple, cyan, magenta, lime, pink

**Standard Colors:** brown, gray, olive, navy, teal, maroon, coral, gold, indigo, and more
""")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 GenBank File", "📊 CSV/Excel Upload", "✍️ Manual Entry", "❓ Help"])

# TAB 1: GenBank File Upload
with tab1:
    st.header("Upload GenBank File")
    st.markdown("""
    Upload a GenBank (.gb, .gbk, .genbank) file to automatically extract features and generate a plasmid map.
    
    **Features:**
    - ✅ Automatic feature parsing
    - ✅ Arrow-shaped promoter boxes (pointing based on strand direction)
    - ✅ Individual color customization with preview
    - ✅ Show/hide specific elements (with Select/Deselect All)
    - ✅ Region selection
    - ✅ Editable element labels
    - ✅ Element sizes in brackets (grey text)
    - ✅ Staggered horizontal labels (no overlap!)
    - ✅ Smart positioning (strand-based)
    - ✅ High resolution output (500 DPI)
    """)
    
    if not BIOPYTHON_AVAILABLE:
        st.error("""
        ⚠️ **BioPython not installed**
        
        To use GenBank file parsing, install BioPython:
        ```
        pip install biopython
        ```
        """)
    else:
        uploaded_gb = st.file_uploader("Choose a GenBank file", type=['gb', 'gbk', 'genbank'], key='gb_uploader')
        
        if uploaded_gb is not None:
            # Parse GenBank file
            result = parse_genbank_file(uploaded_gb)
            
            if result is not None:
                df, plasmid_length = result
                st.success(f"✅ Successfully parsed {len(df)} features from {uploaded_gb.name} ({plasmid_length} bp)")
                
                # Store in session state
                st.session_state.gb_data = df
                st.session_state.plasmid_length = plasmid_length
                
                # Initialize color and position preferences if not exists
                if 'color_prefs' not in st.session_state:
                    st.session_state.color_prefs = {}
                if 'position_prefs' not in st.session_state:
                    st.session_state.position_prefs = {}
                if 'enabled_elements' not in st.session_state:
                    st.session_state.enabled_elements = {}
                if 'edited_labels' not in st.session_state:
                    st.session_state.edited_labels = {}
                
                # Initialize enabled state for all elements using unique keys
                for idx, row in df.iterrows():
                    element = row['Element']
                    unique_key = f"{idx}_{element}"
                    if unique_key not in st.session_state.enabled_elements:
                        st.session_state.enabled_elements[unique_key] = True
                    # Initialize edited labels with original element name
                    if unique_key not in st.session_state.edited_labels:
                        st.session_state.edited_labels[unique_key] = element
                
                # Region selection
                st.markdown("---")
                st.subheader("🎯 Region Selection (Optional)")
                show_region = st.checkbox("Show only specific region", value=False, key='show_region_gb')
                
                region_start = None
                region_end = None
                
                if show_region:
                    col1, col2 = st.columns(2)
                    with col1:
                        region_start = st.number_input("Region Start (bp)", 
                                                      min_value=1, 
                                                      max_value=plasmid_length,
                                                      value=1,
                                                      key='region_start_gb')
                    with col2:
                        region_end = st.number_input("Region End (bp)", 
                                                    min_value=1, 
                                                    max_value=plasmid_length,
                                                    value=min(1000, plasmid_length),
                                                    key='region_end_gb')
                
                # Initialize checkbox refresh counter if not exists
                if 'checkbox_refresh' not in st.session_state:
                    st.session_state.checkbox_refresh = 0
                
                # Customization section
                st.markdown("---")
                st.subheader("🎨 Customize Elements")
                
                # Add Select All / Deselect All buttons
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
                with col_btn1:
                    if st.button("✅ Select All", use_container_width=True, key="select_all_btn"):
                        for idx, row in df.iterrows():
                            element = row['Element']
                            unique_key = f"{idx}_{element}"
                            st.session_state.enabled_elements[unique_key] = True
                        st.session_state.checkbox_refresh += 1  # Force checkbox recreation
                
                with col_btn2:
                    if st.button("❌ Deselect All", use_container_width=True, key="deselect_all_btn"):
                        for idx, row in df.iterrows():
                            element = row['Element']
                            unique_key = f"{idx}_{element}"
                            st.session_state.enabled_elements[unique_key] = False
                        st.session_state.checkbox_refresh += 1  # Force checkbox recreation
                
                with st.expander("Customize individual element colors, positions, and visibility"):
                    # Initialize edited labels dict if not exists
                    if 'edited_labels' not in st.session_state:
                        st.session_state.edited_labels = {}
                    
                    # Add column headers
                    header_cols = st.columns([3, 2, 2, 2, 1])
                    with header_cols[0]:
                        st.markdown("**Original Name**")
                    with header_cols[1]:
                        st.markdown("**Display Label**")
                    with header_cols[2]:
                        st.markdown("**Color**")
                    with header_cols[3]:
                        st.markdown("**Position**")
                    with header_cols[4]:
                        st.markdown("**Show**")
                    
                    st.markdown("---")
                    
                    for idx, row in df.iterrows():
                        element = row['Element']
                        is_promoter = row.get('IsPromoter', False)
                        strand = row.get('Strand', 1)
                        
                        # Create unique key using both index and element name to avoid duplicates
                        unique_key = f"{idx}_{element}"
                        
                        # Create columns for each element's controls
                        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                        
                        with col1:
                            # Show element name with promoter indicator and strand info
                            if is_promoter:
                                arrow_symbol = "→" if strand >= 0 else "←"
                                strand_info = "forward" if strand >= 0 else "reverse"
                                st.markdown(f"**{element}** {arrow_symbol} (Promoter, {strand_info})")
                            else:
                                strand_info = "forward" if strand >= 0 else "reverse"
                                st.markdown(f"**{element}** ({strand_info})")
                        
                        with col2:
                            # Editable label
                            current_label = st.session_state.edited_labels.get(unique_key, element)
                            new_label = st.text_input(
                                "Label",
                                value=current_label,
                                key=f"label_{unique_key}",
                                label_visibility="collapsed",
                                placeholder="Edit label..."
                            )
                            st.session_state.edited_labels[unique_key] = new_label
                        
                        with col3:
                            # Color selector with preview
                            current_color = st.session_state.color_prefs.get(unique_key, row['Color'])
                            
                            new_color = st.selectbox(
                                f"Color",
                                options=ALL_COLORS,
                                index=ALL_COLORS.index(current_color) if current_color in ALL_COLORS else 0,
                                key=f"color_{unique_key}",
                                label_visibility="collapsed"
                            )
                            st.session_state.color_prefs[unique_key] = new_color
                            
                            # Show color preview with actual color styling
                            st.markdown(
                                f'<div style="background-color: {new_color}; width: 100%; height: 25px; '
                                f'border: 2px solid #333; border-radius: 4px; margin-top: 5px;"></div>', 
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            # Position selector
                            current_position = st.session_state.position_prefs.get(unique_key, row['Position'])
                            new_position = st.selectbox(
                                f"Position",
                                options=["Up", "Down"],
                                index=0 if current_position == "Up" else 1,
                                key=f"position_{unique_key}",
                                label_visibility="collapsed"
                            )
                            st.session_state.position_prefs[unique_key] = new_position
                        
                        with col5:
                            # Enable/disable checkbox with refresh counter in key
                            current_enabled = st.session_state.enabled_elements.get(unique_key, True)
                            enabled = st.checkbox(
                                "Show",
                                value=current_enabled,
                                key=f"enabled_{unique_key}_{st.session_state.checkbox_refresh}"
                            )
                            st.session_state.enabled_elements[unique_key] = enabled
                
                # Apply customizations to dataframe
                df_display = df.copy()
                for idx, row in df_display.iterrows():
                    element = row['Element']
                    unique_key = f"{idx}_{element}"
                    
                    # Apply edited labels
                    if unique_key in st.session_state.edited_labels:
                        df_display.loc[idx, 'Element'] = st.session_state.edited_labels[unique_key]
                    
                    # Apply color preferences
                    if unique_key in st.session_state.color_prefs:
                        df_display.loc[idx, 'Color'] = st.session_state.color_prefs[unique_key]
                    
                    # Apply position preferences
                    if unique_key in st.session_state.position_prefs:
                        df_display.loc[idx, 'Position'] = st.session_state.position_prefs[unique_key]
                
                # Filter out disabled elements
                enabled_indices = []
                for idx, row in df_display.iterrows():
                    element = row['Element']
                    unique_key = f"{idx}_{element}"
                    if st.session_state.enabled_elements.get(unique_key, True):
                        enabled_indices.append(idx)
                
                df_display = df_display.loc[enabled_indices]
                
                # Show feature table
                st.markdown("---")
                st.subheader("📋 Feature Table")
                
                # Create display dataframe with edited labels
                display_df = df_display[['Element', 'Start', 'End', 'Color', 'Position', 'IsPromoter']].copy()
                display_df['IsPromoter'] = display_df['IsPromoter'].apply(lambda x: '✓' if x else '')
                display_df.columns = ['Element Name', 'Start (bp)', 'End (bp)', 'Color', 'Box Position', 'Promoter']
                st.dataframe(display_df, use_container_width=True)
                
                # Generate button
                st.markdown("---")
                if st.button("🎨 Generate Plasmid Map", type="primary", key='generate_gb'):
                    if len(df_display) == 0:
                        st.warning("⚠️ No elements selected! Please enable at least one element.")
                    else:
                        with st.spinner("Generating plasmid map..."):
                            fig = create_plasmid_map(df_display, plasmid_length, label_font, 
                                                   show_positions, text_orientation,
                                                   region_start, region_end)
                            
                            if fig:
                                st.pyplot(fig)
                                
                                # Download options
                                st.markdown("### 💾 Download Options")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    pdf_buf = fig_to_bytes(fig, 'pdf', dpi=500)
                                    st.download_button(
                                        label="📄 Download PDF",
                                        data=pdf_buf,
                                        file_name="plasmid_map.pdf",
                                        mime="application/pdf"
                                    )
                                
                                with col2:
                                    svg_buf = fig_to_bytes(fig, 'svg', dpi=500)
                                    st.download_button(
                                        label="🎨 Download SVG",
                                        data=svg_buf,
                                        file_name="plasmid_map.svg",
                                        mime="image/svg+xml"
                                    )
                                
                                with col3:
                                    png_buf = fig_to_bytes(fig, 'png', dpi=500)
                                    st.download_button(
                                        label="🖼️ Download PNG (500 DPI)",
                                        data=png_buf,
                                        file_name="plasmid_map.png",
                                        mime="image/png"
                                    )
                                
                                plt.close(fig)

# TAB 2: CSV/Excel Upload
with tab2:
    st.header("Upload CSV or Excel File")
    st.markdown("""
    Upload a CSV or Excel file with plasmid element information.
    
    **Required columns:**
    - `Element`: Name of the genetic element
    - `Start`: Start position (bp)
    - `End`: End position (bp)
    - `Color`: Color name (e.g., 'lightblue', 'red')
    - `Position`: 'Up' or 'Down'
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'], key='csv_uploader')
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate columns
            required_cols = ['Element', 'Start', 'End', 'Color', 'Position']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing required columns. Need: {', '.join(required_cols)}")
            else:
                # Get plasmid length from data
                plasmid_length = st.number_input("Plasmid Length (bp)", 
                                                min_value=int(df['End'].max()), 
                                                value=int(df['End'].max()) + 500)
                
                # Add IsPromoter column (default False for uploaded data)
                df['IsPromoter'] = False
                
                st.subheader("📋 Data Preview")
                st.dataframe(df)
                
                if st.button("🎨 Generate Plasmid Map", type="primary", key='generate_csv'):
                    with st.spinner("Generating plasmid map..."):
                        fig = create_plasmid_map(df, plasmid_length, label_font, 
                                               show_positions, text_orientation)
                        
                        if fig:
                            st.pyplot(fig)
                            
                            # Download options
                            st.markdown("### 💾 Download Options")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                pdf_buf = fig_to_bytes(fig, 'pdf', dpi=500)
                                st.download_button(
                                    label="📄 Download PDF",
                                    data=pdf_buf,
                                    file_name="plasmid_map.pdf",
                                    mime="application/pdf"
                                )
                            
                            with col2:
                                svg_buf = fig_to_bytes(fig, 'svg', dpi=500)
                                st.download_button(
                                    label="🎨 Download SVG",
                                    data=svg_buf,
                                    file_name="plasmid_map.svg",
                                    mime="image/svg+xml"
                                )
                            
                            with col3:
                                png_buf = fig_to_bytes(fig, 'png', dpi=500)
                                st.download_button(
                                    label="🖼️ Download PNG (500 DPI)",
                                    data=png_buf,
                                    file_name="plasmid_map.png",
                                    mime="image/png"
                                )
                            
                            plt.close(fig)
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# TAB 3: Manual Entry
with tab3:
    st.header("Manual Entry")
    st.markdown("Enter plasmid elements one at a time")
    
    # Initialize manual data in session state
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = []
    
    # Input form
    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            element = st.text_input("Element Name")
            start = st.number_input("Start Position (bp)", min_value=1, value=1)
            end = st.number_input("End Position (bp)", min_value=1, value=100)
        
        with col2:
            color = st.selectbox("Color", ALL_COLORS)
            position = st.selectbox("Position", ["Up", "Down"])
        
        submitted = st.form_submit_button("➕ Add Element")
        
        if submitted and element:
            st.session_state.manual_data.append({
                'Element': element,
                'Start': start,
                'End': end,
                'Color': color,
                'Position': position,
                'IsPromoter': False
            })
            st.success(f"Added: {element}")
    
    # Display current entries
    if st.session_state.manual_data:
        st.subheader("📋 Current Elements")
        df_manual = pd.DataFrame(st.session_state.manual_data)
        st.dataframe(df_manual)
        
        # Clear button
        if st.button("🗑️ Clear All Elements"):
            st.session_state.manual_data = []
            st.rerun()
        
        # Plasmid length
        plasmid_length = st.number_input("Plasmid Length (bp)", 
                                        min_value=int(df_manual['End'].max()), 
                                        value=int(df_manual['End'].max()) + 500,
                                        key='manual_length')
        
        # Generate button
        if st.button("🎨 Generate Plasmid Map", type="primary", key='generate_manual'):
            with st.spinner("Generating plasmid map..."):
                fig = create_plasmid_map(df_manual, plasmid_length, label_font,
                                       show_positions, text_orientation)
                
                if fig:
                    st.pyplot(fig)
                    
                    # Download options
                    st.markdown("### 💾 Download Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pdf_buf = fig_to_bytes(fig, 'pdf', dpi=500)
                        st.download_button(
                            label="📄 Download PDF",
                            data=pdf_buf,
                            file_name="plasmid_map.pdf",
                            mime="application/pdf"
                        )
                    
                    with col2:
                        svg_buf = fig_to_bytes(fig, 'svg', dpi=500)
                        st.download_button(
                            label="🎨 Download SVG",
                            data=svg_buf,
                            file_name="plasmid_map.svg",
                                mime="image/svg+xml"
                        )
                    
                    with col3:
                        png_buf = fig_to_bytes(fig, 'png', dpi=500)
                        st.download_button(
                            label="🖼️ Download PNG (500 DPI)",
                            data=png_buf,
                            file_name="plasmid_map.png",
                            mime="image/png"
                        )
                    
                    plt.close(fig)

# TAB 4: Help
with tab4:
    st.header("📚 Help & Documentation")
    
    st.markdown("""
    ## GenBank File Input
    
    ### File Format
    Upload GenBank format files (.gb, .gbk, .genbank) exported from SnapGene, Benchling, or other tools.
    
    ### Automatic Parsing
    - Extracts all features except 'source'
    - Feature names from: `/standard_name=` → `/label=` → `/gene=` → `/product=`
    - **Promoters:** Regulatory features with "promoter" in `/note=` get arrow shapes
      - Forward strand (no complement) → Arrow points RIGHT →
      - Reverse strand (complement) → Arrow points LEFT ←
    - Random pastel colors assigned
    - Smart positioning: `misc_feature` → Down, others → Up
    
    ### Element Control
    Each element can be:
    - **Shown/Hidden:** Check/uncheck "Show" box
    - **Recolored:** Choose from 40+ colors
    - **Repositioned:** Switch between Up/Down
    
    ### Region Selection
    Focus on specific plasmid regions (useful for large plasmids):
    1. Check "Show only specific region"
    2. Enter start and end positions
    3. Generate map showing only that region
    
    ### Advanced Features
    - **Element Sizes:** Show sizes in grey brackets next to element names (e.g., "GFP (720 bp)")
    - **Text Orientation:** Horizontal (staggered to avoid overlap) or vertical
    - **Select/Deselect All:** Quickly show or hide all elements at once
    - **Staggered Labels:** Horizontal labels at different heights to prevent overlap
    - **High Resolution:** 500 DPI output for publications
    
    ## CSV/Excel Input
    
    ### Required Columns
    - `Element`: Feature name
    - `Start`: Start position (1-based)
    - `End`: End position
    - `Color`: Color name (e.g., 'lightblue', 'red')
    - `Position`: 'Up' or 'Down'
    
    ## Manual Entry
    
    Add elements one at a time:
    1. Enter element details
    2. Click "Add Element"
    3. Repeat for all elements
    4. Generate map
    
    ## Tips
    
    - **Large plasmids:** Use region selection + vertical text
    - **Publications:** Enable element size labels, use SVG format
    - **Teaching:** Show element sizes for reference
    - **Promoters:** Automatically get directional arrows from GenBank
    - **Quick maps:** GenBank → Upload → Generate (30 seconds!)
    
    ## Color Reference
    
    **Pastel:** lightblue, lightcoral, lightgreen, lightyellow, lightpink, lightsalmon, lightcyan, lavender, peachpuff, palegreen, mistyrose, wheat, lightsteelblue, thistle
    
    **Bright:** red, blue, green, yellow, orange, purple, cyan, magenta, lime, pink
    
    **Standard:** brown, gray, olive, navy, teal, maroon, coral, gold, indigo, and 30+ more
    
    ## Troubleshooting
    
    **BioPython not found?**
    ```bash
    pip install biopython
    ```
    
    **No features extracted?**
    - Check file is valid GenBank format
    - File might only contain 'source' feature
    - Check features have qualifiers (`/label=`, `/gene=`, etc.)
    
    **Elements overlapping?**
    - Increase font size
    - Use vertical text orientation
    - Use region selection for focused view
    
    **Need help?**
    Contact your lab coordinator or check the documentation files.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🧬 Plasmid Map Generator v1.1 | Dunkelmann Lab | Plant Synthetic Biology at MPI-MP 
</div>
""", unsafe_allow_html=True)
