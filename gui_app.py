from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
import plotting_modes as pm
import os
import tempfile

# --- UI DEFINITION ---

app_ui = ui.page_fluid(
    ui.h2("Protein Plotting App", class_="mb-4"),
    
    ui.row(
        ui.column(4,
            ui.card(
                ui.card_header("Configuration"),
                ui.input_file("struc_file", "Structure File (PDB/CIF)", accept=[".pdb", ".cif", ".ent"]),
                ui.input_checkbox("split_mode", "Split Folding/Disorder Plots", value=False),
                
                ui.hr(),
                ui.h6("Plot Dimensions (Inches)"),
                ui.row(
                    ui.column(6, ui.input_numeric("plot_w", "Width", value=12, min=5, max=50)),
                    ui.column(6, ui.input_numeric("plot_h", "Height", value=8, min=5, max=50))
                ),
                ui.input_action_button("generate_btn", "Generate Plot", class_="btn-primary w-100 mt-3")
            )
        ),
        ui.column(8,
            ui.card(
                ui.card_header(
                    "Output Preview",
                    ui.download_button("download_plot", "Download High-Res PNG", class_="btn-sm btn-secondary float-end")
                ),
                # CHANGED: Using output_image instead of output_plot for better control
                ui.output_image("main_plot")
            )
        )
    ),
    
    ui.hr(),
    
    ui.row(
        ui.column(6, ui.h3("Tracks")),
        ui.column(6, ui.input_action_button("add_track_btn", "Add New Track", class_="btn-success float-end"))
    ),
    
    ui.div(id="tracks_container"),
    ui.output_text_verbatim("debug_info", placeholder=True)
)

# --- SERVER LOGIC ---

def server(input, output, session):
    track_indices = reactive.Value([0])
    counter = reactive.Value(1)
    
    # 1. Track Management
    def get_track_ui(i):
        general_formats = [".txt", ".tsv", ".csv", ".out", ".dat", ".tab"]
        return ui.div(
            ui.card(
                ui.card_header(f"Track #{i+1}"),
                ui.row(
                    ui.column(2, 
                        ui.input_text(f"name_{i}", "Name", value=f"Protein {i+1}"),
                        ui.input_text(f"chain_{i}", "Chain", value="A")
                    ),
                    ui.column(5, 
                        ui.h6("Disorder & Folding (FuzPred/AIU)"),
                        ui.input_file(f"aiu_{i}", "AIUPred Output", accept=general_formats),
                        ui.input_file(f"fuz_{i}", "FuzPred Scores", accept=general_formats),
                        ui.input_file(f"fuz_reg_{i}", "FuzPred Regions", accept=general_formats),
                    ),
                    ui.column(5,
                        ui.h6("Droplets & Aggregation (FuzDrop)"), 
                        ui.input_file(f"drop_{i}", "FuzDrop Scores", accept=general_formats),
                        ui.input_file(f"drop_reg_{i}", "FuzDrop Regions", accept=general_formats),
                    )
                ),
                class_="mb-3"
            ),
            id=f"track_wrapper_{i}"
        )

    @reactive.Effect
    def _init():
        ui.insert_ui(selector="#tracks_container", where="beforeEnd", ui=get_track_ui(0))

    @reactive.Effect
    @reactive.event(input.add_track_btn)
    def _add():
        new_idx = counter.get()
        ui.insert_ui(selector="#tracks_container", where="beforeEnd", ui=get_track_ui(new_idx))
        track_indices.set(track_indices.get() + [new_idx])
        counter.set(new_idx + 1)

    # 2. STEP A: Parsing Data
    # Only runs on "Generate" click.
    @reactive.Calc
    @reactive.event(input.generate_btn)
    def parsed_data():
        active_indices = track_indices.get()
        parsed_tracks = []
        chain_map = {}
        
        for i in active_indices:
            def get_path(input_id):
                try:
                    return input[input_id]()[0]["datapath"]
                except:
                    return None

            try:
                name = input[f"name_{i}"]()
                chain = input[f"chain_{i}"]()
            except:
                continue
            
            fuz_data = pm.parse_fuzpred(get_path(f"fuz_{i}"), get_path(f"fuz_reg_{i}")) 
            drop_data = pm.parse_fuzdrop(get_path(f"drop_{i}"), get_path(f"drop_reg_{i}"))

            track_obj = {
                'name': name,
                'chain': chain,
                'aiu': pm.parse_aiupred(get_path(f"aiu_{i}")),
                'fuz_scores': fuz_data['scores'],
                'fuz_regions': fuz_data['regions'],
                'drop_scores': drop_data['scores'],
                'drop_regions': drop_data['regions']
            }
            parsed_tracks.append(track_obj)
            chain_map[name] = chain

        interactions = {}
        struc_info = input.struc_file()
        if struc_info:
            interactions, _ = pm.parse_structure_contacts(struc_info[0]["datapath"], chain_map)
        
        return parsed_tracks, interactions

    # 3. STEP B: Plotting Calculation
    # Runs when Data is ready OR when Width/Height changes.
    @reactive.Calc
    def final_fig():
        # Get data (triggers parse if needed)
        tracks, interactions = parsed_data()
        
        # Get current dimensions
        try:
            w = float(input.plot_w())
            h = float(input.plot_h())
        except:
            w, h = 12, 8
        
        plt.close('all') 
        
        # We need a dummy path because the original script requires it,
        # but we will rely on plt.gcf() to get the object back.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dummy_path = tmp.name
            
        try:
            # Generate the plot
            pm.plot_integrated_modes(
                tracks, 
                interactions, 
                dummy_path, 
                split=input.split_mode()
            )
        finally:
            # Clean up the dummy file immediately, we don't need it
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
        
        # Retrieve figure from memory and apply dynamic size
        fig = plt.gcf()
        fig.set_size_inches(w, h)
        return fig

    # 4. Render Logic (Preview)
    # Using render.image prevents the 'function/float' error
    @render.image
    def main_plot():
        fig = final_fig()
        
        # Calculate pixel dimensions for preview (100 DPI is standard for screen)
        # This explicit math avoids the error you were seeing
        dpi = 100
        w_in = input.plot_w()
        h_in = input.plot_h()
        w_px = int(w_in * dpi)
        h_px = int(h_in * dpi)
        
        # Save to a temp file for the browser to display
        # Shiny automatically deletes this file after sending it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name
        
        # Use bbox_inches='tight' for cleaner look, or remove it for exact dimensions
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        return {
            "src": save_path,
            "width": w_px,
            "height": h_px,
            "mime_type": "image/png"
        }

    # 5. Download Logic
    @render.download(filename="protein_analysis.png")
    def download_plot():
        fig = final_fig()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name
        
        # 300 DPI for publication quality
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return save_path

app = App(app_ui, server)
