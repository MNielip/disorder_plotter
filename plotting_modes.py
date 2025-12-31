import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import sys
import os

# --- 1. CONFIGURATION & CONSTANTS ---

REGION_COLORS = {
    'Disorder-to-Disorder region': '#d73027', # Red (DD)
    'Disorder-to-Order region': '#fc8d59',    # Orange (DO)
    'Context-dependent region': '#fee090',    # Yellow
    'Ordered-to-Ordered region': '#4575b4',   # Blue (OO)
    'Droplet-promoting region': '#984ea3',    # Purple (FuzDrop)
    'Aggregation hot-spot': '#e7298a',        # Magenta (FuzDrop Aggregation)
}

LEGEND_LABELS = {
    'Disorder-to-Disorder region': 'DD (Fuzzy)',
    'Disorder-to-Order region': 'DO (Folding)',
    'Context-dependent region': 'Context-dep',
    'Ordered-to-Ordered region': 'OO (Static)',
    'Droplet-promoting region': 'Droplet (LLPS)',
    'Aggregation hot-spot': 'Aggreg. Hotspot'
}

# --- 2. DATA LOADING MODES ---

def parse_aiupred(filepath):
    """ Parses AIUPred output files. """
    if not filepath or not os.path.exists(filepath):
        return None
    data = []
    print(f"[AIUPred] Loading {filepath}...")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        data.append([int(parts[0]), parts[1], float(parts[2]), float(parts[3])])
                    except ValueError: continue
        return pd.DataFrame(data, columns=['Position', 'Residue', 'Disorder', 'ANCHOR2'])
    except Exception as e:
        print(f"[Error] Failed to parse AIUPred file: {e}")
        return None

def parse_fuzpred(score_path, region_path=None):
    """ Parses FuzPred score files and optional region files. """
    result = {'scores': None, 'regions': None}
    
    if score_path and os.path.exists(score_path):
        print(f"[FuzPred] Loading Scores {score_path}...")
        try:
            df = pd.read_csv(score_path, sep='\t')
            df.columns = [c.lower().strip() for c in df.columns] 
            result['scores'] = df
        except Exception as e:
            print(f"[Error] Failed to parse FuzPred scores: {e}")

    if region_path and os.path.exists(region_path):
        print(f"[FuzPred] Loading Regions {region_path}...")
        try:
            df = pd.read_csv(region_path, sep='\t')
            df.columns = [c.lower().strip() for c in df.columns]
            rename_map = {'class': 'type', 'classification': 'type', 'region_type': 'type'}
            df = df.rename(columns=rename_map)
            if 'type' in df.columns:
                df['type'] = df['type'].astype(str).str.strip()
            result['regions'] = df
        except Exception as e:
            print(f"[Error] Failed to parse FuzPred regions: {e}")
            
    return result

def parse_fuzdrop(score_path, region_path=None):
    """ Parses FuzDrop score files and optional region files. """
    result = {'scores': None, 'regions': None}

    # 1. Scores
    if score_path and os.path.exists(score_path):
        print(f"[FuzDrop] Loading Scores {score_path}...")
        try:
            # Try reading with header
            df = pd.read_csv(score_path, sep=r'\s+', comment='#')
            col_map = {c.lower(): c for c in df.columns}
            
            # Identify Position Column
            pos_col = col_map.get('position', None)
            if not pos_col and 'residue' in col_map: pos_col = 'Residue' # Fallback
            
            # Identify pDP (or pLLPS) Column
            pdp_col = None
            if 'pdp' in col_map: pdp_col = col_map['pdp']
            elif 'pllps' in col_map: pdp_col = col_map['pllps']
            elif 'prob' in col_map: pdp_col = col_map['prob']

            # Identify Sbind Column
            sbind_col = None
            if 'sbind' in col_map: sbind_col = col_map['sbind']
            elif 's_bind' in col_map: sbind_col = col_map['s_bind']

            # Fallback if no headers found (Standard 3 column: Pos, Res, Prob)
            if not pdp_col: 
                 df = pd.read_csv(score_path, sep=r'\s+', header=None, comment='#')
                 if df.shape[1] >= 3:
                     df = df.rename(columns={0: 'Position', 1: 'Residue', 2: 'pDP'})
                     pdp_col = 'pDP'
                     pos_col = 'Position'

            # Rename and Extract
            rename_dict = {}
            if pos_col: rename_dict[pos_col] = 'Position'
            if pdp_col: rename_dict[pdp_col] = 'pDP'
            if sbind_col: rename_dict[sbind_col] = 'Sbind'
            
            df = df.rename(columns=rename_dict)
            
            # Normalize Sbind if it exists
            if 'Sbind' in df.columns:
                df['Sbind'] = pd.to_numeric(df['Sbind'], errors='coerce')
                min_v = df['Sbind'].min()
                max_v = df['Sbind'].max()
                if max_v > min_v:
                    df['Sbind'] = (df['Sbind'] - min_v) / (max_v - min_v)
                else:
                    df['Sbind'] = 0.0

            if 'pDP' in df.columns:
                df['pDP'] = pd.to_numeric(df['pDP'], errors='coerce')
            
            cols_to_keep = ['Position', 'pDP']
            if 'Sbind' in df.columns: cols_to_keep.append('Sbind')
            
            result['scores'] = df[cols_to_keep].dropna()

        except Exception as e:
            print(f"[Error] Failed to parse FuzDrop scores: {e}")

    # 2. Regions
    if region_path and os.path.exists(region_path):
        print(f"[FuzDrop] Loading Regions {region_path}...")
        try:
            try:
                df = pd.read_csv(region_path, sep='\t')
                if len(df.columns) < 2: raise ValueError 
            except:
                df = pd.read_csv(region_path, sep=r'\s+')

            df.columns = [c.lower().strip() for c in df.columns]
            
            if 'start' in df.columns and 'end' in df.columns:
                # Standardize 'type' column
                if 'type' not in df.columns and 'classification' in df.columns:
                    df['type'] = df['classification']
                elif 'type' not in df.columns:
                     df['type'] = 'Droplet-promoting region' # Default
                
                # Clean strings
                df['type'] = df['type'].astype(str).str.strip()
                result['regions'] = df
            else:
                print(f"[Warning] FuzDrop regions file missing 'start'/'end' columns.")
        except Exception as e:
            print(f"[Error] Failed to parse FuzDrop regions: {e}")

    return result

def parse_structure_contacts(filepath, chain_map, cutoff=3.5):
    """ Parses PDB or CIF files to find contact residues between chains. """
    if not filepath or not os.path.exists(filepath):
        return {}, []
    print(f"[Structure] Parsing interactions from {filepath}...")
    chain_coords = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    parts = line.split()
                    if len(parts) > 12 and '.' in parts[10]: # Likely CIF
                        try:
                            c_id, r_seq = parts[6], int(parts[8])
                            coords = np.array([float(parts[10]), float(parts[11]), float(parts[12])])
                            chain_coords.setdefault(c_id, {}).setdefault(r_seq, []).append(coords)
                        except: continue
                    elif len(line) > 54: # PDB lines
                        try:
                            c_id = line[21].strip()
                            r_seq = int(line[22:26])
                            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                            chain_coords.setdefault(c_id, {}).setdefault(r_seq, []).append(np.array([x, y, z]))
                        except: continue
    except Exception as e:
        print(f"[Error] Structure parsing failed: {e}")
        return {}, []

    interaction_sets = {} 
    contact_pairs = []
    target_chains = list(chain_map.values())
    
    for i in range(len(target_chains)):
        c1 = target_chains[i]
        if c1 not in chain_coords: continue
        interaction_sets.setdefault(c1, set())
        for j in range(i + 1, len(target_chains)):
            c2 = target_chains[j]
            if c2 not in chain_coords: continue
            interaction_sets.setdefault(c2, set())
            for r1, atoms1 in chain_coords[c1].items():
                for r2, atoms2 in chain_coords[c2].items():
                    dist_min = 999
                    for a1 in atoms1:
                        for a2 in atoms2:
                            d = np.linalg.norm(a1 - a2)
                            if d < dist_min: dist_min = d
                    if dist_min < cutoff:
                        contact_pairs.append(((c1, r1), (c2, r2)))
                        interaction_sets[c1].add(r1)
                        interaction_sets[c2].add(r2)
    return interaction_sets, contact_pairs

# --- 3. PLOTTING LOGIC ---

def draw_regions(ax, df_regions, y_pos=1.03, height=0.06):
    """Helper to draw colored bars for regions."""
    if df_regions is None or df_regions.empty:
        return
    for _, row in df_regions.iterrows():
        try:
            start, end = int(row['start']), int(row['end'])
            r_type = row.get('type', '')
            color = REGION_COLORS.get(r_type, 'gray')
            ax.add_patch(patches.Rectangle((start, y_pos), end - start + 1, height, 
                                            facecolor=color, edgecolor='none', alpha=0.9, clip_on=False))
        except: continue

def plot_integrated_modes(tracks, structure_interactions, output_filename, split=False):
    """ Plots tracks with a single unified legend on the right. """
    n_tracks = len(tracks)
    if n_tracks == 0:
        print("No tracks to plot.")
        return

    rows_per_track = 2 if split else 1
    total_rows = n_tracks * rows_per_track
    
    Y_LIMIT = 1.25
    
    fig, axes = plt.subplots(total_rows, 1, figsize=(16, 5 * total_rows), squeeze=False)
    axes = axes.flatten()
    
    legend_collector = {}
    def add_leg(label, handle):
        if label not in legend_collector: legend_collector[label] = handle

    for i, track in enumerate(tracks):
        name = track.get('name', f'Protein_{i}')
        chain = track.get('chain', '')
        
        if split:
            ax_fold = axes[i * 2]      
            ax_disorder = axes[i * 2 + 1] 
            current_axes = [ax_fold, ax_disorder]
            titles = [f"{name} (Chain {chain}) - Folding/Binding", f"{name} (Chain {chain}) - Disorder/Droplets"]
        else:
            ax_main = axes[i]
            current_axes = [ax_main]
            titles = [f"{name} (Chain {chain}) - Combined"]

        # --- A. PLOT LINES ---

        # 1. AIUPred
        if track.get('aiu') is not None:
            df = track['aiu']
            for ax in current_axes:
                line, = ax.plot(df['Position'], df['ANCHOR2'], color='red', label='ANCHOR2', lw=1.5, alpha=0.8)
                add_leg('ANCHOR2', line)
            
            target_ax = current_axes[-1]
            line, = target_ax.plot(df['Position'], df['Disorder'], color='orange', label='Disorder', lw=0.8, ls='--')
            add_leg('Disorder', line)

        # 2. FuzPred Scores
        if track.get('fuz_scores') is not None:
            df = track['fuz_scores']
            if 'pdo' in df.columns:
                target_ax = current_axes[0] 
                line, = target_ax.plot(df['position'], df['pdo'], color='blue', label='pDO (Folding)', ls='--', lw=1.5)
                add_leg('pDO (Folding)', line)
            if 'pdd' in df.columns:
                target_ax = current_axes[-1]
                line, = target_ax.plot(df['position'], df['pdd'], color='green', label='pDD (Fuzzy)', ls=':', lw=1.5)
                add_leg('pDD (Fuzzy)', line)

        # 3. FuzDrop Scores (pDP and Sbind)
        if track.get('drop_scores') is not None:
            df = track['drop_scores']
            target_ax = current_axes[-1] # Bottom/Main axis
            
            # Plot pDP (Droplet Probability)
            if 'pDP' in df.columns:
                line, = target_ax.plot(df['Position'], df['pDP'], color='purple', label='pDP (Droplet)', lw=1.5, alpha=0.7)
                add_leg('pDP (Droplet)', line)
            
            # Plot Sbind (Binding Probability, Normalized)
            if 'Sbind' in df.columns:
                line, = target_ax.plot(df['Position'], df['Sbind'], color='cyan', label='Sbind (Binding)', lw=1.2, ls='-.')
                add_leg('Sbind (Binding)', line)

        # --- B. PLOT CONTEXT ---
        for ax_idx, ax in enumerate(current_axes):
            # Interactions
            if chain in structure_interactions and structure_interactions[chain]:
                for r in structure_interactions[chain]:
                    ax.add_patch(patches.Rectangle((r - 0.5, 0), 1, Y_LIMIT, facecolor='black', alpha=0.15, zorder=0))
                patch = patches.Patch(facecolor='black', alpha=0.15, label='Interaction')
                add_leg('Interaction', patch)

            # FuzPred Regions (Lower Line)
            if track.get('fuz_regions') is not None:
                draw_regions(ax, track['fuz_regions'], y_pos=1.02, height=0.06)
                for r_type in track['fuz_regions']['type'].unique():
                    if r_type in REGION_COLORS and r_type in LEGEND_LABELS:
                        patch = patches.Patch(color=REGION_COLORS[r_type], label=LEGEND_LABELS[r_type])
                        add_leg(LEGEND_LABELS[r_type], patch)

            # FuzDrop Regions (Upper Line: Droplets AND Aggregations)
            if track.get('drop_regions') is not None:
                draw_regions(ax, track['drop_regions'], y_pos=1.10, height=0.06)
                for r_type in track['drop_regions']['type'].unique():
                    if r_type in REGION_COLORS and r_type in LEGEND_LABELS:
                         patch = patches.Patch(color=REGION_COLORS[r_type], label=LEGEND_LABELS[r_type])
                         add_leg(LEGEND_LABELS[r_type], patch)

            ax.set_ylabel("Probability")
            ax.set_ylim(0, Y_LIMIT)
            ax.set_title(titles[ax_idx])
            ax.grid(True, linestyle=':', alpha=0.4)

    # --- C. LEGEND ---
    # Custom Order
    priority = ['ANCHOR2', 'Disorder', 'pDO (Folding)', 'pDD (Fuzzy)', 'pDP (Droplet)', 'Sbind (Binding)']
    final_handles = []
    final_labels = []
    
    for p in priority:
        if p in legend_collector:
            final_handles.append(legend_collector[p])
            final_labels.append(p)
            del legend_collector[p]
            
    for k, v in legend_collector.items():
        final_handles.append(v)
        final_labels.append(k)

    fig.legend(final_handles, final_labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize='medium', title="Legend")
    plt.subplots_adjust(right=0.85, hspace=0.4)
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename}")


# --- 4. MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(description="Robust Protein Plotter v7")
    parser.add_argument('--struc', help="Path to PDB/CIF structure file.")
    parser.add_argument('--output', default='protein_analysis_plot.png', help="Output filename.")
    parser.add_argument('--split', action='store_true', help="Split pDO (Folding) and pDD (Disorder) into separate plots.")
    parser.add_argument('--track', action='append', help="Define track: name=X,chain=Y,aiu=path,fuz=path,regions=path,drop=path,drop_regions=path")
    args = parser.parse_args()

    tracks = []
    chain_map = {} 

    if args.track:
        print("Parsing CLI arguments...")
        for track_str in args.track:
            config = {}
            for item in track_str.split(','):
                if '=' in item:
                    k, v = item.split('=', 1)
                    config[k.strip()] = v.strip()
            
            fuz_regions = parse_fuzpred(None, config.get('regions')).get('regions')
            fuz_drop_res = parse_fuzdrop(config.get('drop'), config.get('drop_regions'))

            track_data = {
                'name': config.get('name', 'Unknown'),
                'chain': config.get('chain', 'A'),
                'aiu': parse_aiupred(config.get('aiu')),
                'fuz_scores': parse_fuzpred(config.get('fuz')).get('scores'),
                'fuz_regions': fuz_regions,
                'drop_scores': fuz_drop_res.get('scores'),
                'drop_regions': fuz_drop_res.get('regions')
            }
            tracks.append(track_data)
            chain_map[track_data['name']] = track_data['chain']

    if not tracks:
        print("No CLI arguments found. Using manual configuration.")
        t1 = {
            'name': 'CBP', 'chain': 'B',
            'aiu': parse_aiupred('AIUPred_CBP.txt'),
            'fuz_scores': parse_fuzpred('FuzPred_scores_CBP.tsv')['scores'],
            'fuz_regions': parse_fuzpred(None, 'FuzPred_regions_CBP.tsv')['regions'],
            'drop_scores': None,
            'drop_regions': None
        }
        tracks = [t1]
        chain_map = {'CBP': 'B'}
        if not args.struc: args.struc = 'fold_model.cif'

    interactions = {}
    if args.struc and os.path.exists(args.struc):
        interactions, _ = parse_structure_contacts(args.struc, chain_map)

    print(f"Generating plot for {len(tracks)} tracks (Split Mode: {args.split})...")
    plot_integrated_modes(tracks, interactions, args.output, split=args.split)

if __name__ == "__main__":
    main()
