import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from collections import defaultdict
import re
import ast
import math

def get_column_groups(columns):
    """Groups columns based on their base name (e.g., 'tau_cal')."""
    groups = defaultdict(list)
    for col in columns:
        match = re.match(r'([a-zA-Z0-9_]+_)\d+$', col)
        if match:
            base_name = match.group(1)
            groups[base_name].append(col)
    
    # Columns not matching the pattern go individually
    for col in columns:
        if col not in [c for lst in groups.values() for c in lst]:
            groups[col] = [col]
    return groups

def parse_series(series):
    """Parse series: returns a DataFrame if it's multidimensional, or Series if numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.DataFrame({series.name: series})
    new_data = {}
    for idx, v in enumerate(series):
        try:
            if isinstance(v, str):
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple)):
                    for i, val in enumerate(parsed):
                        new_data.setdefault(f"{series.name}[{i}]", []).append(val)
                else:
                    new_data.setdefault(series.name, []).append(parsed)
            else:
                new_data.setdefault(series.name, []).append(v)
        except:
            new_data.setdefault(series.name, []).append(None)
    return pd.DataFrame(new_data).apply(pd.to_numeric, errors='coerce')

def plot_data(csv_files):
    output_dir = 'output_plots'
    os.makedirs(output_dir, exist_ok=True)

    if not csv_files:
        print("No CSV files provided.")
        return

    # Load all CSVs first
    dfs = {}
    for f in csv_files:
        try:
            df = pd.read_csv(f).dropna(axis=1, how='all')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # remove unnamed cols
            dfs[f] = df
        except Exception as e:
            print(f"Could not read {f}: {e}")

    if not dfs:
        print("No valid CSV files to process.")
        return

    sample_df = list(dfs.values())[0]
    column_groups = get_column_groups(sample_df.columns)

    for group_name, columns in column_groups.items():
        num_columns = len(columns)
        if num_columns == 0:
            continue

        rows = math.ceil(math.sqrt(num_columns))
        cols = math.ceil(num_columns / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        fig.suptitle(f'Comparison for {group_name}*', fontsize=16)
        axes = axes.flatten() if num_columns > 1 else [axes]

        for i, col_name in enumerate(columns):
            ax = axes[i]
            plotted = False
            for csv_file, df in dfs.items():
                if col_name not in df.columns:
                    continue
                parsed_df = parse_series(df[col_name])
                for sub_col in parsed_df.columns:
                    y_values = parsed_df[sub_col].to_numpy().flatten()  # 转成一维
                    if len(y_values) == 0 or all(pd.isna(y_values)):
                        continue
                    x_values = parsed_df.index.values  # 转成一维
                    ax.plot(x_values, y_values, label=f"{os.path.basename(csv_file)}-{sub_col}")
                    plotted = True
            ax.set_title(col_name)
            ax.grid(True)
            if plotted:
                ax.legend(fontsize=8)

        for i in range(num_columns, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        sanitized_group_name = group_name.replace('/', '_').replace('\\', '_').rstrip('_')
        output_filename = os.path.join(output_dir, f'{sanitized_group_name}_comparison.png')
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"Saved plot to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize and compare data from multiple CSV files.')
    parser.add_argument('csv_files', nargs='+', help='List of CSV files to process.')
    args = parser.parse_args()

    plot_data(args.csv_files)

