import json
import logging
from dataclasses import dataclass, fields
import textwrap
import torch
from typing import Tuple, Any
from torch import nn
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from pandas.io.formats.style import Styler

### logging ###

MAX_OUTPUT_WIDTH = 80

class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def log_bold(text, color):
  logging.info(Color.BOLD + color + text + Color.END)

def log_long_text(text):
  logging.info(textwrap.fill(text, width=MAX_OUTPUT_WIDTH))

def log_seperator():
  logging.info("#" * MAX_OUTPUT_WIDTH)

def log_allocated_memory_GB(device):
  GB = 1024**3
  if device == 'mps':
    logging.info(f"MPS: allocated memory = {torch.mps.current_allocated_memory() / GB: .2f} GB")
  elif device == 'cuda':
    logging.info(f"Cuda: allocated memory = {torch.cuda.memory_allocated() / GB: .2f} GB")
  else:
    logging.info("no memory information available.")

def logging_decorator(description=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"{description}...")
            result = func(*args, **kwargs)
            logging.info(f"{description} done.")
            return result
        return wrapper
    return decorator

### config ###

@dataclass
class DatasetConfig:
    name: str
    good_rating_threshold: float
    rating_edge_name: Tuple

@dataclass
class ModelConfig:
    with_neural_network: bool
    num_of_learned_atoms: int
    num_of_horn_clauses: int

@dataclass
class TrainingConfig:
    learning_rates: dict[float]
    num_of_epochs: int
    batch_size: int
    loss_fn: Any
    l1_lambda: float

@dataclass
class EvaluationConfig:
    val_size: float
    test_size: float
    lightgcn_lr: float
    lightgcn_epochs: int

@dataclass
class Config:
    data: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    results_path: str
    use_final_testset: bool

    def __repr__(self):
        def format_dataclass(obj):
            field_values = ((field.name, getattr(obj, field.name)) for field in fields(obj))
            formatted_fields = (f"{name}={value!r}" if isinstance(value, str) else f"{name}={value}" for name, value in field_values)
            return ",\n    ".join(formatted_fields)

        return (
            f"Config(\n"
            f"  data=DatasetConfig(\n    {format_dataclass(self.data)}\n  ),\n"
            f"  model=ModelConfig(\n    {format_dataclass(self.model)}\n  ),\n"
            f"  training=TrainingConfig(\n    {format_dataclass(self.training)}\n  ),\n"
            f"  evaluation=EvaluationConfig(\n    {format_dataclass(self.evaluation)}\n  )\n"
            f"  results_path='{self.results_path}'\n"
            f"  use_final_testset='{self.use_final_testset}'\n"
            f")"
        )

def get_loss_function(loss_fn_name):
        # Directly map the loss function name to the PyTorch class
        loss_functions = {
            "MSELoss": nn.MSELoss,
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "BCELoss": nn.BCELoss,
            "L1Loss": nn.L1Loss
        }
        return loss_functions[loss_fn_name]()

@logging_decorator("Load config file")
def load_config(config_path='config/config.json'):
    with open(config_path, 'r') as file:
        json_data = json.load(file)
    # NOTE: handling of special types
    json_data['data']["rating_edge_name"] = tuple(json_data['data']["rating_edge_name"])
    json_data['training']["loss_fn"] = get_loss_function(json_data['training']["loss_fn"])
    config = Config(
        data=DatasetConfig(**json_data['data']),
        model=ModelConfig(**json_data['model']),
        training=TrainingConfig(**json_data['training']),
        evaluation=EvaluationConfig(**json_data['evaluation']),
        results_path=json_data['results_path'],
        use_final_testset=json_data['use_final_testset']
    )
    # alter the results path based on evaluation
    if config.use_final_testset:
        config.results_path = os.path.join(config.results_path, "final")
    else:
        config.results_path = os.path.join(config.results_path, "validation")
    logging.info(f"config={config}")
    return config

### visualisation ###
def highlight_values(s):
    return [
        f"background-color: white; color: rgba(0, 0, 0, {v})"
        for v in s
    ]

def highlight_max(s):
    def extract_value(s):
        return float(s.split(' ± ')[0])
    values = s.apply(extract_value)
    is_max = values == values.max()
    return ['font-weight: bold' if v else '' for v in is_max]

def store_df_as_html(df_or_styler, html_path):
    styles = [
        {
            'selector': '',
            'props': [
                ('border-collapse', 'collapse'),
                ('border', '1px solid black'),
                #('width', '100%')
            ]
        },
        {
            'selector': 'th, td',
            'props': [
                ('border', '1px solid black'),
                ('padding', '8px'),
                ('text-align', 'center')
            ]
        }
    ]

    if isinstance(df_or_styler, Styler):
        styled_df = df_or_styler.set_table_styles(styles)
    else:
        styled_df = df_or_styler.style.set_table_styles(styles)

    with open(html_path, "w") as file:
        file.write(styled_df.to_html())

def store_df_as_latex(df_or_styler, latex_path):
    if isinstance(df_or_styler, Styler):
        df_or_styler = df_or_styler.data

    with open(latex_path, "w") as file:
        # only write the data to tex without styling
        latex_table = f"""\\begin{{table}}
\\caption{{COMBAK}}
\\label{{tab:COMBAK}}
{df_or_styler.to_latex()}\\end{{table}}"""
        file.write(latex_table)

def plot_and_save_loss(train_losses, val_losses, filename):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, linestyle='-', label='Train Loss')
    plt.plot(val_losses, linestyle='-', label='Evaluation Loss')

    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(filename, format='pdf')
    plt.close()

@dataclass
class RuleFlow:
    origin: np.ndarray
    width: float
    positions: np.ndarray

    @property
    def x_max(self):
        return np.max(self.positions[:, 0])
    @property
    def x_min(self):
        return np.min(self.positions[:, 0])
    @property
    def y_max(self):
        return np.max(self.positions[:, 1])
    @property
    def y_min(self):
        return np.min(self.positions[:, 1])

class Flowchart:
    def __init__(self, rules, weights):
        # automatically add new lines for long texts
        max_line_length = 25
        self.rules = [[textwrap.fill(variable, width=max_line_length) for variable in rule] for rule in rules]
        self.weights = weights

    def draw(self, filepath):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        head_length = 1
        arrow_spacing_end = 0.75
        height = 12
        ver_spacing = 6
        hor_spacing = 2
        width = 30
        fontsize = 12
        yes_xy = [3, -1]
        start_opacity = 0.5

        rule_flows = []

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # matplotlib default_colors

        # Draw nodes as diamonds
        positions = []
        for r_i, rule in enumerate(self.rules):
            color = default_colors[r_i % len(default_colors)]
            num_of_var = len(rule)
            if r_i == 0:
                rule_origin = np.array([0, 0])
            else:
                last_rule_flow = rule_flows[r_i-1]
                rule_origin = last_rule_flow.origin + np.array([last_rule_flow.width + hor_spacing, 0])
            # Draw text above flow
            ax.text(rule_origin[0], rule_origin[1] + height/2 + 4, f'Rule {r_i+1}', fontsize=2*fontsize, ha='center', va='center')
            positions.append(np.zeros((num_of_var, 2)))
            for v_i, variable in enumerate(rule):
                weight = self.weights[r_i][v_i]
                rgb = tuple(int(color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
                rgba = rgb + (start_opacity*weight,)
                pos = rule_origin.copy()
                pos += [0, -v_i * (height + ver_spacing)]
                positions[r_i][v_i] = pos

                x, y = pos[0], pos[1]

                diamond = patches.Polygon([
                                [x, y - height/2],
                                [x + width/2, y],
                                [x, y + height/2],
                                [x - width/2, y]
                            ], closed=True, facecolor=rgba, edgecolor='black')
                ax.add_patch(diamond)
                ax.text(x, y, variable, fontsize=fontsize, ha='center', va='center')

                # Draw edges
                if v_i > 0:
                    start = positions[r_i][v_i - 1] - [0, height/2]
                    end = positions[r_i][v_i] + [0, height/2 + head_length]
                    dx, dy = end[0] - start[0], end[1] - start[1] + arrow_spacing_end
                    ax.arrow(start[0], start[1], dx, dy, head_width=head_length, head_length=head_length, fc='black', ec='black')
                    ax.text(start[0]+yes_xy[0], start[1]+yes_xy[1], 'yes', fontsize=fontsize, ha='center', va='center')
            rule_flows.append(RuleFlow(rule_origin, width, positions[r_i]))

        # draw recommend rectangle
        rect_height = 6
        def draw_rectangle(x, y):
            final_text = 'Recommend'
            width = len(final_text) + 6
            rectangle = patches.Rectangle((x - width/2, y - rect_height/2), width, rect_height, facecolor='white', edgecolor='black')
            ax.text(x, y, final_text, fontsize=fontsize, ha='center', va='center')
            ax.add_patch(rectangle)
            return width

        x_max = np.max([r.x_max for r in rule_flows])
        x_min = np.min([r.x_min for r in rule_flows])
        x_middle = (x_max - x_min) / 2
        y_min = np.min([r.y_min for r in rule_flows])
        rectangle_xy = np.array([x_middle, y_min - height/2 - 3 * ver_spacing])
        draw_rectangle(rectangle_xy[0], rectangle_xy[1])

        # draw final lines
        anchor = np.array(rectangle_xy) + [0, rect_height/2 + ver_spacing]
        for rule_flow in rule_flows:
            # Define the path for the arrow with corners
            vertices = [(rule_flow.origin[0], rule_flow.y_min - height/2), (rule_flow.origin[0], anchor[1]), anchor]  # Corner points
            arrow = patches.FancyArrowPatch(path=Path(vertices), arrowstyle='-', color='black')
            ax.add_patch(arrow)
            ax.text(rule_flow.origin[0]+yes_xy[0], rule_flow.y_min - height/2+yes_xy[1], 'yes', fontsize=fontsize, ha='center', va='center')
        ax.arrow(anchor[0], anchor[1], 0, -ver_spacing + (head_length + arrow_spacing_end), head_width=head_length, head_length=head_length, fc='black', ec='black')

        x_range = np.max([r.x_max for r in rule_flows]) - np.min([r.x_min for r in rule_flows])
        y_range = np.max([r.y_max for r in rule_flows]) - np.min([r.y_min for r in rule_flows])
        fig_width = 4 + x_range * 0.1
        fig_height = 4 + y_range * 0.1
        fig.set_size_inches((fig_width, fig_height))

        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)

def store_flowchart(rules, weights, filepath):
    flowchart = Flowchart(rules, weights)
    flowchart.draw(filepath)

### Miscellaneous ###

@logging_decorator("Get pytorch device")
def get_pytorch_device():
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logging.info(f"{device=}")
    return device

@logging_decorator("Check if results folder exists")
def check_results_folder_exists(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        logging.info(f"Created results folder at: {results_path}")
    else:
        logging.info(f"Results folder already exists at: {results_path}")
