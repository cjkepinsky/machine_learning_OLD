from libs.simple_analyzer import print_analytics, print_simple_correlations
from libs.simpleplotter import simple_heatmap, simple_correlations
from constants import source_path, target_name
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

DATA = pd.read_csv(source_path)

print_analytics(DATA, target_name)