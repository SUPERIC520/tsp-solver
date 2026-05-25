import pandas as pd
import numpy as np
import os

def update_best_tour(best_tour_file: str, new_tour: np.ndarray, new_length: float):
    # If the file doesn't exist, create it with the new tour
    # If it exists, read it, compare, and update if better
    
    # We'll save with columns ['tour', 'length']
    # 'tour' will be a string representation of the array
    
    tour_str = ",".join(map(str, new_tour.tolist()))
    new_data = pd.DataFrame({'tour': [tour_str], 'length': [new_length]})
    
    if os.path.exists(best_tour_file):
        df = pd.read_csv(best_tour_file)
        if not df.empty:
            current_best = df.iloc[0]['length']
            if new_length < current_best:
                new_data.to_csv(best_tour_file, index=False)
                return True
            return False
    
    new_data.to_csv(best_tour_file, index=False)
    return True
