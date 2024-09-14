import os
import pandas as pd
import argparse
import time
from kolar.model import VisionaryKolar

# Initialize the model and global variables
kolar = VisionaryKolar()
inference_times = []

def batch_predictor(image_links, entities):
    '''
    Call batch_predictions and handle the model's output
    '''
    start_time = time.time()
    batch_predictions = kolar.batch_predictions(image_links, entities)
    inference_times.append(time.time() - start_time)
    
    print(f"Batch predictions: {batch_predictions}")
    return batch_predictions

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process some images in batches.")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of instances to process in a batch')
    args = parser.parse_args()
    
    # Load the dataset
    DATASET_FOLDER = 'dataset'
    input_filepath = os.path.join(DATASET_FOLDER, args.input_file)
    
    test = pd.read_csv(input_filepath)
    
    # Show number of rows and ask for index range
    num_rows = len(test)
    print(f"The dataset contains {num_rows} rows.")
    
    start_index = int(input(f"Enter the start index (0 - {num_rows - 1}): "))
    end_index = int(input(f"Enter the end index ({start_index} - {num_rows - 1}): "))
    
    # Validate the input range
    if start_index < 0 or end_index >= num_rows or start_index > end_index:
        print("Invalid index range. Exiting.")
        exit(1)
    
    # Process the selected range
    selected_test = test.iloc[start_index:end_index+1]
    
    # Prepare for batch processing
    image_links = selected_test['image_link'].tolist()
    entities = selected_test['entity_name'].tolist()
    batch_size = args.batch_size
    
    predictions = []
    
    # Process in batches
    for i in range(0, len(image_links), batch_size):
        batch_links = image_links[i:i+batch_size]
        batch_entities = entities[i:i+batch_size]
        
        # Get batch predictions
        batch_predictions = batch_predictor(batch_links, batch_entities)
        
        # Append the predictions to the overall list
        predictions.extend(batch_predictions)
    
    # Add predictions to the DataFrame
    selected_test['prediction'] = predictions
    
    # Save the result to the file
    output_filename = os.path.join(DATASET_FOLDER, f'kolar_out_{start_index}_{end_index}.csv')
    selected_test[['index', 'prediction']].to_csv(output_filename, index=False)

    # Output statistics
    total_instances = len(predictions)
    print(f"Results saved to {output_filename}")
    print("Average inference time: {:.2f} seconds".format(sum(inference_times) / len(inference_times)))
    print(f"Total instances processed: {total_instances}")
