import pandas as pd
import torch
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import multiprocessing
from multiprocessing import get_context
import os

class LargeCSVSentimentAnalyzer:
    def __init__(self,
                 model_name='distilbert-base-uncased-finetuned-sst-2-english',
                 batch_size=1024,
                 output_path='./sentiment_analyzed.csv'):
        """
        Initialize sentiment analyzer for large CSV files with CUDA multiprocessing support

        Args:
            model_name (str): Transformer model for sentiment analysis
            batch_size (int): Number of rows to process in each batch
            output_path (str): Path to save the analyzed CSV
        """
        # Multiprocessing context
        multiprocessing.set_start_method('spawn', force=True)

        # Configuration
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_path = output_path

        # Number of CPU cores
        self.num_workers = max(1, multiprocessing.cpu_count() - 2)

    def _sentiment_worker(self, texts):
        """
        Process texts in a separate process to handle CUDA multiprocessing

        Args:
            texts (list): Batch of texts to analyze

        Returns:
            list: Sentiment analysis results
        """
        # Detect device within the worker process
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer in each worker
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Create sentiment pipeline
        sentiment_pipeline = pipeline(
            'sentiment-analysis',
            model=model,
            tokenizer=tokenizer,
            device=0 if device.type == 'cuda' else -1
        )

        # Skip empty or NaN texts
        filtered_texts = [str(text) for text in texts if pd.notna(text) and text != '']

        # Perform sentiment analysis
        if not filtered_texts:
            return [{'label': 'NEUTRAL', 'score': 0.5}] * len(texts)

        results = sentiment_pipeline(filtered_texts)

        # Pad results to match original batch size
        full_results = []
        result_idx = 0
        for original_text in texts:
            if pd.isna(original_text) or original_text == '':
                full_results.append({'label': 'NEUTRAL', 'score': 0.5})
            else:
                full_results.append(results[result_idx])
                result_idx += 1

        return full_results

    def analyze_csv(self, input_path):
        """
        Analyze sentiment for a large CSV file

        Args:
            input_path (str): Path to input CSV file
        """
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Read CSV in chunks to manage memory
        csv_reader = pd.read_csv(input_path, chunksize=self.batch_size)

        # Open output file in write mode
        with open(self.output_path, 'w') as output_file:
            first_chunk = True

            # Process each chunk
            for chunk in csv_reader:
                # Ensure 'text' column exists
                if 'text' not in chunk.columns:
                    raise ValueError("CSV must contain a 'text' column")

                # Use spawn context for multiprocessing
                ctx = get_context('spawn')

                # Process sentiment for current chunk
                with ctx.Pool(processes=self.num_workers) as pool:
                    # Split texts into batches
                    text_batches = np.array_split(chunk['text'], self.num_workers)

                    # Submit processing tasks
                    results = pool.map(self._sentiment_worker, text_batches)

                    # Flatten results
                    all_results = [item for sublist in results for item in sublist]

                # Add sentiment columns
                chunk['sentiment_label'] = [result['label'] for result in all_results]
                chunk['sentiment_score'] = [result['score'] for result in all_results]

                # Write to output file
                chunk.to_csv(output_file, mode='a', header=first_chunk, index=False)
                first_chunk = False

        print(f"Sentiment analysis complete. Results saved to {self.output_path}")

        # Return some basic stats
        return {
            'output_file': self.output_path,
            'processing_device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        }

def main():
    # Initialize analyzer
    analyzer = LargeCSVSentimentAnalyzer(
        batch_size=1024,  # Adjust based on your system's memory
        output_path='./sentiment_analyzed_1mLines.csv'
    )

    # Analyze the CSV
    try:
        results = analyzer.analyze_csv('./1MLines.csv')
        print("Analysis Results:")
        print(results)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
