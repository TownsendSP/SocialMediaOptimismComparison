import json
from langdetect import detect
import langdetect.lang_detect_exception

def filter_english_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                if detect(data.get('text', '')) == 'en':
                    outfile.write(line)
            except (json.JSONDecodeError, langdetect.lang_detect_exception.LangDetectException):
                continue

# Replace 'input.jsonl' and 'output.jsonl' with your file paths
filter_english_lines('input.jsonl', 'output.jsonl')