#!/usr/bin/env python3
"""
Example script showing how other applications can catch and parse the NFL prediction output
"""

import subprocess
import json
import re
import csv
import io
from xml.etree import ElementTree as ET

def run_nfl_predictions():
    """Run the NFL prediction system and capture output"""
    try:
        result = subprocess.run(['python', 'main.py', '-xgb'], 
                              capture_output=True, text=True, timeout=60)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "Timeout occurred", "Process timed out"
    except Exception as e:
        return "", str(e)

def parse_json_output(output):
    """Parse JSON output from the prediction system"""
    json_pattern = r'===JSON_OUTPUT_START===(.*?)===JSON_OUTPUT_END==='
    match = re.search(json_pattern, output, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
    return None

def parse_csv_output(output):
    """Parse CSV output from the prediction system"""
    csv_pattern = r'===CSV_OUTPUT_START===(.*?)===CSV_OUTPUT_END==='
    match = re.search(csv_pattern, output, re.DOTALL)
    
    if match:
        csv_str = match.group(1).strip()
        csv_reader = csv.DictReader(io.StringIO(csv_str))
        return list(csv_reader)
    return None

def parse_text_output(output):
    """Parse simple text output from the prediction system"""
    text_pattern = r'===TEXT_OUTPUT_START===(.*?)===TEXT_OUTPUT_END==='
    match = re.search(text_pattern, output, re.DOTALL)
    
    if match:
        text_str = match.group(1).strip()
        lines = text_str.split('\n')
        return [line.strip() for line in lines if line.strip()]
    return None

def parse_xml_output(output):
    """Parse XML output from the prediction system"""
    xml_pattern = r'===XML_OUTPUT_START===(.*?)===XML_OUTPUT_END==='
    match = re.search(xml_pattern, output, re.DOTALL)
    
    if match:
        xml_str = match.group(1).strip()
        try:
            root = ET.fromstring(xml_str)
            games = []
            for game in root.findall('game'):
                game_data = {
                    'home': game.get('home'),
                    'away': game.get('away'),
                    'recommended_bet': {
                        'team': game.find('recommended_bet').get('team'),
                        'type': game.find('recommended_bet').get('type'),
                        'spread': game.find('recommended_bet').get('spread'),
                        'confidence': game.find('recommended_bet').get('confidence')
                    }
                }
                games.append(game_data)
            return games
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            return None
    return None

def main():
    """Main function to demonstrate output parsing"""
    print("=" * 60)
    print("NFL PREDICTION OUTPUT PARSER DEMO")
    print("=" * 60)
    
    # Run the prediction system
    print("Running NFL prediction system...")
    stdout, stderr = run_nfl_predictions()
    
    if stderr:
        print(f"Error: {stderr}")
        return
    
    print("Prediction system completed successfully!")
    print("\n" + "=" * 60)
    print("PARSING OUTPUT IN DIFFERENT FORMATS")
    print("=" * 60)
    
    # Parse JSON output
    print("\n1. JSON OUTPUT:")
    json_data = parse_json_output(stdout)
    if json_data:
        print(f"Found {len(json_data)} games in JSON format")
        for game in json_data:
            print(f"  {game['game']}: {game['recommended_bet']['team']} {game['recommended_bet']['confidence']}%")
    else:
        print("  No JSON data found")
    
    # Parse CSV output
    print("\n2. CSV OUTPUT:")
    csv_data = parse_csv_output(stdout)
    if csv_data:
        print(f"Found {len(csv_data)} games in CSV format")
        for game in csv_data:
            print(f"  {game['Game']}: {game['Recommended_Team']} {game['Confidence']}%")
    else:
        print("  No CSV data found")
    
    # Parse Text output
    print("\n3. TEXT OUTPUT:")
    text_data = parse_text_output(stdout)
    if text_data:
        print(f"Found {len(text_data)} games in text format")
        for line in text_data:
            print(f"  {line}")
    else:
        print("  No text data found")
    
    # Parse XML output
    print("\n4. XML OUTPUT:")
    xml_data = parse_xml_output(stdout)
    if xml_data:
        print(f"Found {len(xml_data)} games in XML format")
        for game in xml_data:
            print(f"  {game['home']} vs {game['away']}: {game['recommended_bet']['team']} {game['recommended_bet']['confidence']}%")
    else:
        print("  No XML data found")
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES FOR OTHER APPLICATIONS")
    print("=" * 60)
    
    print("""
To integrate with other applications:

1. Python Application:
   ```python
   import subprocess
   result = subprocess.run(['python', 'main.py', '-xgb'], capture_output=True, text=True)
   # Parse using the functions above
   ```

2. PowerShell Script:
   ```powershell
   $output = python main.py -xgb
   # Extract data between markers
   ```

3. Batch File:
   ```batch
   python main.py -xgb > predictions.txt
   # Process predictions.txt
   ```

4. Web Application:
   ```python
   # Run as subprocess and parse JSON/XML output
   ```

5. Database Integration:
   ```python
   # Parse CSV output and insert into database
   ```
    """)

if __name__ == "__main__":
    main()
