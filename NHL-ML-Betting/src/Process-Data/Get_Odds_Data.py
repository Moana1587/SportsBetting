"""
NHL Odds Data Processor

This script reads NHL odds data from OddsData.csv and stores it in a SQLite database.
It processes the data by season, cleans and validates the data, converts team names
to full team names, and creates separate database tables for each season.

No web scraping functionality - purely CSV to database processing.
"""

import os
import sqlite3
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

class NHLOddsProcessor:
    """
    NHL Odds Data Processor
    
    Processes NHL odds data from CSV files and prepares it for database storage.
    Handles data cleaning, validation, and team name translation to full team names.
    """
    
    def __init__(self):
        self.blacklist = ['', 'N/A', 'n/a', 'N/a', 'NA', 'na', 'Na', '-', '--', '---', '----', '-----']
        
        # Team name translation dictionary for NHL (converts to full team names)
        self.team_translations = {
            'Ducks': 'Anaheim Ducks',
            'Coyotes': 'Arizona Coyotes', 
            'Bruins': 'Boston Bruins',
            'Sabres': 'Buffalo Sabres',
            'Flames': 'Calgary Flames',
            'Hurricanes': 'Carolina Hurricanes',
            'Blackhawks': 'Chicago Blackhawks',
            'Avalanche': 'Colorado Avalanche',
            'Blue Jackets': 'Columbus Blue Jackets',
            'Stars': 'Dallas Stars',
            'Red Wings': 'Detroit Red Wings',
            'Oilers': 'Edmonton Oilers',
            'Panthers': 'Florida Panthers',
            'Kings': 'Los Angeles Kings',
            'Wild': 'Minnesota Wild',
            'Canadiens': 'Montréal Canadiens',
            'Predators': 'Nashville Predators',
            'New Jersey': 'New Jersey Devils',
            'Devils': 'New Jersey Devils',
            'NY Islanders': 'New York Islanders',
            'Islanders': 'New York Islanders',
            'Rangers': 'New York Rangers',
            'Senators': 'Ottawa Senators',
            'Flyers': 'Philadelphia Flyers',
            'Penguins': 'Pittsburgh Penguins',
            'San Jose': 'San Jose Sharks',
            'Sharks': 'San Jose Sharks',
            'Seattle Kraken': 'Seattle Kraken',
            'St.Louis': 'St. Louis Blues',
            'Blues': 'St. Louis Blues',
            'Lightning': 'Tampa Bay Lightning',
            'Maple Leafs': 'Toronto Maple Leafs',
            'Utah Hockey Club': 'Utah Hockey Club',
            'Canucks': 'Vancouver Canucks',
            'Golden Knights': 'Vegas Golden Knights',
            'Capitals': 'Washington Capitals',
            'Jets': 'Winnipeg Jets',
            'Atlanta': 'Atlanta Thrashers',  # Thrashers (relocated to Winnipeg)
            'Phoenix': 'Arizona Coyotes'   # Old name for Coyotes
        }


    def _make_datestr(self, date_str, season=None, start=0, yr_end=2):
        """Convert date string to proper date format"""
        try:
            # Handle YYYYMMDD format from CSV
            if isinstance(date_str, (int, float)) and len(str(int(date_str))) == 8:
                date_str = str(int(date_str))
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                return datetime(year, month, day).date()
            
            # Handle string format with spaces (original format)
            if isinstance(date_str, str) and len(date_str.split()) > 1:
                month_day = date_str.split()[0]
                year_part = date_str.split()[1]
                if len(year_part) == 2:
                    year = 2000 + int(year_part)
                else:
                    year = int(year_part)
                
                # Parse month and day
                if '/' in month_day:
                    month, day = month_day.split('/')
                    return datetime.strptime(f"{year}-{month.zfill(2)}-{day.zfill(2)}", "%Y-%m-%d").date()
            return None
        except:
            return None

    def _translate(self, team_name):
        """Translate team name to abbreviation"""
        return self.team_translations.get(team_name, team_name)


    def _clean_numeric_value(self, value, is_int=False):
        """Clean and convert numeric values, handling blacklisted values"""
        if pd.isna(value) or value in self.blacklist:
            return 0
        
        try:
            if is_int:
                return int(float(value))
            else:
                return float(value)
        except (ValueError, TypeError):
            return 0

    def _validate_team_name(self, team_name):
        """Validate and clean team name"""
        if pd.isna(team_name) or team_name in self.blacklist:
            return "UNKNOWN"
        return str(team_name).strip()

    def _calculate_days_rest(self, df):
        """Calculate days of rest for each team based on previous games"""
        print("  Calculating days of rest for teams...")
        
        # Sort by date to ensure proper calculation
        df_sorted = df.sort_values('Date').copy()
        
        # Initialize days rest columns
        df_sorted['Days_Rest_Home'] = 0
        df_sorted['Days_Rest_Away'] = 0
        
        # Track last game date for each team
        last_game_dates = {}
        
        for idx, row in df_sorted.iterrows():
            home_team = row['Home']
            away_team = row['Away']
            current_date = row['Date']
            
            # Calculate days rest for home team
            if home_team in last_game_dates:
                days_rest_home = (current_date - last_game_dates[home_team]).days
                df_sorted.at[idx, 'Days_Rest_Home'] = max(0, days_rest_home)
            
            # Calculate days rest for away team
            if away_team in last_game_dates:
                days_rest_away = (current_date - last_game_dates[away_team]).days
                df_sorted.at[idx, 'Days_Rest_Away'] = max(0, days_rest_away)
            
            # Update last game dates for both teams
            last_game_dates[home_team] = current_date
            last_game_dates[away_team] = current_date
        
        return df_sorted

    def _reformat_data(self, df, season, covid=False):
        """Reformat CSV data to match the required schema with validation"""
        print(f"  Reformatting {len(df)} records for season {season}...")
        
        new_df = pd.DataFrame()
        
        # Date field
        new_df["Date"] = df["date"].apply(lambda x: self._make_datestr(x, season))
        
        # Team fields
        new_df["Home"] = df["home_team"].apply(lambda x: self._translate(self._validate_team_name(x)))
        new_df["Away"] = df["away_team"].apply(lambda x: self._translate(self._validate_team_name(x)))
        
        # Over/Under (OU) - using close over/under as the main value
        new_df["OU"] = df["close_over_under"].apply(lambda x: self._clean_numeric_value(x))
        
        # Spread - using close spread as the main value
        new_df["Spread"] = df["home_close_spread"].apply(lambda x: self._clean_numeric_value(x))
        
        # Moneyline fields
        new_df["ML_Home"] = df["home_close_ml"].apply(lambda x: self._clean_numeric_value(x, True))
        new_df["ML_Away"] = df["away_close_ml"].apply(lambda x: self._clean_numeric_value(x, True))
        
        # Points - total points scored in the game
        home_final = df["home_final"].apply(lambda x: max(0, self._clean_numeric_value(x, True)))
        away_final = df["away_final"].apply(lambda x: max(0, self._clean_numeric_value(x, True)))
        new_df["Points"] = home_final + away_final
        
        # Win_Margin - difference in final scores
        new_df["Win_Margin"] = home_final - away_final
        
        # Keep season for reference
        new_df["season"] = df["season"]

        # Remove rows with invalid dates
        valid_dates = new_df["Date"].notna()
        new_df = new_df[valid_dates].copy()
        
        # Remove rows with unknown teams
        valid_teams = (new_df["Home"] != "UNKNOWN") & (new_df["Away"] != "UNKNOWN")
        new_df = new_df[valid_teams].copy()
        
        # Calculate days of rest for each team
        new_df = self._calculate_days_rest(new_df)
        
        print(f"  Cleaned data: {len(new_df)} valid records remaining")
        return new_df


    def process_season(self, df, season):
        """Process odds data for a specific season from provided DataFrame"""
        print(f"Processing NHL odds data for season: {season}")
        
        try:
            # Filter data for the specific season
            season_data = df[df['season'] == season].copy()
            
            if season_data.empty:
                print(f"No data found for season {season}")
                return pd.DataFrame()
            
            # Reformat the data
            formatted_data = self._reformat_data(season_data, season)
            
            print(f"Processed {len(formatted_data)} records for season {season}")
            return formatted_data
            
        except Exception as e:
            print(f"Error processing data for season {season}: {str(e)}")
            return pd.DataFrame()
    

# Main execution
def main():
    """Main function to read NHL odds data from CSV and store in database"""
    print("Starting NHL Odds Data Processing...")
    print("Reading from OddsData.csv and storing in database...")
    
    # Read the CSV file once
    try:
        print("Reading OddsData.csv...")
        df = pd.read_csv("../../Data/OddsData.csv")
        print(f"Successfully loaded {len(df)} total records from CSV")
        
        # Get available seasons
        available_seasons = sorted(df['season'].unique())
        print(f"Found data for seasons: {available_seasons}")
        
        # Connect to database
        db_path = "../../Data/OddsData.sqlite"
        print(f"Connecting to database: {db_path}")
        
        # Ensure the Data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        con = sqlite3.connect(db_path)
        
        # Process each available season
        total_processed = 0
        for season in available_seasons:
            print(f"\n{'='*50}")
            print(f"Processing season: {season}")
            print(f"{'='*50}")
            
            # Create processor and process data
            processor = NHLOddsProcessor()
            season_df = processor.process_season(df, season)
            
            if not season_df.empty:
                # Create season string for table name (e.g., 2009 -> "2009_10")
                season_str = f"{season}_{str(season + 1)[2:]}"
                table_name = f"odds_{season_str}"
                
                # Save to database
                season_df.to_sql(table_name, con, if_exists="replace", index=False)
                print(f"✓ Saved {len(season_df)} odds records to table '{table_name}'")
                total_processed += len(season_df)
                
                # Display sample of data
                print(f"Sample data for {season}:")
                print(season_df[['Date', 'Home', 'Away', 'OU', 'Spread', 'ML_Home', 'ML_Away', 'Points', 'Win_Margin', 'Days_Rest_Home', 'Days_Rest_Away']].head(3))
            else:
                print(f"⚠ No odds data found for season {season}")
        
        print(f"\n{'='*50}")
        print(f"Processing completed!")
        print(f"Total records processed: {total_processed}")
        print(f"Seasons processed: {len(available_seasons)}")
        print(f"{'='*50}")
        
    except FileNotFoundError:
        print("❌ Error: OddsData.csv file not found in Data/ directory")
        return False
    except pd.errors.EmptyDataError:
        print("❌ Error: OddsData.csv file is empty")
        return False
    except sqlite3.OperationalError as e:
        print(f"❌ Database error: {str(e)}")
        print("   This might be due to incorrect file path or permissions")
        return False
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        return False
    finally:
        if 'con' in locals():
            con.close()
            print("Database connection closed.")
    
    return True

def create_database_summary():
    """Create a summary of the database tables and their contents"""
    try:
        db_path = "../../Data/OddsData.sqlite"
        con = sqlite3.connect(db_path)
        cursor = con.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'odds_%'")
        tables = cursor.fetchall()
        
        print(f"\n{'='*60}")
        print("DATABASE SUMMARY")
        print(f"{'='*60}")
        print(f"Total tables created: {len(tables)}")
        
        total_records = 0
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            total_records += count
            print(f"  {table_name}: {count:,} records")
        
        print(f"\nTotal records in database: {total_records:,}")
        print(f"{'='*60}")
        
        con.close()
        return True
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database error in summary: {str(e)}")
        print("   Database file might not exist or be accessible")
        return False
    except Exception as e:
        print(f"Error creating database summary: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ NHL Odds Data import and database storage completed successfully!")
        create_database_summary()
    else:
        print("\n❌ NHL Odds Data import and database storage failed!")
