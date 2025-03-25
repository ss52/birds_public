import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# Import the stopover detection function from the provided code
from stopover import stopover_detection

def process_bird_data(filename, min_duration=48, max_radius=30000, annotate=True, year=False):
    """
    Process bird movement data to detect stopovers
    
    Parameters:
    filename (str): Path to the CSV or Excel file with bird movement data
    min_duration (float): Minimum stopover duration in hours
    max_radius (float): Maximum stopover radius in meters
    annotate (bool): If True, annotate original data with stopover indicators
                     If False, return only stopover points as separate tracks
    
    Returns:
    None, but saves processed data to CSV files in the same directory as input
    """
    print(f"Processing file: {filename}")
    
    # Load data CSV or Excel file
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filename)
        else:
            df = pd.read_csv(filename, sep=';', decimal=',', low_memory=False)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Validate required columns
    required_columns = ['event_id', 'timestamp', 'location_long', 'location_lat', 'Bird_id', 'year']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return
    
    # Convert timestamp string to datetime
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    except Exception as e:
        print(f"Error converting timestamps: {e}")
        return
    
    # Create geometry column for GeoPandas
    geometry = [Point(xy) for xy in zip(df['location_long'], df['location_lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Add track_id column (required by stopover_detection)
    # Use Bird_id as the track identifier
    gdf['track_id'] = gdf['Bird_id']
    
    # Add individual_local_identifier column for stopover_detection output
    gdf['individual_local_identifier'] = gdf['Bird_id']
    
    # Run stopover detection
    print(f"Running stopover detection (duration: {min_duration}h, radius: {max_radius}m)...")
    result_gdf, stopover_info = stopover_detection(
        gdf, 
        duration=min_duration, 
        radius=max_radius, 
        annot=annotate
    )
    
    if result_gdf is None:
        print("No stopovers detected.")
        return
    
    # Create output filenames
    base_name = os.path.splitext(filename)[0]
    
    # Save processed data
    if annotate:
        output_filename = f"{base_name}_with_stopovers.csv"
        result_gdf.drop('geometry', axis=1).to_csv(output_filename, index=False)
        print(f"Saved annotated data to {output_filename}")
    else:
        output_filename = f"{base_name}_stopover_tracks.csv"
        result_gdf.drop('geometry', axis=1).to_csv(output_filename, index=False)
        print(f"Saved stopover tracks to {output_filename}")
    
    # Stopover information already saved by stopover_detection function
    print(f"Stopover details saved to {base_name}_stopover_sites.csv")
    stopover_csv_path = os.path.join(os.getcwd(), f"{base_name}_stopover_sites.csv")
    stopover_info.to_csv(stopover_csv_path, index=False)
    
    # Analyze by year if requested
    if year:
        analyze_by_year(result_gdf, base_name, annotate)
    
    return

def analyze_by_year(gdf, base_name, annotate):
    """Split results by year and save separate files"""
    if 'year' not in gdf.columns:
        print("No year column found, skipping year analysis")
        return
    
    # Group by year
    for year, year_data in gdf.groupby('year'):
        if annotate:
            output_filename = f"{base_name}_{year}_with_stopovers.csv"
        else:
            output_filename = f"{base_name}_{year}_stopover_tracks.csv"
        
        year_data.drop('geometry', axis=1).to_csv(output_filename, index=False)
        print(f"Saved year {year} data to {output_filename}")

def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process bird movement data for stopover detection')
    parser.add_argument('filename', type=str, help='Path to CSV file with bird data')
    parser.add_argument('--duration', type=float, default=48, 
                        help='Minimum stopover duration in hours (default: 48)')
    parser.add_argument('--radius', type=float, default=30000, 
                        help='Maximum stopover radius in meters (default: 30000)')
    parser.add_argument('--stopover-only', action='store_false', dest='annotate', default=True,
                        help='Return only stopover locations instead of annotating all data')
    parser.add_argument('--by-year', action='store_false', dest='year', default=False,
                        help='Return stopover location by years')
    
    
    args = parser.parse_args()
    
    process_bird_data(
        args.filename,
        min_duration=args.duration,
        max_radius=args.radius,
        annotate=args.annotate,
        year=args.year
    )

if __name__ == "__main__":
    main()