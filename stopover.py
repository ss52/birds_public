import pandas as pd
import numpy as np
# import geopandas as gpd
# from shapely.geometry import Point
# import datetime
import math
# import os
# from datetime import timedelta

def ll2xyz(ll):
    """Convert lat/long coordinates (in radians) to 3d Cartesian (on unit sphere)"""
    return np.array([
        math.cos(ll[0]) * math.cos(ll[1]),
        math.cos(ll[0]) * math.sin(ll[1]),
        math.sin(ll[0])
    ])

def xyz2ll(xyz):
    """Convert 3d Cartesian coordinates to lat/long (in radians)"""
    return np.array([
        math.atan2(xyz[2], math.sqrt(sum(xyz[0:2]**2))),
        math.atan2(xyz[1], xyz[0])
    ])

def sphere_circle(lat, lon, r, r_sphere=6371008.8, n=360):
    """Generate a circle centered at (lat,lon) of radius r on the sphere or radius rSphere"""
    # Normalize radius to unit circle
    r = math.sin(r / r_sphere)
    
    # Generate n points equidistant around the North Pole at radius r
    a = (2 * math.pi / n) * np.arange(n + 1)
    P = np.column_stack([
        r * np.cos(a),
        r * np.sin(a),
        np.repeat(math.sqrt(1 - r**2), n + 1)
    ])
    
    # Compute rotation matrix so North Pole ends up at (lat,lon)
    l = -math.pi * (0.5 - (lat / 180))  # to radian and subtract from 90 degree
    Ry = np.array([
        [math.cos(l), 0, math.sin(l)],
        [0, 1, 0],
        [-math.sin(l), 0, math.cos(l)]
    ])
    
    l = lon / 180 * math.pi
    Rz = np.array([
        [math.cos(l), math.sin(l), 0],
        [-math.sin(l), math.cos(l), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry
    
    # Rotate points to lie around new center
    P = np.array([R @ p for p in P])
    
    # Unproject to lat/long
    P = np.array([xyz2ll(p) * (180 / math.pi) for p in P])
    
    # Find the break in longitude if any and reorder
    m = np.argmax(P[:, 1])
    if m != P.shape[0] - 1:
        P = np.vstack([P[m+1:], P[:m+1]])
    
    return P

def dist_ct(p, q):
    """Cartesian distance between points"""
    return math.sqrt(sum((p - q)**2))

def dist_ll(p, q):
    """Great-circle distance between p,q in lat/long coordinates (in radians)"""
    # Haversine formula
    d = q - p
    a = math.sin(d[0]/2)**2 + math.cos(p[0]) * math.cos(q[0]) * math.sin(d[1]/2)**2
    return 2 * math.asin(min(1, math.sqrt(a)))

def midpoint_ct(p, q):
    """Midpoint of p and q in Cartesian coordinates"""
    return (p + q) * 0.5

def midpoint_ll(p, q):
    """Midpoint of shortest great-circle arc between p,q (lat/lon in radians)"""
    return xyz2ll(midpoint_ct(ll2xyz(p), ll2xyz(q)))

def circumcircle_ct(p, q, r):
    """Compute circumcircle of triangle embedded in higher dimensional space"""
    def l(x):
        return math.sqrt(sum(x**2))  # vector length
    
    def l2(x):
        return sum(x**2)  # squared vector length
    
    # Translate p to origin
    q = q - p
    r = r - p
    
    # Compute translated circumcenter
    qxr2 = l2(q) * l2(r) - np.dot(q, r)**2  # ||q cross r||^2
    d = (l2(q) * r - l2(r) * q)
    m = (np.dot(d, r) * q - np.dot(d, q) * r) / (2 * qxr2)
    
    # Compute radius
    radius = l(q) * l(r) * l(q - r) / (2 * math.sqrt(qxr2))
    
    # Translate back circumcenter and return
    return np.append(m + p, radius)

def circumcircle_ll(p, q, r):
    """Midpoint of smallest circumcircle of p,q,r (lat/long in radians)"""
    # Convert to Cartesian, compute circumcircle, convert back
    m = xyz2ll(circumcircle_ct(ll2xyz(p), ll2xyz(q), ll2xyz(r))[0:3])
    # Recompute radius along great circle
    return np.append(m, dist_ll(m, p))

def mini_disc(P, r_max=float('inf'), r_sphere=float('inf')):
    """
    Compute smallest enclosing disc, stop if radius becomes > r_max
    P is a point set (without temporal coordinates)
    """
    # Function to compute SED with boundary points
    def mini_disc_with_points(i, Q):
        # Compute SED of P[0:i] under condition that all points in Q are on boundary
        
        # Set up initial disc
        init_points = np.vstack((Q, P[0:2]))[:2]  # Get 2 points from either Q or P
        disc = np.append(
            midpoint(init_points[0], init_points[1]),
            dist(init_points[0], init_points[1]) / 2
        )
        
        if disc[-1] > r_max:
            return None
        
        # Iteratively add points
        for j in range(2 - Q.shape[0], i - 2 + Q.shape[0]):
            # Test if P[j] is inside disc
            if dist(P[j], disc[:-1]) > disc[-1]:
                # Recursive call requiring P[j] on boundary
                if Q.shape[0] >= 2:
                    disc = circumcircle(Q[0], Q[1], P[j])
                    if disc[-1] > r_max:
                        return None
                else:
                    disc = mini_disc_with_points(j, np.vstack((Q, P[j])))
                    if disc is None:
                        return None
        
        return disc
    
    # Convert P to numpy array with float64 dtype at the start
    try:
        P = np.array([np.array(p) for p in P], dtype=np.float64)
    except ValueError:
        # If conversion fails, try stacking the arrays
        P = np.vstack(P).astype(np.float64)

    if P.shape[0] <= 1:
        if P.shape[0] == 0:
            return None
        disc = np.append(P[0], 0)
    else:
        if np.isfinite(r_sphere):
            # Convert to unit sphere for simplicity of calculations
            r_max = r_max / r_sphere
            # Convert ll coordinates to radians
            P = P * (np.pi / 180)
            
            # Switch to distance functions for spherical coordinates
            global dist, midpoint, circumcircle
            dist, midpoint, circumcircle = dist_ll, midpoint_ll, circumcircle_ll
        else:
            dist, midpoint, circumcircle = dist_ct, midpoint_ct, circumcircle_ct
        
        # Remove duplicate points
        P = np.unique(P, axis=0)
        
        # If only one unique location
        if P.shape[0] == 1:
            disc = np.append(P[0], 0)
        else:
            # Randomly permute points
            # Start with subtrajectory endpoints as a heuristic
            perm = np.concatenate(([0, P.shape[0]-1], 
                                  np.random.permutation(P.shape[0]-2) + 1))
            P = P[perm]
            
            # Compute disc
            Q = np.empty((0, P.shape[1]))
            disc = mini_disc_with_points(P.shape[0], Q)
    
    # Clean result and return
    if disc is not None:
        if np.isinf(r_sphere):
            names = ["x", "y", "r"]
        else:
            disc[0:2] = disc[0:2] * (180 / np.pi)  # Convert lat/lon to degrees
            disc[2] = disc[2] * r_sphere  # Convert radius from unit sphere
            names = ["lat", "long", "r"]
        
        return {name: value for name, value in zip(names, disc)}
    
    return None

def stopovers(trajectory, t_min, r_max):
    """
    Compute stopovers in trajectory.
    t_min: min stopover duration (seconds)
    r_max: max stopover radius (meters)
    """
    # Check if coordinates are lat/long or projected
    if trajectory.crs.to_string().find("EPSG:4326") >= 0:
        # Lat/long coordinates
        dist = dist_ll
        r_sphere = 6371008.8  # Mean Earth radius
        rm = r_max / r_sphere  # Normalize to unit sphere
    else:
        # Projected coordinates
        dist = dist_ct
        r_sphere = float('inf')  # Cartesian coordinates
        rm = r_max  # No normalization necessary
    
    t_min = pd.Timedelta(seconds=t_min)
    
    # Extract timestamps and coordinates
    ts = trajectory.timestamp.values
    coords = trajectory.geometry.apply(lambda p: np.array([p.y, p.x])).values
    
    # Initialize results dataframe
    stopovers = pd.DataFrame(columns=[
        "iStart", "iEnd", "duration", 
        "cLat" if np.isfinite(r_sphere) else "cX", 
        "cLong" if np.isfinite(r_sphere) else "cY", 
        "radius"
    ])
    
    # Scan over trajectory
    start = end = 0  # These are already integers
    while end < len(trajectory) - 1:
        # Find first point more than t_min time away from start
        while pd.Timedelta(ts[end] - ts[start]) < t_min and end < len(trajectory) - 1:
            end += 1
        
        if pd.Timedelta(ts[end] - ts[start]) >= t_min and dist(coords[start], coords[end]) <= rm:
            # Potential stopover, compute SED
            disc = mini_disc(coords[start:end+1], r_max, r_sphere)
            
            if disc is not None:
                # Stopover detected
                # Maximize duration with exponential and binary search
                while disc is not None and end < len(coords) - 1:
                    # Store current stopover with explicit integer conversion
                    so = [
                        int(start), int(end),  # Convert to integers explicitly 
                        pd.Timedelta(ts[end] - ts[start]).total_seconds(),
                        disc["lat" if np.isfinite(r_sphere) else "x"],
                        disc["long" if np.isfinite(r_sphere) else "y"],
                        disc["r"]
                    ]
                    # Double number of points (exponential search)
                    end = min(end + (end - start), len(coords) - 1)
                    disc = mini_disc(coords[start:end+1], r_max, r_sphere)
                
                if disc is not None:  # Stopover lasts until end of trajectory
                    so = [
                        int(start), int(end),  # Convert to integers explicitly 
                        pd.Timedelta(ts[end] - ts[start]).total_seconds(),
                        disc["lat" if np.isfinite(r_sphere) else "x"],
                        disc["long" if np.isfinite(r_sphere) else "y"],
                        disc["r"]
                    ]
                else:
                    # Binary search for maximum duration
                    while end > so[1]:  # so[1] is now guaranteed to be integer
                        m = math.ceil((so[1] + end) / 2)  # m is integer from ceil
                        disc = mini_disc(coords[start:m+1], r_max, r_sphere)
                        if disc is None:
                            end = m - 1
                        else:
                            # Found new lower bound, store longer stopover
                            so = [
                                int(start), int(m),  # Convert to integers explicitly 
                                pd.Timedelta(ts[m] - ts[start]).total_seconds(),
                                disc["lat" if np.isfinite(r_sphere) else "x"],
                                disc["long" if np.isfinite(r_sphere) else "y"],
                                disc["r"]
                            ]
                
                # Add to stopovers dataframe
                stopovers.loc[len(stopovers)] = so
                
                # Find disjoint stopovers only
                start = end = so[1]
        
        start += 1
        if end < start:
            end = start
    
    return stopovers

def stopover_detection(data, duration=None, radius=None, annot=False):
    """
    Main function to detect stopovers in movement data
    
    Parameters:
    data (GeoDataFrame): Movement data with timestamp and geometry columns
                         Must have 'track_id' and 'timestamp' columns
    duration (float): Minimum stopover duration in hours
    radius (float): Maximum stopover radius in meters
    annot (bool): If True, annotate original data with stopover indicators
                  If False, return only stopover points as separate tracks
    
    Returns:
    GeoDataFrame or None: Either annotated data, stopover points, or None if no stopovers found
    DataFrame: Table with stopover information saved as CSV
    """
    # Set timezone to UTC
    pd.Timestamp.utc = True
    
    # Thin data to 5-minute resolution for better performance
    n_all = len(data)
    # Round timestamps to 5 minutes and remove duplicates
    data['rounded_time'] = data.timestamp.dt.round('5min')
    data['track_id_time'] = data.track_id + data.rounded_time.astype(str)
    data = data.drop_duplicates('track_id_time')
    print(f"For better performance, the data have been thinned to max 5 minute resolution. "
          f"From the total {n_all} positions, the algorithm retained {len(data)} positions for calculation.")
    
    # Parameter validation
    if duration is None and radius is None:
        print("You didn't provide any stopover site radius or minimum stopover duration. "
              "Please go back and configure them. Returning input data set.")
        return data, None
    
    if duration is None and radius is not None:
        print(f"You have selected a stopover site radius of {radius}m, but no minimum stopover duration. "
              f"We here use 48 hours by default. If that is not what you need, please reconfigure the parameters.")
        duration = 48
    
    if duration is not None and radius is None:
        print(f"You have selected a minimum stopover duration of {duration}h, but no radius. "
              f"We here use 30,000 m = 30 km by default. If that is not what you need, please reconfigure the parameters.")
        radius = 30000
    
    if duration is not None and radius is not None:
        print(f"You have selected a minimum stopover duration of {duration} hours and a radius of {radius} metres.")
        
        # Process each track separately
        track_groups = data.groupby('track_id')
        
        # Initialize stopover table
        stopover_tab = pd.DataFrame(columns=[
            "animal_ID", 
            "individual_local_identifier", 
            "timestamp_arrival", 
            "timestamp_departure",
            "location_long", 
            "location_lat", 
            "duration", 
            "radius",
            "taxon_canonical_name", 
            "sensor"
        ])
        
        # Process each track
        for track_id, track_data in track_groups:
            # Sort by timestamp
            track_data = track_data.sort_values('timestamp')
            
            # Calculate stopovers for this track
            track_stopovers = stopovers(track_data, duration * 3600, radius)  # Convert hours to seconds
            
            if len(track_stopovers) > 0:
                # Extract information for each stopover
                for _, stopover in track_stopovers.iterrows():
                    # Convert indices to integers before using with iloc
                    start_idx = int(stopover.iStart)
                    end_idx = int(stopover.iEnd)
                    
                    # Get arrival and departure times
                    arrival_time = track_data.iloc[start_idx].timestamp
                    departure_time = track_data.iloc[end_idx].timestamp
                    
                    # Get animal ID
                    if 'individual_local_identifier' in track_data.columns:
                        animal_id = track_data.iloc[start_idx].individual_local_identifier
                    elif 'local_identifier' in track_data.columns:
                        animal_id = track_data.iloc[start_idx].local_identifier
                    else:
                        animal_id = track_id
                    
                    # Calculate duration in hours
                    duration_hours = stopover.duration / 3600
                    
                    # Get taxonomic information if available
                    if 'taxon_canonical_name' in track_data.columns:
                        taxon = track_data.iloc[start_idx].taxon_canonical_name
                    elif hasattr(track_data, 'attrs') and 'taxon_canonical_name' in track_data.attrs:
                        taxon = track_data.attrs['taxon_canonical_name']
                    else:
                        taxon = np.nan
                    
                    # Get sensor information if available
                    if 'sensor_type_ids' in track_data.columns:
                        sensor = track_data.iloc[start_idx].sensor_type_ids
                    elif hasattr(track_data, 'attrs') and 'sensor_type_ids' in track_data.attrs:
                        sensor = track_data.attrs['sensor_type_ids']
                    else:
                        sensor = np.nan
                    
                    # Get coordinates
                    if 'cLong' in stopover:
                        lon = stopover.cLong
                        lat = stopover.cLat
                    elif 'cX' in stopover:
                        lon = stopover.cX
                        lat = stopover.cY
                    else:
                        lon = np.nan
                        lat = np.nan
                    
                    # Add to stopover table
                    stopover_tab = pd.concat([stopover_tab, pd.DataFrame({
                        "animal_ID": [animal_id],
                        "individual_local_identifier": [track_id],
                        "timestamp_arrival": [arrival_time],
                        "timestamp_departure": [departure_time],
                        "location_long": [lon],
                        "location_lat": [lat],
                        "duration": [duration_hours],
                        "radius": [stopover.radius],
                        "taxon_canonical_name": [taxon],
                        "sensor": [sensor]
                    })], ignore_index=True)
        
        # Create output based on annotation preference
        if len(stopover_tab) == 0:
            print("Your output file contains no positions/stopover sites. No csv saved. Return None.")
            return None, None
        
        # Save stopover table as CSV
        stopover_tab.rename(columns={
            "location_long": "mid_longitude",
            "location_lat": "mid_latitude",
            "duration": "duration (h)",
            "radius": "radius (m)"
        }, inplace=True)
        
        # Save CSV
        # stopover_csv_path = os.path.join(os.getcwd(), "stopover_sites.csv")
        # stopover_tab.to_csv(stopover_csv_path, index=False)
        
        if not annot:
            print("Your have selected to return an object with only locations within stopovers. Each stopover location object is returned as a separate track.")
            
            # Extract and stack all stopover sites
            stopover_sites = []
            track_ids = []
            
            for i, stopover in stopover_tab.iterrows():
                track_id = stopover["individual_local_identifier"]
                track_data = track_groups.get_group(track_id)
                
                # Filter for locations within this stopover
                mask = (track_data.timestamp >= stopover["timestamp_arrival"]) & (track_data.timestamp <= stopover["timestamp_departure"])
                stopover_data = track_data[mask].copy()
                
                if len(stopover_data) > 0:
                    # Create a unique ID for this stopover track
                    new_track_id = f"{track_id}_stopover_{i+1}"
                    stopover_data["track_id"] = new_track_id
                    stopover_sites.append(stopover_data)
                    track_ids.append(new_track_id)
            
            if len(stopover_sites) > 0:
                # Combine all stopover tracks
                result = pd.concat(stopover_sites)
                return result, stopover_tab
            else:
                return None, stopover_tab
        else:
            print("Your have selected to return all input data with an additional attribute 'stopover' indicating if the respective location is in a 'stopover' or alternatively 'move'. The tracks are returned as in the input data.")
            
            # Add stopover annotation to original data
            data["stopover"] = "move"
            
            # Update stopover locations
            for _, stopover in stopover_tab.iterrows():
                track_id = stopover["individual_local_identifier"]
                mask = (
                    (data.track_id == track_id) & 
                    (data.timestamp >= stopover["timestamp_arrival"]) & 
                    (data.timestamp <= stopover["timestamp_departure"])
                )
                data.loc[mask, "stopover"] = "stopover"
            
            return data, stopover_tab

