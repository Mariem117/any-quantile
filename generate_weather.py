import csv
import math
from datetime import datetime, timedelta

# Configuration
start_date = datetime(2006, 1, 1)
end_date = datetime(2019, 1, 1)
current_date = start_date

# Prepare data
rows = []
row_count = 0

print("Generating synthetic weather data...")

while current_date < end_date:
    # Calculate cycles
    day_of_year = current_date.timetuple().tm_yday
    hour_of_day = current_date.hour
    
    # Seasonal cycle (0 to 2*pi over the year)
    seasonal = 2 * math.pi * (day_of_year - 1) / 365.25
    # Diurnal cycle (0 to 2*pi over the day)
    diurnal = 2 * math.pi * hour_of_day / 24
    
    # Temperature (°C) - using simple sine waves
    temperature = (
        15 +                                    # Base
        10 * math.sin(seasonal) +              # Seasonal
        3 * math.sin(diurnal - math.pi/2) +   # Diurnal
        (hash(str(current_date)) % 200 - 100) / 50  # Pseudo-random
    )
    
    # Humidity (%)
    humidity = (
        60 +
        15 * math.sin(seasonal + math.pi) +
        5 * math.sin(diurnal + math.pi) +
        (hash(str(current_date) + "h") % 160 - 80) / 10
    )
    humidity = max(20, min(100, humidity))  # Clip to 20-100%
    
    # Pressure (hPa)
    pressure = (
        1013 +
        3 * math.sin(seasonal) +
        (hash(str(current_date) + "p") % 100 - 50) / 10
    )
    pressure = max(980, min(1040, pressure))  # Clip to realistic range
    
    # Wind Speed (m/s)
    wind_speed = (
        5 +
        2 * math.sin(seasonal + math.pi) +
        abs((hash(str(current_date) + "w") % 60 - 30) / 10)
    )
    wind_speed = max(0, min(25, wind_speed))  # Clip to 0-25 m/s
    
    # Format datetime as string
    ds = current_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add row
    rows.append({
        'ds': ds,
        'temperature': round(temperature, 2),
        'humidity': round(humidity, 2),
        'pressure': round(pressure, 2),
        'wind_speed': round(wind_speed, 2)
    })
    
    row_count += 1
    
    # Progress indicator
    if row_count % 10000 == 0:
        print(f"Generated {row_count} rows...")
    
    # Move to next hour
    current_date += timedelta(hours=1)

# Write to CSV
output_path = './data/exogenous/mhlv_weather.csv'

with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ds', 'temperature', 'humidity', 'pressure', 'wind_speed'])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✓ Generated {len(rows)} weather records")
print(f"✓ Saved to: {output_path}")

# Calculate statistics
temps = [r['temperature'] for r in rows]
humids = [r['humidity'] for r in rows]
pressures = [r['pressure'] for r in rows]
winds = [r['wind_speed'] for r in rows]

print(f"\nData Statistics:")
print(f"Temperature: {min(temps):.1f}°C to {max(temps):.1f}°C (mean: {sum(temps)/len(temps):.1f}°C)")
print(f"Humidity: {min(humids):.1f}% to {max(humids):.1f}% (mean: {sum(humids)/len(humids):.1f}%)")
print(f"Pressure: {min(pressures):.1f} hPa to {max(pressures):.1f} hPa (mean: {sum(pressures)/len(pressures):.1f} hPa)")
print(f"Wind Speed: {min(winds):.1f} m/s to {max(winds):.1f} m/s (mean: {sum(winds)/len(winds):.1f} m/s)")

print(f"\nFirst row: {rows[0]}")
print(f"Last row: {rows[-1]}")
print("\n✓ No external dependencies required - data is clean")