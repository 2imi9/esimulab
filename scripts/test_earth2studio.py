"""Test Earth2Studio ERA5 data fetch inside Docker container.

Usage:
    docker compose run genesis-sim python scripts/test_earth2studio.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def test_arco_fetch():
    """Fetch ERA5 data via ARCO and save to data/atmo/."""
    print("=" * 60)
    print("Earth2Studio ERA5 (ARCO) Integration Test")
    print("=" * 60)

    try:
        from earth2studio.data import ARCO

        print("[OK] earth2studio.data.ARCO imported successfully")
    except ImportError as e:
        print(f"[FAIL] Cannot import ARCO: {e}")
        sys.exit(1)

    # Fetch ERA5 for a small region
    bbox = (-119.1, 34.0, -119.0, 34.1)  # small California coastal area
    time = datetime(2023, 6, 15, 12, 0)
    variables = ["u10m", "v10m", "t2m", "tcwv", "tp"]

    print(f"\nFetching ERA5 for bbox={bbox}, time={time}")
    print(f"Variables: {variables}")

    arco = ARCO(cache=True, verbose=True)
    da = arco(time=time, variable=variables)
    print(f"\n[OK] Raw DataArray shape: {da.shape}")
    print(f"     Dimensions: {da.dims}")
    print(f"     Coordinates: {list(da.coords)}")

    # Crop to bbox
    west, south, east, north = bbox
    lats = da.coords["lat"].values
    lons = da.coords["lon"].values

    # ERA5 ARCO uses 0-360 longitude; convert negative bbox lons
    lon_min = west % 360
    lon_max = east % 360

    print(f"  Lon range in data: [{lons.min():.1f}, {lons.max():.1f}]")
    print(f"  Converted bbox lon: [{lon_min:.1f}, {lon_max:.1f}]")

    if lats[0] > lats[-1]:
        da_region = da.sel(lat=slice(north, south), lon=slice(lon_min, lon_max))
    else:
        da_region = da.sel(lat=slice(south, north), lon=slice(lon_min, lon_max))

    print(f"\n[OK] Cropped shape: {da_region.shape}")

    # Extract variables
    for var in variables:
        vals = da_region.sel(variable=var).values.squeeze()
        print(f"     {var}: mean={np.nanmean(vals):.4f}, shape={vals.shape}")

    # Save to data/atmo/
    output_dir = Path("data/atmo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wind
    u = float(da_region.sel(variable="u10m").values.squeeze().mean())
    v = float(da_region.sel(variable="v10m").values.squeeze().mean())
    mag = float(np.sqrt(u**2 + v**2))
    wind_data = {
        "direction": [u / mag if mag > 0 else 1.0, v / mag if mag > 0 else 0.0, 0.0],
        "magnitude": mag,
        "turbulence_strength": mag * 0.2,
        "source": "ERA5_ARCO",
        "time": time.isoformat(),
    }
    (output_dir / "wind.json").write_text(json.dumps(wind_data, indent=2))
    print(f"\n[OK] Wind data saved: {mag:.2f} m/s")

    # Precip
    tp = float(da_region.sel(variable="tp").values.squeeze().mean())
    precip_data = {
        "rate_mm_hr": max(0.0, tp),
        "terminal_velocity": 9.0,
        "droplet_size": 0.05 * (1.0 + tp / 10.0),
        "source": "ERA5_ARCO",
    }
    (output_dir / "precip.json").write_text(json.dumps(precip_data, indent=2))
    print(f"[OK] Precip data saved: {tp:.4f} mm/hr")

    # Temperature
    t2m = float(da_region.sel(variable="t2m").values.squeeze().mean())
    print(f"[OK] Temperature: {t2m:.1f} K ({t2m - 273.15:.1f} °C)")

    print("\n" + "=" * 60)
    print("All Earth2Studio tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_arco_fetch()
