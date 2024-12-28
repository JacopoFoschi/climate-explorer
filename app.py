import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import glob
import cfgrib

# Import our pipeline
from copernicus_data_pipeline import CopernicusDataPipeline

class ClimateDataDashboard:
    def __init__(
        self, 
        config_path: str = 'datasets/copernicus/reanalysis_era5_pressure_levels/config.yaml', 
        schema_path: str = 'datasets/copernicus/reanalysis_era5_pressure_levels/schemas.yaml'
    ):
        """Initialize dashboard with pipeline"""
        self.pipeline = CopernicusDataPipeline(config_path, schema_path)
    
    def close_dataset(self, ds):
        """Safely close the dataset and clean up resources"""
        if ds is not None:
            try:
                ds.close()
            except Exception as e:
                print(f"Error closing dataset: {e}")
        
    def cleanup_idx_files(self, grib_path: Path):
        """Clean up .idx files associated with a GRIB file"""
        # Get the directory and filename
        directory = grib_path.parent
        filename = grib_path.stem
        
        # Find and remove all .idx files matching the pattern
        idx_pattern = str(directory / f"{filename}*.idx")
        for idx_file in glob.glob(idx_pattern):
            try:
                os.remove(idx_file)
                print(f"Removed index file: {idx_file}")
            except Exception as e:
                print(f"Error removing index file {idx_file}: {e}")

    def get_available_data_types(self, file_path: Path) -> list:
        """
        Get available data types in the GRIB file
        """
        try:
            # Open the GRIB file without any filters first
            backend_kwargs = {'indexpath': ''}
            ds = xr.open_dataset(file_path, engine='cfgrib', backend_kwargs=backend_kwargs)
            
            if hasattr(ds, 'attrs') and 'GRIB_dataType' in ds.attrs:
                data_types = [ds.attrs['GRIB_dataType']]
            else:
                # Try alternative method using cfgrib directly
                messages = cfgrib.open_file(file_path)
                data_types = set()
                for msg in messages:
                    if 'dataType' in msg:
                        data_types.add(msg['dataType'])
            
            # Clean up
            ds.close()
            self.cleanup_idx_files(file_path)
            
            return list(data_types) if data_types else ['an']  # Default to 'an' if no types found
            
        except Exception as e:
            st.error(f"Error reading data types: {e}")
            # Default to common ERA5 data types if we can't read them
            return ['an', 'em', 'es']

    def read_data(self, file_path: Path, data_type: str = None):
        """
        Read and process climate data file with data type filtering
        """
        try:
            file_path = Path(file_path)
            self.cleanup_idx_files(file_path)

            if file_path.suffix == '.grib':
                # Set up backend kwargs with filter
                backend_kwargs = {
                    'filter_by_keys': {'dataType': data_type or 'an'},  # Default to 'an' if no type specified
                    'indexpath': '',
                }

                ds = xr.open_dataset(
                    file_path,
                    engine='cfgrib',
                    backend_kwargs=backend_kwargs
                )

            elif file_path.suffix == '.nc':
                ds = xr.open_dataset(file_path)
            else:
                st.error(f"Unsupported file format: {file_path}")
                return None

            return ds

        except Exception as e:
            st.error(f"Error reading data file: {e}")
            return None

    def plot_map(self, ds, variable):
        """Create an interactive map plot"""
        try:
            # Try to get the data, handling different possible structures
            if 'time' in ds.dims:
                data = ds[variable].isel(time=0)
            else:
                data = ds[variable]

            if 'level' in ds.dims:
                # If there are multiple pressure levels, take the first one
                data = data.isel(level=0)

            # Create figure using plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=data.values,
                y=data.latitude.values,
                x=data.longitude.values,
                colorscale='Viridis',
                colorbar=dict(title=f"{variable} ({getattr(data, 'units', 'unknown')})"),
                hoverongaps=False
            ))

            # Update layout for a map-like appearance
            fig.update_layout(
                title=f"{variable.title()} Distribution",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=600,
                # Make it look more like a map
                xaxis=dict(
                    range=[data.longitude.min(), data.longitude.max()],
                    showgrid=True,
                    gridcolor='lightgray',
                ),
                yaxis=dict(
                    range=[data.latitude.min(), data.latitude.max()],
                    showgrid=True,
                    gridcolor='lightgray',
                    scaleanchor="x",  # Make the aspect ratio equal
                    scaleratio=1,
                ),
            )

            return fig

        except Exception as e:
            st.error(f"Error creating map plot: {str(e)}")
            st.write("Dataset structure:", ds)
            return None

    def plot_timeseries(self, ds, variable, lat, lon):
        """Create time series plot for a specific location"""
        try:
            if 'time' not in ds.dims:
                st.warning("No time dimension available for time series plot")
                return None

            # Extract data
            data = ds[variable]
            
            # Handle level dimension if present
            if 'level' in ds.dims:
                data = data.isel(level=0)

            # Select nearest point
            data = data.sel(
                latitude=lat,
                longitude=lon,
                method='nearest'
            )

            # Create figure
            fig = px.line(
                x=data.time.values,
                y=data.values,
                title=f"{variable.title()} at {lat:.2f}°N, {lon:.2f}°E",
                labels={'x': 'Time', 'y': f"{variable} ({getattr(data, 'units', 'unknown')})"}
            )

            return fig

        except Exception as e:
            st.error(f"Error creating time series plot: {str(e)}")
            return None

    def run(self):
        """Run the Streamlit dashboard"""
        st.title("Climate Data Explorer")
        
        # Sidebar for data selection
        st.sidebar.header("Data Selection")
        
        # Dataset selection
        dataset = st.sidebar.selectbox(
            "Select Dataset",
            ["reanalysis-era5-pressure-levels", "reanalysis-era5-single-levels"]
        )
        
        # Variable selection
        variables = {
            "reanalysis-era5-pressure-levels": ["geopotential", "temperature", "specific_humidity"],
            "reanalysis-era5-single-levels": ["2m_temperature", "total_precipitation", "surface_pressure"]
        }
        
        variable = st.sidebar.selectbox(
            "Select Variable",
            variables[dataset]
        )
        
        # Time period selection
        current_year = datetime.now().year
        year = st.sidebar.selectbox("Select Year", range(current_year, current_year-5, -1))
        month = st.sidebar.selectbox("Select Month", range(1, 13))
        day = st.sidebar.selectbox("Select Day", range(1, 32))
        
        # Pressure level selection for pressure-level data
        pressure_level = None
        if dataset == "reanalysis-era5-pressure-levels":
            pressure_level = st.sidebar.selectbox(
                "Select Pressure Level (hPa)",
                ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "50"]
            )

        # Create request
        request = {
            'variable': [variable],
            'year': [str(year)],
            'month': [str(month)],
            'day': [str(day)],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'format': 'grib'
        }
        
        if pressure_level:
            request['pressure_level'] = [pressure_level]
        
        # Download button
        if st.sidebar.button("Download/Update Data"):
            with st.spinner("Downloading data..."):
                result = self.pipeline.download_dataset(dataset, request)
                if result:
                    st.success("Data downloaded successfully!")
                else:
                    st.error("Failed to download data")
                    return
        
        # Variable to plot
        variable = st.sidebar.text_input(
            "Select Variable to plot",
            'isobaricInhPa'
        )

        # Check if we have data available
        is_valid, filepath = self.pipeline.check_existing_download(dataset, request)
        
        if is_valid and filepath:
            # Data type selection (if needed)
            data_type = None
            if Path(filepath).suffix == '.grib':
                available_types = self.get_available_data_types(filepath)
                if available_types:
                    data_type = st.sidebar.selectbox(
                        "Select Data Type",
                        available_types,
                        help="Choose the type of data to display"
                    )

            # Read the data
            ds = self.read_data(filepath, data_type)
            
        if ds is not None:
            try:
                # Main content area
                st.header("Data Visualization")
                
                # Display dataset information
                st.subheader("Dataset Information")
                st.write("Available dimensions:", list(ds.dims))
                st.write("Available variables:", list(ds.variables))
                
                # Map plot
                st.subheader("Spatial Distribution")
                map_fig = self.plot_map(ds, variable)
                if map_fig is not None:
                    st.plotly_chart(map_fig, use_container_width=True)
                
                # Time series plot (only if time dimension exists)
                if 'time' in ds.dims:
                    st.subheader("Time Series at Location")
                    col1, col2 = st.columns(2)
                    lat = col1.number_input("Latitude", -90.0, 90.0, 0.0)
                    lon = col2.number_input("Longitude", -180.0, 180.0, 0.0)
                    
                    ts_fig = self.plot_timeseries(ds, variable, lat, lon)
                    if ts_fig is not None:
                        st.plotly_chart(ts_fig, use_container_width=True)
                else:
                    st.info("Time series plot not available - no time dimension in data")
                
                # Data statistics
                st.subheader("Data Statistics")
                try:
                    stats = ds[variable].describe()
                    st.write(stats.to_pandas().round(2))
                except Exception as e:
                    st.error(f"Could not compute statistics: {str(e)}")
                
            finally:
                self.close_dataset(ds)
                self.cleanup_idx_files(Path(filepath))
        else:
            st.info("Please download data using the sidebar controls")

                
if __name__ == "__main__":
    dashboard = ClimateDataDashboard()
    dashboard.run()