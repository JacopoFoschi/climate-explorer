from copernicus_data_pipeline import CopernicusDataPipeline

config_path = 'datasets/copernicus/reanalysis_era5_pressure_levels/config.yaml'
schema_path = 'datasets/copernicus/reanalysis_era5_pressure_levels/schemas.yaml'
pipeline = CopernicusDataPipeline(config_path, schema_path)
result = pipeline.download_dataset(
    dataset='reanalysis-era5-pressure-levels',
    request={
        'variable': ['geopotential'],
        'year': ['2024'],
        'month': ['03'],
        'day': ['01'],
        'time': ['12:00'],
        'pressure_level': ['1000'],
        'format': 'grib'
    }
)
