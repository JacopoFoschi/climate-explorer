
# TODO: pagination, rate limits, improve validation scheme, improve caching (now only based on filename and timestamp)
# generalize class to any API: 
# CDS 
# ISIMIP 
# CEDA 
# NGFS (climate): https://climate-impact-explorer.climateanalytics.org/data-download/#user-guide-for-the-climate-impact-explorer-api 
# NGFS (IAM)
# cru 
# hydrogfd 
# grdc

import cdsapi
import logging
import yaml
from pathlib import Path
from typing import Dict, Union, List, Any, Tuple
import time
from datetime import datetime
import json
from jsonschema import validate, ValidationError
import hashlib
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    """Data class for cache entries"""
    filepath: Path
    request_hash: str
    download_time: datetime
    metadata: Dict


class CopernicusDataPipeline:
    def __init__(self, config_path: str = 'config.yaml', schema_path: str = 'schemas.yaml'):
        """
        Initialize the pipeline with configuration, validation schemas, and cache
        """
        self.client = cdsapi.Client()
        self.config = self._load_yaml(config_path)
        self.schemas = self._load_yaml(schema_path)
        self._setup_logging()
        self._setup_directories()
        self._init_cache()
    @staticmethod
    def _load_yaml(file_path: str) -> Dict:
        """Load YAML file"""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_logging(self):
        """Configure logging"""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=self.config.get('log_level', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_directories(self):
        """Create necessary directories"""
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
            
    def _validate_request(self, dataset: str, request: Dict) -> bool:
        """
        Validate the request parameters using JSON Schema validation
        
        Args:
            dataset (str): Dataset identifier
            request (Dict): Request parameters
            
        Returns:
            bool: True if request is valid, False otherwise
        """
        try:
            # Get dataset-specific schema
            schema = self.schemas.get(dataset)
            if not schema:
                self.logger.warning(f"No validation schema found for dataset {dataset}")
                return True
                
            # Validate request against schema
            validate(instance=request, schema=schema)
            
            # Additional custom validations from config
            dataset_config = self.config.get('datasets', {}).get(dataset, {})
            custom_validations = dataset_config.get('custom_validations', {})
            
            for field, rules in custom_validations.items():
                if field in request:
                    value = request[field]
                    # Check range validations
                    if 'range' in rules:
                        min_val, max_val = rules['range']
                        if not all(min_val <= float(v) <= max_val for v in value):
                            self.logger.error(f"Value for {field} outside allowed range [{min_val}, {max_val}]")
                            return False
                            
                    # Check allowed values
                    if 'allowed_values' in rules:
                        if not all(v in rules['allowed_values'] for v in value):
                            self.logger.error(f"Invalid values for {field}. Allowed: {rules['allowed_values']}")
                            return False
                            
            return True
            
        except ValidationError as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {str(e)}")
            return False
    
    def _init_cache(self):
        """Initialize the cache system"""
        self.cache_dir = Path(self.config.get('cache_dir', 'cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'download_cache.pkl'
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load the cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return {}

    def _save_cache(self):
        """Save the cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")

    def _generate_request_hash(self, dataset: str, request: Dict) -> str:
        """Generate a unique hash for the request"""
        request_str = json.dumps({
            'dataset': dataset,
            'request': request
        }, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: CacheEntry) -> bool:
        """
        Check if a cache entry is still valid
        """
        # Check if file exists
        if not cache_entry.filepath.exists():
            return False

        # Get dataset-specific cache settings
        dataset = str(cache_entry.filepath.stem).split('_')[0]  # Extract dataset from filename
        dataset_config = self.config.get('datasets', {}).get(dataset, {})
        cache_duration = dataset_config.get('cache_duration', 
                                         self.config.get('default_cache_duration', '30d'))

        # Parse cache duration
        duration_map = {'h': 'hours', 'd': 'days', 'w': 'weeks', 'm': 'months'}
        value = int(cache_duration[:-1])
        unit = duration_map.get(cache_duration[-1], 'days')
        max_age = timedelta(**{unit: value})

        # Check if cache is expired
        return datetime.now() - cache_entry.download_time < max_age

    def check_existing_download(self, dataset: str, request: Dict) -> Tuple[bool, Union[Path, None]]:
        """
        Check if a valid download already exists for the given request
        
        Returns:
            Tuple[bool, Union[Path, None]]: (is_valid, filepath if exists else None)
        """
        request_hash = self._generate_request_hash(dataset, request)
        
        if request_hash in self.cache:
            cache_entry = self.cache[request_hash]
            if self._is_cache_valid(cache_entry):
                self.logger.info(f"Found valid cached download at {cache_entry.filepath}")
                return True, cache_entry.filepath
            else:
                self.logger.info("Found cached download but it's outdated")
                return False, cache_entry.filepath
        
        return False, None
            
    def construct_filename(self, dataset: str, request: Dict) -> str:
        """
        Construct a meaningful filename using configured patterns
        """
        try:
            # Get filename pattern from config
            pattern = self.config.get('datasets', {}).get(dataset, {}).get(
                'filename_pattern',
                "{dataset}_{variables}_{date}_{format}"
            )
            
            # Prepare variables for pattern
            variables = '_'.join(request.get('variable', ['unknown']))
            date_parts = [
                request.get('year', ['0000'])[0],
                request.get('month', ['00'])[0].zfill(2),
                request.get('day', ['00'])[0].zfill(2)
            ]
            date_str = ''.join(date_parts)
            
            # Additional optional components
            pressure = '_'.join(request.get('pressure_level', ['surface']))
            time = '_'.join(request.get('time', [''])).replace(':', '')
            
            # Format filename
            filename = pattern.format(
                dataset=dataset,
                variables=variables,
                date=date_str,
                pressure=pressure,
                time=time,
                format=request.get('format', 'grib')
            )
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error constructing filename: {str(e)}")
            return f"{dataset}_data.{request.get('format', 'grib')}"
        
    def download_dataset(
        self,
        dataset: str,
        request: Dict,
        target: str = None,
        force_download: bool = False,
        max_retries: int = None,
        retry_delay: int = None
    ) -> Union[Path, None]:
        """
        Download data from Copernicus services with cache checking
        
        Args:
            dataset (str): The dataset identifier
            request (Dict): The request parameters
            target (str): Optional target filename
            force_download (bool): Force download even if valid cache exists
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds
        """
        if not self._validate_request(dataset, request):
            self.logger.error(f"Invalid request parameters for dataset {dataset}")
            return None

        # Check cache unless force_download is True
        if not force_download:
            is_valid, existing_file = self.check_existing_download(dataset, request)
            if is_valid:
                return existing_file

        # Prepare for new download
        if target is None:
            target = self.data_dir / self.construct_filename(dataset, request)
        else:
            target = Path(target)

        # Get dataset-specific settings
        dataset_config = self.config.get('datasets', {}).get(dataset, {})
        max_retries = max_retries or dataset_config.get('max_retries', self.config.get('max_retries', 3))
        retry_delay = retry_delay or dataset_config.get('retry_delay', self.config.get('retry_delay', 60))

        # Perform download
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading {dataset} to {target}")
                self.logger.debug(f"Request parameters: {json.dumps(request, indent=2)}")
                
                self.client.retrieve(dataset, request, str(target))
                
                if target.exists():
                    # Update cache
                    request_hash = self._generate_request_hash(dataset, request)
                    self.cache[request_hash] = CacheEntry(
                        filepath=target,
                        request_hash=request_hash,
                        download_time=datetime.now(),
                        metadata={
                            'dataset': dataset,
                            'request': request
                        }
                    )
                    self._save_cache()
                    
                    self.logger.info(f"Successfully downloaded to {target}")
                    return target
                    
            except Exception as e:
                self.logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Max retries reached. Download failed.")
                    return None
                    
        return None
