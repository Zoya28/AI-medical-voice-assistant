"""
influx_data_manager.py

Manages health and fitness data retrieval from InfluxDB.
Optimized version using InfluxDB native aggregation functions (sum, mean, max, min).
"""

from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dotenv import load_dotenv
import os
import pytz
from dateutil import parser

load_dotenv()
logger = logging.getLogger("InfluxDataManager")

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

class InfluxDataManager:
    """
    Manages health and fitness data retrieval from InfluxDB.
    
    Features:
    - Proper timezone handling (IST to UTC conversion)
    - Native InfluxDB aggregations (sum, mean, max, min)
    - Real-time queries (current, last N minutes)
    - Date range queries with InfluxDB aggregations
    - Single date queries
    - Seamless integration with EntityExtractor
    - Context manager support for proper resource cleanup
    """
    
    def __init__(self, user_id: str, url: str = INFLUX_URL, 
                 token: str = INFLUX_TOKEN, org: str = INFLUX_ORG, 
                 bucket: str = INFLUX_BUCKET, timezone: str = "Asia/Kolkata"):
        """
        Initialize InfluxDB Data Manager.
        
        Args:
            user_id: User ID to query data for
            url: InfluxDB URL
            token: InfluxDB authentication token
            org: InfluxDB organization
            bucket: InfluxDB bucket name
            timezone: User's timezone (default: Asia/Kolkata for IST)
        """
        self.user_id = user_id
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        
        # Timezone handling
        self.local_tz = pytz.timezone(timezone)
        self.utc_tz = pytz.UTC
        
        # Initialize InfluxDB client
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.query_api = self.client.query_api()
        
        logger.info(f"Initialized InfluxDB Data Manager for user: {user_id} (Timezone: {timezone})")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False
    
    def close(self):
        """Close InfluxDB client connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("Closed InfluxDB client connection")

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse a date string, including relative terms like 'today', 'yesterday'."""
        try:
            # Use dateutil.parser for robust parsing
            dt = parser.parse(date_str)
            return self.local_tz.localize(dt) if dt.tzinfo is None else dt
        except (ValueError, TypeError):
            logger.error(f"Could not parse date string: {date_str}")
            return None

    def _convert_to_utc(self, dt: datetime) -> datetime:
        """Convert local datetime to UTC."""
        if dt.tzinfo is None:
            dt = self.local_tz.localize(dt)
        utc_dt = dt.astimezone(self.utc_tz)
        logger.debug(f"Converted {dt} ({dt.tzinfo}) to {utc_dt} UTC")
        return utc_dt
    
    def _convert_from_utc(self, dt: datetime) -> datetime:
        """Convert UTC datetime to local timezone."""
        if dt.tzinfo is None:
            dt = self.utc_tz.localize(dt)
        local_dt = dt.astimezone(self.local_tz)
        return local_dt
    
    def _execute_query(self, query: str) -> List[Dict]:
        """Execute Flux query and return results."""
        try:
            tables = self.query_api.query(query, org=self.org)
            results = []
            
            for table in tables:
                for record in table.records:
                    result_dict = {
                        '_value': record.get_value(),
                        '_field': record.get_field(),
                        '_measurement': record.get_measurement()
                    }
                    
                    # Safely get time field and convert to local timezone
                    try:
                        utc_time = record.get_time()
                        result_dict['_time'] = self._convert_from_utc(utc_time)
                        result_dict['_time_utc'] = utc_time
                    except (KeyError, AttributeError):
                        result_dict['_time'] = datetime.now(self.local_tz)
                    
                    # Add any additional fields from the record
                    for key in record.values.keys():
                        if key not in ['_time', '_value', '_field', '_measurement']:
                            result_dict[key] = record.values[key]
                    
                    results.append(result_dict)
            
            return results
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            logger.error(f"Query was: {query}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _empty_result(self, metric: str) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'metric': metric,
            'summary': {},
            'data': [],
            'error': f'No data found for {metric}'
        }
    
    def _normalize_time_of_day(self, time_of_day: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """
        Normalize time of day names to time ranges.
        
        Supports:
        - morning: 05:00 - 11:59
        - afternoon: 12:00 - 16:59
        - evening: 17:00 - 20:59
        - night: 21:00 - 04:59
        - custom ranges: {'start': '09:00', 'end': '17:00'}
        """
        if not time_of_day:
            return None
        
        # If already has start/end, return as is
        if 'start' in time_of_day and 'end' in time_of_day:
            return time_of_day
        
        # Map named time periods to hour ranges
        time_period_map = {
            'morning': {'start': '05:00', 'end': '11:59'},
            'early morning': {'start': '05:00', 'end': '08:00'},
            'afternoon': {'start': '12:00', 'end': '16:59'},
            'evening': {'start': '17:00', 'end': '20:59'},
            'early evening': {'start': '17:00', 'end': '19:00'},
            'night': {'start': '21:00', 'end': '04:59'},
            'late night': {'start': '23:00', 'end': '02:59'},
            'dawn': {'start': '05:00', 'end': '07:00'},
            'dusk': {'start': '18:00', 'end': '20:00'},
            'noon': {'start': '11:00', 'end': '13:00'},
            'midnight': {'start': '23:00', 'end': '01:00'},
        }
        
        # Get the period name
        period = time_of_day.get('period', '').lower()
        
        if period in time_period_map:
            result = time_period_map[period].copy()
            logger.info(f"Normalized time period '{period}' to {result}")
            return result
        
        return time_of_day
    
    def _build_time_range(self, date: Optional[str], 
                         date_range: Optional[Dict[str, str]], 
                         time_of_day: Optional[Dict[str, str]]) -> tuple:
        """Build start and end datetime from date parameters with proper timezone handling."""
        if date_range:
            start_date = self._parse_date_string(date_range['start'])
            end_date = self._parse_date_string(date_range['end'])
        else:
            start_date = self._parse_date_string(date)
            end_date = start_date

        if not start_date or not end_date:
            raise ValueError("Invalid date or date_range provided.")
        
        # Normalize time of day (convert 'morning', 'evening' etc. to time ranges)
        time_of_day = self._normalize_time_of_day(time_of_day)
        
        # Apply time of day filters
        if time_of_day:
            start_time_str = time_of_day.get('start', '00:00')
            end_time_str = time_of_day.get('end', '23:59')
            
            # Parse hours and minutes
            start_hour, start_minute = map(int, start_time_str.split(':'))
            end_hour, end_minute = map(int, end_time_str.split(':'))
            
            # Handle night time that spans midnight (e.g., 21:00 - 04:59)
            if end_hour < start_hour:
                # Night period spans two days
                logger.info(f"Time period spans midnight: {start_time_str} to {end_time_str}")
                start_date = start_date.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                # End time is next day
                end_date = end_date.replace(hour=end_hour, minute=end_minute, second=59, microsecond=999999)
                end_date = end_date + timedelta(days=1)
            else:
                start_date = start_date.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                end_date = end_date.replace(hour=end_hour, minute=end_minute, second=59, microsecond=999999)
            
            logger.info(f"Applied time filter: {start_time_str} to {end_time_str}")
        else:
            # Full day range
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)       
        # Convert to UTC for InfluxDB
        start_utc = self._convert_to_utc(start_date)
        end_utc = self._convert_to_utc(end_date)
        logger.info(f"Time range: {start_date} to {end_date} (local) => {start_utc} to {end_utc} (UTC)")
        return start_utc, end_utc

    
    def _format_date_range(self, start: datetime, end: datetime) -> str:
        """Format date range for display (in local timezone)."""
        if start.tzinfo == self.utc_tz:
            start = self._convert_from_utc(start)
        if end.tzinfo == self.utc_tz:
            end = self._convert_from_utc(end)
        
        if start.date() == end.date():
            return start.strftime('%Y-%m-%d')
        return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    
    def _get_query_method(self, metric: str):
        """Get appropriate query method for metric."""
        metric_map = {
            'steps': self.query_steps,
            'step_count': self.query_steps,
            'heart_rate': self.query_heart_rate,
            'hr': self.query_heart_rate,
            'calories': self.query_calories,
            'blood_pressure': self.query_blood_pressure,
            'bp': self.query_blood_pressure,
            'spo2': self.query_spo2,
            'oxygen': self.query_spo2,
            'oxygen_saturation': self.query_spo2
        }
        return metric_map.get(metric)
    
    def query_from_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Query data based on extracted entities.
        Can handle single dates, date ranges, and multiple date comparisons.
        """
        metrics = entities.get('metrics', [])
        dates = entities.get('dates', [])
        
        if not metrics:
            return {'error': 'No metrics specified'}
        
        if not dates:
            dates = [{'type': 'realtime', 'subtype': 'current', 'timestamp': datetime.now(self.local_tz).isoformat()}]
        
        metric = metrics[0]
        
        # Route based on number of dates provided
        if len(dates) > 1:
            logger.info(f"Processing multi-date comparison query for {metric}")
            return self._query_multi_date(metric, dates)
        
        # Handle single date entity
        date_entity = dates[0]
        
        # Route based on entity type
        if date_entity['type'] == 'realtime':
            logger.info(f"Processing realtime query for {metric}")
            return self._query_realtime(metric, date_entity)
        elif date_entity['type'] == 'date_range':
            return self._query_date_range(metric, date_entity)
        else:
            return self._query_single_date(metric, date_entity)
    
    def _query_realtime(self, metric: str, date_entity: Dict) -> Dict[str, Any]:
        """Handle real-time queries."""
        subtype = date_entity.get('subtype')
        
        if subtype == 'current':
            return self._get_current_metric(metric)
        elif subtype == 'duration':
            start_time = self._parse_date_string(date_entity['start_timestamp'])
            end_time = self._parse_date_string(date_entity['end_timestamp'])
            minutes_ago = date_entity['duration']['amount']
            return self._get_metric_time_window(metric, start_time, end_time, minutes_ago)
        
        return {'error': 'Unsupported realtime subtype'}
    
    def _query_date_range(self, metric: str, date_entity: Dict) -> Dict[str, Any]:
        """Handle date range queries."""
        start_date = date_entity['start']
        end_date = date_entity['end']
        time_of_day = date_entity.get('time_range')
        
        query_method = self._get_query_method(metric)
        if query_method:
            return query_method(
                date_range={'start': start_date, 'end': end_date},
                time_of_day=time_of_day
            )
        return {'error': f'No query method for metric: {metric}'}
    
    def _query_single_date(self, metric: str, date_entity: Dict) -> Dict[str, Any]:
        """Handle single date queries."""
        date = date_entity.get('value')
        time_of_day = date_entity.get('time_range')
        
        query_method = self._get_query_method(metric)
        if query_method:
            return query_method(date=date, time_of_day=time_of_day)
        return {'error': f'No query method for metric: {metric}'}
    
    def _query_multi_date(self, metric: str, date_entities: List[Dict]) -> Dict[str, Any]:
        """Handle queries for multiple, non-consecutive dates for comparison."""
        query_method = self._get_query_method(metric)
        if not query_method:
            return {'error': f'No query method for metric: {metric}'}

        comparison_data = []
        queried_dates = []

        for date_entity in date_entities:
            date = date_entity.get('value')
            if not date:
                continue

            time_of_day = date_entity.get('time_range')
            
            # Query data for each specific date
            result = query_method(date=date, time_of_day=time_of_day)
            
            if 'error' not in result and result.get('data'):
                # The 'data' list from query methods contains daily summaries
                comparison_data.extend(result['data'])
                queried_dates.append(date)

        if not comparison_data:
            return self._empty_result(metric)
        
        # Sort data by date
        comparison_data = sorted(comparison_data, key=lambda x: x['date'])

        return {
            'metric': metric,
            'data': comparison_data,
            'summary': {
                'days_count': len(comparison_data),
                'queried_dates': queried_dates
            },
            'date_range': ", ".join(sorted(queried_dates)),
            'query_type': 'comparison'
        }
    
    def _get_current_metric(self, metric: str) -> Dict[str, Any]:
        """Get current/latest reading using InfluxDB last() function."""
        
        if metric in ['steps', 'step_count', 'distance', 'active_minutes']:
            today_start_local = datetime.now(self.local_tz).replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_utc = self._convert_to_utc(today_start_local)
            
            logger.info(f"Querying steps from {today_start_local} (local) = {today_start_utc} (UTC)")
            
            # Query today's cumulative steps using InfluxDB sum()
            today_query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {today_start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')})
              |> filter(fn: (r) => r["_measurement"] == "bpst_data")
              |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
              |> filter(fn: (r) => r["_field"] == "ts" or r["_field"] == "dc")
              |> sum()
            '''
            logger.info(f"Query: {today_query}")
            result = self._execute_query(today_query)
            
            if not result:
                logger.warning(f"No data found for today for user {self.user_id}")
                return self._empty_result(metric)
            
            today_data = {}
            for record in result:
                today_data[record['_field']] = record['_value']
            
            current_steps = int(today_data.get('ts', 0))
            current_distance = round(today_data.get('dc', 0) / 1000, 2)
            active_minutes = int(current_steps * 3 / 60) if current_steps > 0 else 0
            
            logger.info(f"Successfully retrieved steps: {current_steps}")
            
            return {
                'metric': 'steps',
                'summary': {
                    'current_steps': current_steps,
                    'current_distance_km': current_distance,
                    'current_active_minutes': active_minutes,
                    'last_updated': datetime.now(self.local_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(datetime.now(self.local_tz).date())
            }
        
        elif metric in ['heart_rate', 'hr']:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -1h)
              |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
              |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
              |> filter(fn: (r) => r["_field"] == "HR")
              |> last()
            '''
            
            result = self._execute_query(query)
            if not result:
                return self._empty_result('heart_rate')
            
            return {
                'metric': 'heart_rate',
                'summary': {
                    'current_heart_rate': int(result[0]['_value']),
                    'last_updated': result[0]['_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'readings_count': 1,
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(datetime.now(self.local_tz).date())
            }
        
        elif metric in ['blood_pressure', 'bp']:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -1h)
              |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
              |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
              |> filter(fn: (r) => r["_field"] == "SBP" or r["_field"] == "DBP")
              |> last()
            '''
            
            result = self._execute_query(query)
            if not result:
                return self._empty_result('blood_pressure')
            
            data = {}
            for record in result:
                data[record['_field']] = record['_value']
            
            return {
                'metric': 'blood_pressure',
                'summary': {
                    'current_systolic': int(data.get('SBP', 0)),
                    'current_diastolic': int(data.get('DBP', 0)),
                    'last_updated': datetime.now(self.local_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    'readings_count': 1,
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(datetime.now(self.local_tz).date())
            }
        
        elif metric in ['spo2', 'oxygen', 'oxygen_saturation']:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -1h)
              |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
              |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
              |> filter(fn: (r) => r["_field"] == "SPO2")
              |> last()
            '''
            
            result = self._execute_query(query)
            if not result:
                return self._empty_result('spo2')
            
            return {
                'metric': 'spo2',
                'summary': {
                    'current_spo2': round(result[0]['_value'], 1),
                    'last_updated': result[0]['_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'readings_count': 1,
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(datetime.now(self.local_tz).date())
            }
        
        return self._empty_result(metric)
    
    def _get_metric_time_window(self, metric: str, start_time: datetime, 
                                 end_time: datetime, minutes: int) -> Dict[str, Any]:
        """Get metric data for time window using InfluxDB aggregations."""
        
        start_utc = self._convert_to_utc(start_time)
        end_utc = self._convert_to_utc(end_time)
        
        start_str = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        if metric in ['steps', 'step_count']:
            # Use InfluxDB sum() aggregation
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start_str}, stop: {end_str})
              |> filter(fn: (r) => r["_measurement"] == "bpst_data")
              |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
              |> filter(fn: (r) => r["_field"] == "ts")
              |> sum()
            '''
            logger.info(query)
            result = self._execute_query(query)
            steps = int(result[0]['_value']) if result else 0
            
            return {
                'metric': 'steps',
                'summary': {
                    'steps_in_period': steps,
                    'duration_minutes': minutes,
                    'days_count': 1
                },
                'query_type': 'time_window',
                'date_range': f"Last {minutes} minutes"
            }
        
        elif metric in ['heart_rate', 'hr']:
            # Use InfluxDB mean() aggregation
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start_str}, stop: {end_str})
              |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
              |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
              |> filter(fn: (r) => r["_field"] == "HR")
              |> mean()
            '''
            
            mean_result = self._execute_query(query)
            avg_hr = round(mean_result[0]['_value'], 1) if mean_result else 0.0
            
            return {
                'metric': 'heart_rate',
                'summary': {
                    'avg_heart_rate': avg_hr,
                    'duration_minutes': minutes,
                    'days_count': 1
                },
                'query_type': 'time_window',
                'date_range': f"Last {minutes} minutes"
            }
        
        return self._empty_result(metric)
    
    def query_steps(self, date: str = None, date_range: Dict[str, str] = None, 
                    time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Query steps data from InfluxDB using native aggregations.
        All calculations (sum, avg, max, min) done in InfluxDB.
        """
        if date is None and date_range is None:
            return self._get_current_metric('steps')
        
        start_time, end_time = self._build_time_range(date, date_range, time_of_day)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Base query filter
        base_filter = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "bpst_data")
          |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
          |> filter(fn: (r) => r["_field"] == "ts")
        '''
        
        # Query 1: Daily data for breakdown
        daily_query = base_filter + '''
          |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
        '''
        logger.info(f"Daily query: {daily_query}")
        daily_result = self._execute_query(daily_query)
        
        if not daily_result:
            return self._empty_result('steps')
        
        # Build daily data
        daily_data = []
        for record in daily_result:
            local_time = record['_time']
            daily_data.append({
                'date': local_time.date().isoformat(),
                'step_count': int(record['_value'])
            })
        
        # Sort by date
        daily_data = sorted(daily_data, key=lambda x: x['date'])
        
        # Query 2: Total sum using InfluxDB
        total_query = base_filter + '|> sum()'
        total_result = self._execute_query(total_query)
        total_steps = int(total_result[0]['_value']) if total_result else 0
        
        # Query 3: Max using InfluxDB
        max_query = base_filter + '|> aggregateWindow(every: 1d, fn: sum, createEmpty: false) |> max()'
        max_result = self._execute_query(max_query)
        max_steps = int(max_result[0]['_value']) if max_result else 0
        
        # Query 4: Min using InfluxDB
        min_query = base_filter + '|> aggregateWindow(every: 1d, fn: sum, createEmpty: false) |> min()'
        min_result = self._execute_query(min_query)
        min_steps = int(min_result[0]['_value']) if min_result else 0
        
        # Query 5: Average using InfluxDB
        avg_query = base_filter + '|> aggregateWindow(every: 1d, fn: sum, createEmpty: false) |> mean()'
        avg_result = self._execute_query(avg_query)
        avg_steps = int(avg_result[0]['_value']) if avg_result else 0
        
        num_days = len(daily_data)
        
        return {
            'metric': 'steps',
            'data': daily_data,
            'summary': {
                'total_steps': total_steps,
                'avg_steps': avg_steps,
                'max_steps': max_steps,
                'min_steps': min_steps,
                'days_count': num_days
            },
            'date_range': self._format_date_range(start_time, end_time),
            'query_type': 'historical'
        }
    
    def query_heart_rate(self, date: str = None, date_range: Dict[str, str] = None,
                         time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query heart rate data using InfluxDB mean() aggregation."""
        if date is None and date_range is None:
            return self._get_current_metric('heart_rate')
        
        start_time, end_time = self._build_time_range(date, date_range, time_of_day)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Use InfluxDB mean() for average heart rate
        mean_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
          |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
          |> filter(fn: (r) => r["_field"] == "HR")
          |> mean()
        '''
        
        mean_result = self._execute_query(mean_query)
        
        if not mean_result:
            return self._empty_result('heart_rate')
        
        # Use InfluxDB max()
        max_query = mean_query.replace('|> mean()', '|> max()')
        max_result = self._execute_query(max_query)
        max_hr = int(max_result[0]['_value']) if max_result else 0
        
        # Use InfluxDB min()
        min_query = mean_query.replace('|> mean()', '|> min()')
        min_result = self._execute_query(min_query)
        min_hr = int(min_result[0]['_value']) if min_result else 0
        
        days = (end_time.date() - start_time.date()).days + 1
        
        return {
            'metric': 'heart_rate',
            'summary': {
                'avg_heart_rate': round(mean_result[0]['_value'], 1),
                'max_heart_rate': max_hr,
                'min_heart_rate': min_hr,
                'days_count': days
            },
            'date_range': self._format_date_range(start_time, end_time),
            'query_type': 'historical'
        }
    
    def query_calories(self, date: str = None, date_range: Dict[str, str] = None,
                       time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query calories data using InfluxDB aggregations."""
        if date is None and date_range is None:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -1h)
              |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
              |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
              |> filter(fn: (r) => r["_field"] == "cal")
              |> last()
            '''
            
            result = self._execute_query(query)
            if not result:
                return self._empty_result('calories')
            
            return {
                'metric': 'calories',
                'summary': {
                    'current_calories': int(result[0]['_value']),
                    'days_count': 1
                },
                'query_type': 'current'
            }
        
        start_time, end_time = self._build_time_range(date, date_range, time_of_day)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        base_filter = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
          |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
          |> filter(fn: (r) => r["_field"] == "cal")
        '''
        
        # Daily breakdown
        daily_query = base_filter + '|> aggregateWindow(every: 1d, fn: sum, createEmpty: false)'
        daily_result = self._execute_query(daily_query)
        
        if not daily_result:
            return self._empty_result('calories')
        
        daily_data = []
        for record in daily_result:
            local_time = record['_time']
            daily_data.append({
                'date': local_time.date().isoformat(),
                'calories': int(record['_value'])
            })
        
        daily_data = sorted(daily_data, key=lambda x: x['date'])
        
        # Total using InfluxDB sum()
        total_query = base_filter + '|> sum()'
        total_result = self._execute_query(total_query)
        total_calories = int(total_result[0]['_value']) if total_result else 0
        
        # Average using InfluxDB mean()
        avg_query = base_filter + '|> aggregateWindow(every: 1d, fn: sum, createEmpty: false) |> mean()'
        avg_result = self._execute_query(avg_query)
        avg_calories = int(avg_result[0]['_value']) if avg_result else 0
        
        days = len(daily_data)
        
        return {
            'metric': 'calories',
            'data': daily_data,
            'summary': {
                'total_calories': total_calories,
                'avg_calories': avg_calories,
                'days_count': days
            },
            'date_range': self._format_date_range(start_time, end_time),
            'query_type': 'historical'
        }
    
    def query_blood_pressure(self, date: str = None, date_range: Dict[str, str] = None,
                            time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query blood pressure data using InfluxDB aggregations."""
        if date is None and date_range is None:
            return self._get_current_metric('blood_pressure')
        
        start_time, end_time = self._build_time_range(date, date_range, time_of_day)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Query for systolic (SBP) using InfluxDB mean()
        sbp_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
          |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
          |> filter(fn: (r) => r["_field"] == "SBP")
          |> mean()
        '''
        
        # Query for diastolic (DBP) using InfluxDB mean()
        dbp_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
          |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
          |> filter(fn: (r) => r["_field"] == "DBP")
          |> mean()
        '''
        
        sbp_result = self._execute_query(sbp_query)
        dbp_result = self._execute_query(dbp_query)
        
        if not sbp_result or not dbp_result:
            return self._empty_result('blood_pressure')
        
        avg_systolic = int(sbp_result[0]['_value']) if sbp_result else 0
        avg_diastolic = int(dbp_result[0]['_value']) if dbp_result else 0
        
        days = (end_time.date() - start_time.date()).days + 1
        
        return {
            'metric': 'blood_pressure',
            'summary': {
                'avg_systolic': avg_systolic,
                'avg_diastolic': avg_diastolic,
                'days_count': days
            },
            'date_range': self._format_date_range(start_time, end_time),
            'query_type': 'historical'
        }
    
    def query_spo2(self, date: str = None, date_range: Dict[str, str] = None,
                   time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query SpO2 data using InfluxDB mean() aggregation."""
        if date is None and date_range is None:
            return self._get_current_metric('spo2')
        
        start_time, end_time = self._build_time_range(date, date_range, time_of_day)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Use InfluxDB mean() for average SpO2
        mean_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "dev_bpm_data")
          |> filter(fn: (r) => r["UserId"] == "{self.user_id}")
          |> filter(fn: (r) => r["_field"] == "SPO2")
          |> mean()
        '''
        
        mean_result = self._execute_query(mean_query)
        
        if not mean_result:
            return self._empty_result('spo2')
        
        # Use InfluxDB min()
        min_query = mean_query.replace('|> mean()', '|> min()')
        min_result = self._execute_query(min_query)
        min_spo2 = round(min_result[0]['_value'], 1) if min_result else 0.0
        
        # Use InfluxDB max()
        max_query = mean_query.replace('|> mean()', '|> max()')
        max_result = self._execute_query(max_query)
        max_spo2 = round(max_result[0]['_value'], 1) if max_result else 0.0
        
        days = (end_time.date() - start_time.date()).days + 1
        
        return {
            'metric': 'spo2',
            'summary': {
                'avg_spo2': round(mean_result[0]['_value'], 1),
                'min_spo2': min_spo2,
                'max_spo2': max_spo2,
                'days_count': days
            },
            'date_range': self._format_date_range(start_time, end_time),
            'query_type': 'historical'
        }
    
    def format_for_llm(self, data: Dict[str, Any], query: str) -> str:
        """Format query results for LLM context."""
        if 'error' in data:
            return f"Error: {data['error']}"
        
        metric = data.get('metric', 'unknown')
        summary = data.get('summary', {})
        
        output = [f"Query: {query}", f"Metric: {metric}"]
        
        for key, value in summary.items():
            formatted_key = key.replace('_', ' ').title()
            output.append(f"{formatted_key}: {value}")
        
        # Add daily data if available
        if data.get('data'):
            output.append("\nDaily Breakdown:")
            for day_data in data['data']:
                if 'step_count' in day_data:
                    output.append(f"  {day_data['date']}: {day_data['step_count']:,} steps")
                elif 'calories' in day_data:
                    output.append(f"  {day_data['date']}: {day_data['calories']:,} calories")
        
        return "\n".join(output)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data."""
        return {
            'user_id': self.user_id,
            'bucket': self.bucket,
            'timezone': str(self.local_tz),
            'available_metrics': ['steps', 'heart_rate', 'calories', 'blood_pressure', 'spo2']
        }


# Test usage
if __name__ == "__main__":
    import logging
    
    # Enable detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    USER_ID = "OF36GM"
    
    # Proper usage with context manager
    with InfluxDataManager(user_id=USER_ID) as data_manager:
        print("="*80)
        print("InfluxDB Data Manager Test (Optimized with Native Aggregations)")
        print("="*80)
        
        # Get data summary first
        summary = data_manager.get_data_summary()
        print(f"\nUser ID: {summary['user_id']}")
        print(f"Bucket: {summary['bucket']}")
        print(f"Timezone: {summary['timezone']}")
        print(f"Available Metrics: {summary['available_metrics']}")
        
        # Test 1: Current steps
        print("\n" + "="*80)
        print("Test 1: Current Steps (Today)")
        print("="*80)
        result = data_manager._get_current_metric('steps')
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Steps: {result['summary'].get('current_steps', 0):,}")
            print(f"✓ Distance: {result['summary'].get('current_distance_km', 0)} km")
            print(f"✓ Date Range: {result['date_range']}")
        
        # Test 2: Steps for yesterday (with aggregations)
        print("\n" + "="*80)
        print("Test 2: Steps Yesterday (Using InfluxDB Aggregations)")
        print("="*80)
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        result = data_manager.query_steps(date=yesterday)
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Total Steps: {result['summary'].get('total_steps', 0):,}")
            print(f"✓ Average Steps: {result['summary'].get('avg_steps', 0):,}")
            print(f"✓ Max Steps: {result['summary'].get('max_steps', 0):,}")
            print(f"✓ Min Steps: {result['summary'].get('min_steps', 0):,}")
            print(f"✓ Date Range: {result['date_range']}")
        
        # Test 3: Steps for last 7 days (ALL calculations in InfluxDB)
        print("\n" + "="*80)
        print("Test 3: Steps Last 7 Days (All Aggregations in InfluxDB)")
        print("="*80)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
        result = data_manager.query_steps(date_range={'start': start_date, 'end': end_date})
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Total Steps: {result['summary'].get('total_steps', 0):,}")
            print(f"✓ Average Steps: {result['summary'].get('avg_steps', 0):,}")
            print(f"✓ Max Steps: {result['summary'].get('max_steps', 0):,}")
            print(f"✓ Min Steps: {result['summary'].get('min_steps', 0):,}")
            print(f"✓ Days Count: {result['summary'].get('days_count', 0)}")
            print(f"✓ Date Range: {result['date_range']}")
            print(f"\nDaily Breakdown:")
            if result.get('data'):
                for day_data in result['data']:
                    print(f"  - {day_data['date']}: {day_data['step_count']:,} steps")
        
        # Test 4: Heart rate with min/max (using InfluxDB)
        print("\n" + "="*80)
        print("Test 4: Heart Rate Last 3 Days (With Min/Max from InfluxDB)")
        print("="*80)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        result = data_manager.query_heart_rate(date_range={'start': start_date, 'end': end_date})
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Average Heart Rate: {result['summary'].get('avg_heart_rate', 0)} bpm")
            print(f"✓ Max Heart Rate: {result['summary'].get('max_heart_rate', 0)} bpm")
            print(f"✓ Min Heart Rate: {result['summary'].get('min_heart_rate', 0)} bpm")
            print(f"✓ Days Count: {result['summary'].get('days_count', 0)}")
            print(f"✓ Date Range: {result['date_range']}")
        
        # Test 5: Calories for date range (using InfluxDB aggregations)
        print("\n" + "="*80)
        print("Test 5: Calories Last 7 Days (Aggregated in InfluxDB)")
        print("="*80)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
        result = data_manager.query_calories(date_range={'start': start_date, 'end': end_date})
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Total Calories: {result['summary'].get('total_calories', 0):,}")
            print(f"✓ Average Calories: {result['summary'].get('avg_calories', 0):,}")
            print(f"✓ Days Count: {result['summary'].get('days_count', 0)}")
            print(f"✓ Date Range: {result['date_range']}")
            if result.get('data'):
                print(f"\nDaily Breakdown:")
                for day_data in result['data']:
                    print(f"  - {day_data['date']}: {day_data['calories']:,} cal")
        
        # Test 6: SpO2 with min/max (using InfluxDB)
        print("\n" + "="*80)
        print("Test 6: SpO2 Last 3 Days (With Min/Max from InfluxDB)")
        print("="*80)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        result = data_manager.query_spo2(date_range={'start': start_date, 'end': end_date})
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Average SpO2: {result['summary'].get('avg_spo2', 0)}%")
            print(f"✓ Max SpO2: {result['summary'].get('max_spo2', 0)}%")
            print(f"✓ Min SpO2: {result['summary'].get('min_spo2', 0)}%")
            print(f"✓ Days Count: {result['summary'].get('days_count', 0)}")
            print(f"✓ Date Range: {result['date_range']}")
        
        # Test 7: Blood Pressure (using InfluxDB aggregations)
        print("\n" + "="*80)
        print("Test 7: Blood Pressure Last 3 Days (Aggregated in InfluxDB)")
        print("="*80)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        result = data_manager.query_blood_pressure(date_range={'start': start_date, 'end': end_date})
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Average Systolic: {result['summary'].get('avg_systolic', 0)} mmHg")
            print(f"✓ Average Diastolic: {result['summary'].get('avg_diastolic', 0)} mmHg")
            print(f"✓ Days Count: {result['summary'].get('days_count', 0)}")
            print(f"✓ Date Range: {result['date_range']}")
            
        # Test 8: Multi-date comparison
        print("\n" + "="*80)
        print("Test 8: Multi-date comparison (Today vs Yesterday)")
        print("="*80)
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        entities = {
            'metrics': ['steps'],
            'dates': [
                {'type': 'date', 'value': today},
                {'type': 'date', 'value': yesterday}
            ]
        }
        result = data_manager.query_from_entities(entities)
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Query Type: {result.get('query_type')}")
            print(f"✓ Queried Dates: {result.get('date_range')}")
            print(f"✓ Days Count: {result['summary'].get('days_count', 0)}")
            print("\nComparison Data:")
            if result.get('data'):
                for day_data in result['data']:
                    print(f"  - {day_data['date']}: {day_data.get('step_count', 'N/A'):,} steps")
    
    print("\n" + "="*80)
    print("✓ All tests complete - Connection closed automatically")
    print("✓ All aggregations performed in InfluxDB (sum, mean, max, min)")
    print("="*80)