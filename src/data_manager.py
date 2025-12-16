import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os

logger = logging.getLogger("DataManager")

# Default CSV path
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "health_dataset_week.csv")


class DataManager:
    """
    Manages health and fitness data retrieval and formatting.
    
    Features:
    - Date range queries with proper averaging
    - Real-time queries (current, last N minutes)
    - Single date queries
    - Optimized for memory and speed
    - Seamless integration with EntityExtractor
    """
    
    def __init__(self, csv_path: str = CSV_PATH):
        """
        Initialize Data Manager.
        
        Args:
            csv_path: Path to CSV file with health data
        """
        self.csv_path = csv_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess CSV data with optimizations."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Load CSV with optimized dtypes
        dtypes = {
            'heart_rate': 'float32',
            'steps': 'int32',
            'calories': 'float32',
            'sleep_hours': 'float32'
        }
        parse_dates = ['timestamp', 'date']
        
        # Load CSV with optimized settings
        self.df = pd.read_csv(
            self.csv_path,
            dtype=dtypes,
            parse_dates=parse_dates,
            cache_dates=True,
            memory_map=True  # Memory efficient reading
        )
        
        # Keep timestamp as column for filtering
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.sort_values('timestamp', inplace=True)
        
        # Optimize date storage
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        
        logger.info(f"Loaded {len(self.df)} records from {self.df['date'].nunique()} days")
    
    def query_from_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Query data based on extracted entities.
        
        Args:
            entities: Dictionary from EntityExtractor.extract_entities()
            
        Returns:
            Dictionary with query results
        """
        # Reload data to ensure fresh values
        self._load_data()
        
        metrics = entities.get('metrics', [])
        dates = entities.get('dates', [])
        
        if not metrics:
            return {'error': 'No metrics specified'}
        
        if len(dates) > 1:
            # Handle multi-date comparison
            logger.info(f"Processing multi-date comparison for {metric}")
            return self._query_multi_date(metric, dates)
        
        date_entity = dates[0] if dates else {'type': 'realtime', 'subtype': 'current', 'timestamp': datetime.now().isoformat()}
        
        # Route based on entity type
        if date_entity['type'] == 'realtime':
            logger.info(f"Processing realtime query for {metric}")
            return self._query_realtime(metric, date_entity)
        elif date_entity['type'] == 'date_range':
            return self._query_date_range(metric, date_entity)
        else:
            return self._query_single_date(metric, date_entity)
    
    def _query_realtime(self, metric: str, date_entity: Dict) -> Dict[str, Any]:
        """Handle real-time queries with optimized performance."""
        subtype = date_entity.get('subtype')
        now = pd.Timestamp.now()
        
        # Use vectorized operations instead of loops
        if subtype == 'current':
            # Get today's data only
            today = pd.Timestamp.now().date()
            todays_data = self.df[self.df['date'] == today]
            
            if len(todays_data) == 0:
                return {'error': 'No data available for today'}
            
            # Get the most recent data point for today
            latest_data = todays_data.iloc[-1:]
            
            if metric == 'steps':
                return {
                    'metric': 'steps',
                    'summary': {
                        'current_steps': int(latest_data['step_count'].iloc[0]),
                        'current_distance_km': round(float(latest_data['distance_km'].iloc[0]), 2),
                        'current_active_minutes': int(latest_data['active_minutes'].iloc[0]),
                        'last_updated': str(latest_data['timestamp'].iloc[0]),
                        'days_count': 1
                    },
                    'query_type': 'current',
                    'date_range': str(today)
                }
            else:
                # Use efficient aggregation for other metrics
                result = {
                    'value': float(latest_data[metric].iloc[0]),
                    'timestamp': latest_data['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
                }
                return result
        elif subtype == 'duration':
            start_time = pd.to_datetime(date_entity['start_timestamp'])
            end_time = pd.to_datetime(date_entity['end_timestamp'])
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
        """Get current/latest reading."""
        if 'timestamp' not in self.df.columns or self.df.empty:
            return self._empty_result(metric)

        latest = self.df.iloc[-1]

        if metric == 'steps':
            return {
                'metric': 'steps',
                'summary': {
                    'current_steps': int(latest.get('step_count', 0)),
                    'current_distance_km': round(float(latest.get('distance_km', 0)), 2),
                    'current_active_minutes': int(latest.get('active_minutes', 0)),
                    'last_updated': str(latest.get('timestamp')),
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(latest.get('date'))
            }
        elif metric == 'heart_rate':
            return {
                'metric': 'heart_rate',
                'summary': {
                    'current_heart_rate': int(latest.get('heart_rate', 0)),
                    'current_resting_heart_rate': int(latest.get('resting_heart_rate', 0)),
                    'last_updated': str(latest.get('timestamp')),
                    'readings_count': 1,
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(latest.get('date'))
            }

        return self._empty_result(metric)
    
    def _get_metric_time_window(self, metric: str, start_time, end_time, minutes: int) -> Dict[str, Any]:
        """Get metric data for time window."""
        window_df = self.df[
            (self.df['timestamp'] >= start_time) & 
            (self.df['timestamp'] <= end_time)
        ]
        
        if window_df.empty:
            return self._empty_result(metric)
        
        if metric == 'steps':
            start_steps = int(window_df.iloc[0]['step_count'])
            end_steps = int(window_df.iloc[-1]['step_count'])
            steps_in_period = max(0, end_steps - start_steps)
            
            return {
                'metric': 'steps',
                'summary': {
                    'steps_in_period': steps_in_period,
                    'duration_minutes': minutes,
                    'readings_count': len(window_df),
                    'days_count': 1
                },
                'query_type': 'time_window',
                'date_range': f"Last {minutes} minutes"
            }
        elif metric == 'heart_rate':
            avg_hr = round(window_df['heart_rate'].mean(), 1)
            max_hr = int(window_df['heart_rate'].max())
            min_hr = int(window_df['heart_rate'].min())
            
            return {
                'metric': 'heart_rate',
                'summary': {
                    'avg_heart_rate': avg_hr,
                    'max_heart_rate': max_hr,
                    'min_heart_rate': min_hr,
                    'duration_minutes': minutes,
                    'readings_count': len(window_df),
                    'days_count': 1
                },
                'query_type': 'time_window',
                'date_range': f"Last {minutes} minutes"
            }
        
        return self._empty_result(metric)
    
    # ========== METRIC QUERY METHODS ==========
    
    def query_steps(self, date: str = None, date_range: Dict[str, str] = None, 
                    time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query steps with FIXED averaging."""
        # Reload data to ensure fresh values
        self._load_data()
        
        # **FIX: Handle "current" case when all params are None**
        if date is None and date_range is None:
            # Return today's current data
            today = pd.Timestamp.now().date()
            todays_data = self.df[self.df['date'] == today]
            
            if todays_data.empty:
                return self._empty_result('steps')
            
            # Get the most recent data point for today
            latest_data = todays_data.iloc[-1]
            
            return {
                'metric': 'steps',
                'summary': {
                    'current_steps': int(latest_data['step_count']),
                    'current_distance_km': round(float(latest_data['distance_km']), 2),
                    'current_active_minutes': int(latest_data['active_minutes']),
                    'last_updated': str(latest_data['timestamp']),
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(today)
            }
        
        # Continue with normal date/range filtering
        filtered_df = self._filter_by_date_and_time(date, date_range, time_of_day)
        if filtered_df.empty:
            return self._empty_result('steps')
        
        # Proper daily aggregation
        daily_data = filtered_df.groupby('date', as_index=False).agg({
            'step_count': 'max',
            'distance_km': 'max',
            'active_minutes': 'max'
        })
        
        if daily_data.empty:
            return self._empty_result('steps')
        
        # FIXED: Explicit calculations
        num_days = len(daily_data)
        total_steps = int(daily_data['step_count'].sum())
        avg_steps = int(total_steps / num_days)
        max_steps = int(daily_data['step_count'].max())
        min_steps = int(daily_data['step_count'].min())
        total_distance = round(daily_data['distance_km'].sum(), 2)
        total_active = int(daily_data['active_minutes'].sum())
        
        return {
            'metric': 'steps',
            'data': daily_data.to_dict('records'),
            'summary': {
                'total_steps': total_steps,
                'avg_steps': avg_steps,
                'max_steps': max_steps,
                'min_steps': min_steps,
                'total_distance_km': total_distance,
                'total_active_minutes': total_active,
                'days_count': num_days
            },
            'date_range': self._get_date_range_str(daily_data),
            'time_of_day': time_of_day,
            'query_type': 'historical'
        }
    
    def query_heart_rate(self, date: str = None, date_range: Dict[str, str] = None,
                         time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query heart rate (multiple readings per day)."""
        
        # **FIX: Handle "current" case when all params are None**
        if date is None and date_range is None:
            # Return current/latest reading
            if 'timestamp' not in self.df.columns or self.df.empty:
                return self._empty_result('heart_rate')

            latest = self.df.iloc[-1]

            return {
                'metric': 'heart_rate',
                'summary': {
                    'current_heart_rate': int(latest['heart_rate']),
                    'current_resting_heart_rate': int(latest['resting_heart_rate']),
                    'last_updated': str(latest['timestamp']),
                    'readings_count': 1,
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(latest['date'])
            }
        
        # Continue with normal filtering
        filtered_df = self._filter_by_date_and_time(date, date_range, time_of_day)
        if filtered_df.empty:
            return self._empty_result('heart_rate')
        
        # Stats across all readings
        avg_hr = round(filtered_df['heart_rate'].mean(), 1)
        max_hr = int(filtered_df['heart_rate'].max())
        min_hr = int(filtered_df['heart_rate'].min())
        avg_resting = round(filtered_df['resting_heart_rate'].mean(), 1)
        
        # Reset index to get timestamp as column
        df_with_timestamp = filtered_df.reset_index()
        return {
            'metric': 'heart_rate',
            'data': df_with_timestamp[['timestamp', 'date', 'heart_rate', 'resting_heart_rate']].to_dict('records'),
            'summary': {
                'avg_heart_rate': avg_hr,
                'max_heart_rate': max_hr,
                'min_heart_rate': min_hr,
                'avg_resting_heart_rate': avg_resting,
                'readings_count': len(filtered_df),
                'days_count': filtered_df['date'].nunique()
            },
            'date_range': self._get_date_range_str(filtered_df),
            'time_of_day': time_of_day,
            'query_type': 'historical'
        }
    
    def query_calories(self, date: str = None, date_range: Dict[str, str] = None,
                       time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query calories."""
        
        # **FIX: Handle "current" case when all params are None**
        if date is None and date_range is None:
            # Return today's current data
            today = pd.Timestamp.now().date()
            todays_data = self.df[self.df['date'] == today]
            
            if todays_data.empty:
                return self._empty_result('calories')
            
            # Get the most recent data point for today
            latest_data = todays_data.iloc[-1]
            
            return {
                'metric': 'calories',
                'summary': {
                    'current_burned': int(latest_data['calories_burned']),
                    'current_consumed': int(latest_data['calories_consumed']),
                    'current_net': int(latest_data['calories_consumed'] - latest_data['calories_burned']),
                    'last_updated': str(latest_data['timestamp']),
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(today)
            }
        
        # Continue with normal filtering
        filtered_df = self._filter_by_date_and_time(date, date_range, time_of_day)
        if filtered_df.empty:
            return self._empty_result('calories')
        
        daily_data = filtered_df.groupby('date', as_index=False).agg({
            'calories_burned': 'max',
            'calories_consumed': 'max'
        })
        
        if daily_data.empty:
            return self._empty_result('calories')
        
        daily_data['net_calories'] = daily_data['calories_consumed'] - daily_data['calories_burned']
        
        num_days = len(daily_data)
        total_burned = int(daily_data['calories_burned'].sum())
        total_consumed = int(daily_data['calories_consumed'].sum())
        avg_burned = int(total_burned / num_days)
        avg_consumed = int(total_consumed / num_days)
        
        return {
            'metric': 'calories',
            'data': daily_data.to_dict('records'),
            'summary': {
                'total_burned': total_burned,
                'total_consumed': total_consumed,
                'avg_burned': avg_burned,
                'avg_consumed': avg_consumed,
                'net_total': int(daily_data['net_calories'].sum()),
                'avg_net': int(daily_data['net_calories'].mean()),
                'days_count': num_days
            },
            'date_range': self._get_date_range_str(daily_data),
            'time_of_day': time_of_day,
            'query_type': 'historical'
        }
    
    def query_sleep(self, date: str = None, date_range: Dict[str, str] = None,
                    time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query sleep."""
        
        # **FIX: Handle "current" case when all params are None**
        if date is None and date_range is None:
            # Return last night's sleep (most recent date)
            if self.df.empty:
                return self._empty_result('sleep')
            
            latest_date = self.df['date'].max()
            latest_data = self.df[self.df['date'] == latest_date].iloc[-1]
            
            return {
                'metric': 'sleep',
                'summary': {
                    'last_total_hours': round(float(latest_data['total_hours']), 2),
                    'last_deep_sleep_hours': round(float(latest_data['deep_sleep_hours']), 2),
                    'last_quality_score': int(latest_data['quality_score']),
                    'last_date': str(latest_date),
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(latest_date)
            }
        
        # Continue with normal filtering
        filtered_df = self._filter_by_date_and_time(date, date_range, None)
        if filtered_df.empty:
            return self._empty_result('sleep')
        
        daily_data = filtered_df.groupby('date', as_index=False).agg({
            'total_hours': 'max',
            'deep_sleep_hours': 'max',
            'quality_score': 'max'
        })
        
        if daily_data.empty:
            return self._empty_result('sleep')
        
        num_days = len(daily_data)
        total_sleep = round(daily_data['total_hours'].sum(), 2)
        avg_total = round(total_sleep / num_days, 2)
        
        return {
            'metric': 'sleep',
            'data': daily_data.to_dict('records'),
            'summary': {
                'avg_total_hours': avg_total,
                'avg_deep_sleep_hours': round(daily_data['deep_sleep_hours'].mean(), 2),
                'avg_quality_score': round(daily_data['quality_score'].mean(), 1),
                'total_sleep_hours': total_sleep,
                'best_quality_score': int(daily_data['quality_score'].max()),
                'worst_quality_score': int(daily_data['quality_score'].min()),
                'days_count': num_days
            },
            'date_range': self._get_date_range_str(daily_data),
            'time_of_day': None,
            'query_type': 'historical'
        }
    
    def query_distance(self, date: str = None, date_range: Dict[str, str] = None,
                       time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query distance."""
        
        # **FIX: Handle "current" case when all params are None**
        if date is None and date_range is None:
            # Return today's current data
            today = pd.Timestamp.now().date()
            todays_data = self.df[self.df['date'] == today]
            
            if todays_data.empty:
                return self._empty_result('distance')
            
            # Get the most recent data point for today
            latest_data = todays_data.iloc[-1]
            
            return {
                'metric': 'distance',
                'summary': {
                    'current_distance_km': round(float(latest_data['distance_km']), 2),
                    'last_updated': str(latest_data['timestamp']),
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(today)
            }
        
        # Continue with normal filtering
        filtered_df = self._filter_by_date_and_time(date, date_range, time_of_day)
        if filtered_df.empty:
            return self._empty_result('distance')
        
        daily_data = filtered_df.groupby('date', as_index=False).agg({
            'distance_km': 'max',
            'step_count': 'max'
        })
        
        if daily_data.empty:
            return self._empty_result('distance')
        
        num_days = len(daily_data)
        total_distance = round(daily_data['distance_km'].sum(), 2)
        avg_distance = round(total_distance / num_days, 2)
        
        return {
            'metric': 'distance',
            'data': daily_data.to_dict('records'),
            'summary': {
                'total_distance_km': total_distance,
                'avg_distance_km': avg_distance,
                'max_distance_km': round(daily_data['distance_km'].max(), 2),
                'min_distance_km': round(daily_data['distance_km'].min(), 2),
                'days_count': num_days
            },
            'date_range': self._get_date_range_str(daily_data),
            'time_of_day': time_of_day,
            'query_type': 'historical'
        }
    
    def query_active_minutes(self, date: str = None, date_range: Dict[str, str] = None,
                             time_of_day: Dict[str, str] = None) -> Dict[str, Any]:
        """Query active minutes."""
        
        # **FIX: Handle "current" case when all params are None**
        if date is None and date_range is None:
            # Return today's current data
            today = pd.Timestamp.now().date()
            todays_data = self.df[self.df['date'] == today]
            
            if todays_data.empty:
                return self._empty_result('active_minutes')
            
            # Get the most recent data point for today
            latest_data = todays_data.iloc[-1]
            
            return {
                'metric': 'active_minutes',
                'summary': {
                    'current_active_minutes': int(latest_data['active_minutes']),
                    'last_updated': str(latest_data['timestamp']),
                    'days_count': 1
                },
                'query_type': 'current',
                'date_range': str(today)
            }
        
        # Continue with normal filtering
        filtered_df = self._filter_by_date_and_time(date, date_range, time_of_day)
        if filtered_df.empty:
            return self._empty_result('active_minutes')
        
        daily_data = filtered_df.groupby('date', as_index=False).agg({
            'active_minutes': 'max'
        })
        
        if daily_data.empty:
            return self._empty_result('active_minutes')
        
        num_days = len(daily_data)
        total_active = int(daily_data['active_minutes'].sum())
        avg_active = int(total_active / num_days)
        
        return {
            'metric': 'active_minutes',
            'data': daily_data.to_dict('records'),
            'summary': {
                'total_active_minutes': total_active,
                'avg_active_minutes': avg_active,
                'max_active_minutes': int(daily_data['active_minutes'].max()),
                'min_active_minutes': int(daily_data['active_minutes'].min()),
                'total_active_hours': round(total_active / 60, 2),
                'days_count': num_days
            },
            'date_range': self._get_date_range_str(daily_data),
            'time_of_day': time_of_day,
            'query_type': 'historical'
        }
    
    # ========== LLM FORMATTING ==========
    
    def format_for_llm(self, data: Dict[str, Any], query: str) -> str:
        """Format data for LLM context."""
        if not data or 'metric' not in data:
            return "No data available."
        
        metric = data['metric']
        summary = data.get('summary', {})
        date_range = data.get('date_range', 'Unknown')
        query_type = data.get('query_type', 'historical')
        days = summary.get('days_count', 0)
        
        parts = [f"📊 {metric.upper().replace('_', ' ')}"]
        
        # Header based on query type
        if query_type == 'current':
            parts.append(f"📅 Current Reading")
            if 'last_updated' in summary:
                parts.append(f"⏰ {summary['last_updated']}")
        elif query_type == 'time_window':
            minutes = summary.get('duration_minutes', 0)
            parts.append(f"📅 Last {minutes} minutes")
        else:
            parts.append(f"📅 {date_range}")
            if days > 1:
                parts.append(f"📈 {days} days")
        
        parts.append("")
        
        # Format by metric and query type
        if metric == 'steps':
            if query_type == 'current':
                parts.extend([
                    f"Current: {summary.get('current_steps', 0):,} steps",
                    f"Distance: {summary.get('current_distance_km', 0)} km",
                    f"Active: {summary.get('current_active_minutes', 0)} min"
                ])
            elif query_type == 'time_window':
                parts.extend([
                    f"Steps: {summary.get('steps_in_period', 0):,}",
                    f"Readings: {summary.get('readings_count', 0)}"
                ])
            else:
                parts.extend([
                    f"Total: {summary.get('total_steps', 0):,} steps",
                    f"Average: {summary.get('avg_steps', 0):,} steps/day",
                    f"Distance: {summary.get('total_distance_km', 0)} km"
                ])
                if days > 1:
                    parts.append(f"Range: {summary.get('min_steps', 0):,} - {summary.get('max_steps', 0):,}")
        
        elif metric == 'heart_rate':
            if query_type == 'current':
                parts.extend([
                    f"Current: {summary.get('current_heart_rate', 0)} bpm",
                    f"Resting: {summary.get('current_resting_heart_rate', 0)} bpm"
                ])
            else:
                parts.extend([
                    f"Average: {summary.get('avg_heart_rate', 0)} bpm",
                    f"Range: {summary.get('min_heart_rate', 0)}-{summary.get('max_heart_rate', 0)} bpm",
                    f"Resting: {summary.get('avg_resting_heart_rate', 0)} bpm"
                ])
        
        elif metric == 'calories':
            if query_type == 'current':
                parts.extend([
                    f"Burned: {summary.get('current_burned', 0):,} cal",
                    f"Consumed: {summary.get('current_consumed', 0):,} cal",
                    f"Net: {summary.get('current_net', 0):,} cal"
                ])
            else:
                parts.extend([
                    f"Burned: {summary.get('total_burned', 0):,} cal",
                    f"Consumed: {summary.get('total_consumed', 0):,} cal"
                ])
                if days > 1:
                    parts.append(f"Avg/day: {summary.get('avg_burned', 0):,} cal burned")
        
        elif metric == 'sleep':
            if query_type == 'current':
                parts.extend([
                    f"Last night: {summary.get('last_total_hours', 0)} hours",
                    f"Deep: {summary.get('last_deep_sleep_hours', 0)} hours",
                    f"Quality: {summary.get('last_quality_score', 0)}/100"
                ])
            elif days == 1:
                parts.extend([
                    f"Total: {summary.get('avg_total_hours', 0)} hours",
                    f"Deep: {summary.get('avg_deep_sleep_hours', 0)} hours",
                    f"Quality: {summary.get('avg_quality_score', 0)}/100"
                ])
            else:
                parts.extend([
                    f"Total: {summary.get('total_sleep_hours', 0)} hours",
                    f"Average: {summary.get('avg_total_hours', 0)} hours/night",
                    f"Quality: {summary.get('avg_quality_score', 0)}/100"
                ])
        
        elif metric == 'distance':
            if query_type == 'current':
                parts.extend([
                    f"Current: {summary.get('current_distance_km', 0)} km"
                ])
            else:
                parts.extend([
                    f"Total: {summary.get('total_distance_km', 0)} km",
                    f"Average: {summary.get('avg_distance_km', 0)} km/day"
                ])
        
        elif metric == 'active_minutes':
            if query_type == 'current':
                parts.extend([
                    f"Current: {summary.get('current_active_minutes', 0)} min"
                ])
            else:
                parts.extend([
                    f"Total: {summary.get('total_active_minutes', 0)} min",
                    f"Average: {summary.get('avg_active_minutes', 0)} min/day"
                ])
        
        return "\n".join(parts)
    
    # ========== HELPER METHODS ==========
    
    def _filter_by_date_and_time(self, date: str = None, date_range: Dict[str, str] = None,
                                  time_of_day: Dict[str, str] = None) -> pd.DataFrame:
        """Filter by date/time."""
        # Always start with a fresh copy of the data
        df = self.df.copy()
        
        # Date filtering
        if date:
            date_obj = pd.to_datetime(date).date()
            df = df[df['date'] == date_obj]
        elif date_range:
            start = pd.to_datetime(date_range['start']).date()
            end = pd.to_datetime(date_range['end']).date()
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            
        # Time filtering
        if time_of_day and 'timestamp' in df.columns and not df.empty:
            # Extract hours and minutes for comparison
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            
            # Convert start and end times to hours and minutes
            start_dt = pd.to_datetime(time_of_day['start'], format='%H:%M')
            end_dt = pd.to_datetime(time_of_day['end'], format='%H:%M')
            
            start_hour = start_dt.hour
            start_minute = start_dt.minute
            end_hour = end_dt.hour
            end_minute = end_dt.minute
            
            if start_hour < end_hour or (start_hour == end_hour and start_minute <= end_minute):
                # Time window doesn't cross midnight
                df = df[
                    ((df['hour'] > start_hour) | 
                     (df['hour'] == start_hour) & (df['minute'] >= start_minute)) &
                    ((df['hour'] < end_hour) |
                     (df['hour'] == end_hour) & (df['minute'] <= end_minute))
                ]
            else:
                # Time window crosses midnight
                df = df[
                    ((df['hour'] > start_hour) | 
                     (df['hour'] == start_hour) & (df['minute'] >= start_minute)) |
                    ((df['hour'] < end_hour) |
                     (df['hour'] == end_hour) & (df['minute'] <= end_minute))
                ]
            
            df = df.drop(columns=['hour', 'minute'])
        
        return df
    
    def _get_query_method(self, metric: str):
        """Map metric to query method."""
        mapping = {
            'steps': self.query_steps,
            'step_count': self.query_steps,
            'calories': self.query_calories,
            'calories_burned': self.query_calories,
            'sleep': self.query_sleep,
            'heart_rate': self.query_heart_rate,
            'hr': self.query_heart_rate,
            'distance': self.query_distance,
            'active_minutes': self.query_active_minutes,
            'activity': self.query_active_minutes
        }
        return mapping.get(metric.lower())
    
    def _get_date_range_str(self, df: pd.DataFrame) -> str:
        """Get readable date range."""
        if df.empty:
            return "No data"
        
        if 'date' in df.columns:
            dates = df['date'].unique()
            if len(dates) == 1:
                return str(dates[0])
            return f"{min(dates)} to {max(dates)}"
        
        return "Unknown"
    
    def _empty_result(self, metric: str) -> Dict[str, Any]:
        """Empty result template."""
        return {
            'metric': metric,
            'data': [],
            'summary': {'days_count': 0},
            'date_range': 'No data'
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get dataset summary."""
        return {
            'total_records': len(self.df),
            'date_range': {
                'start': str(self.df['date'].min()),
                'end': str(self.df['date'].max())
            },
            'total_days': self.df['date'].nunique(),
            'available_metrics': [c for c in self.df.columns if c not in ['timestamp', 'date']]
        }


# Test the data manager
if __name__ == "__main__":
    from entity_extractor import EntityExtractor
    
    # Initialize
    extractor = EntityExtractor()
    data_manager = DataManager()
    
    # Test queries
    test_queries = [
        "What's my current step count?",
        "What was my average step count from 1st October to 7th October?",
        "Show me my heart rate yesterday",
        "How many calories did I burn today?",
        "What was my sleep quality last night?"
    ]
    
    print("="*80)
    print("Testing Data Manager with Entity Extractor")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Extract entities
        entities = extractor.extract_entities(query)
        print(f"\nMetrics: {entities['metrics']}")
        print(f"Dates: {entities['dates']}")
        
        # Query data
        result = data_manager.query_from_entities(entities)
        
        if 'error' not in result:
            summary = result.get('summary', {})
            query_type = result.get('query_type', 'unknown')
            print(f"Query Type: {query_type}")
            print(f"Days: {summary.get('days_count')}")
            print(f"Date Range: {result.get('date_range')}")
            
            if result['metric'] == 'steps':
                if query_type == 'current':
                    print(f"Current: {summary.get('current_steps', 0):,}")
                else:
                    print(f"Total: {summary.get('total_steps', 0):,}")
                    print(f"Average: {summary.get('avg_steps', 0):,}")
            
            # Format for LLM
            print(f"\nFormatted:\n{data_manager.format_for_llm(result, query)}")
        else:
            print(f"Error: {result['error']}")
    
    print("\n" + "="*80)
    print("✓ Testing Complete!")
    print("="*80)