import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger("EntityExtractor")


class EntityExtractor:
    """
    High-performance entity extractor for health-related queries.
    
    Enhanced with comprehensive time-of-day support for all query types.
    """
    
    # Pre-compiled regex patterns for performance
    _PATTERNS = {
        # Comparison patterns - UPDATED for more flexibility
        'comparison_vs': re.compile(
            r'\b(compare)\b.*?\b((?:(?:last|this)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|today|yesterday|this week|last week|this month|last month|\d{1,2}(?:st|nd|rd|th)?\s+\w+)\b'
            r'.*?\b(with|to|vs|versus|and)\b.*?'
            r'\b((?:(?:last|this)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|today|yesterday|this week|last week|this month|last month|\d{1,2}(?:st|nd|rd|th)?\s+\w+)\b',
            re.IGNORECASE
        ),
        'comparison_and': re.compile(
            r'\b((?:(?:last|this)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|today|yesterday)\b\s+and\s+\b((?:(?:last|this)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|today|yesterday)\b',
            re.IGNORECASE),
        # "since" patterns
        'since_date': re.compile(
            r'since\s+(?:last\s+)?(\w+day|\d{1,2}(?:st|nd|rd|th)?\s+\w+)',
            re.IGNORECASE
        ),
        # Explicit date ranges
        'date_range_explicit': re.compile(
            r'(?:from|between)\s+(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(?:to|and|until|-)\s+(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)',
            re.IGNORECASE
        ),
        # Duration ranges
        'duration_range': re.compile(
            r'(?:last|past|previous|in the (?:last|past))\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(week|weeks|month|months|day|days)',
            re.IGNORECASE
        ),
        # Single date patterns
        'date1': re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)\b', re.IGNORECASE),
        'date2': re.compile(r'\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b', re.IGNORECASE),
        'date3': re.compile(r'\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b'),
        # Weekday patterns
        'weekday': re.compile(
            r'\b(last|this|next|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            re.IGNORECASE
        ),
        # Real-time duration patterns
        'past_duration': re.compile(
            r'(?:past|last|previous|in the (?:past|last))\s+(\d+)\s+(sec(?:ond)?s?|min(?:ute)?s?|h(?:ou)?rs?|days?)',
            re.IGNORECASE
        ),
        'time_ago': re.compile(r'(\d+)\s+(sec(?:ond)?s?|min(?:ute)?s?|h(?:ou)?rs?)\s+ago', re.IGNORECASE),
        'since_ago': re.compile(r'since\s+(\d+)\s+(sec(?:ond)?s?|min(?:ute)?s?|h(?:ou)?rs?)\s+ago', re.IGNORECASE),
        'within_last': re.compile(
            r'within\s+(?:the\s+)?last\s+(\d+)\s+(sec(?:ond)?s?|min(?:ute)?s?|h(?:ou)?rs?)',
            re.IGNORECASE
        ),
        # Time of day patterns - ENHANCED
        'time_of_day': re.compile(
            r'\b(morning|afternoon|evening|night|dawn|dusk|noon|midnight|early morning|late night|early evening)\b',
            re.IGNORECASE
        ),
        # Specific time patterns (e.g., "at 9am", "between 2pm and 5pm")
        'specific_time': re.compile(
            r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b',
            re.IGNORECASE
        ),
        'time_range_specific': re.compile(
            r'between\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+(?:and|to)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
            re.IGNORECASE
        ),
        # Numerical values
        'values': re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b')
    }
    
    # Static lookup tables
    _MONTHS = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    _WEEKDAYS = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    _RELATIVE_DATES = {
        'today': 0, 'tonight': 0, 'yesterday': -1, 'day before yesterday': -2,
        'tomorrow': 1, 'last night': -1, 'this morning': 0,
        'this evening': 0, 'this afternoon': 0
    }
    
    _NUMBER_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'a': 1, 'an': 1
    }
    
    _CURRENT_KEYWORDS = frozenset([
        'now', 'currently', 'right now', 'current', 'latest',
        'at present', 'presently', 'at the moment', 'at this moment'
    ])
    
    # ENHANCED: More comprehensive time-of-day mappings
    _TIME_OF_DAY = {
        'morning': {'period': 'morning', 'start': '05:00', 'end': '11:59'},
        'early morning': {'period': 'early morning', 'start': '05:00', 'end': '08:00'},
        'afternoon': {'period': 'afternoon', 'start': '12:00', 'end': '16:59'},
        'evening': {'period': 'evening', 'start': '17:00', 'end': '20:59'},
        'early evening': {'period': 'early evening', 'start': '17:00', 'end': '19:00'},
        'night': {'period': 'night', 'start': '21:00', 'end': '04:59'},
        'late night': {'period': 'late night', 'start': '23:00', 'end': '02:59'},
        'dawn': {'period': 'dawn', 'start': '05:00', 'end': '07:00'},
        'dusk': {'period': 'dusk', 'start': '18:00', 'end': '20:00'},
        'noon': {'period': 'noon', 'start': '11:00', 'end': '13:00'},
        'midnight': {'period': 'midnight', 'start': '23:00', 'end': '01:00'},
    }
    
    def __init__(self, use_llm: bool = False):
        """Initialize the entity extractor."""
        self.use_llm = use_llm
        
        # Cache current datetime for performance
        now = datetime.now()
        self.today = now.date()
        self.now = now
        self.current_weekday = self.today.weekday()
        self.current_year = self.today.year
        
        # Metric keyword sets for O(1) lookup
        self._metric_sets = {
            'steps': frozenset([
                'step', 'steps', 'walking', 'walked', 'walk', 
                'run', 'running', 'ran', 'jog', 'jogging'
            ]),
            'calories': frozenset([
                'calorie', 'calories', 'burned', 'kcal', 'energy', 'burn'
            ]),
            'sleep': frozenset([
                'sleep', 'slept', 'sleeping', 'rest', 'rested', 'nap'
            ]),
            'heart_rate': frozenset([
                'heart rate', 'heart', 'pulse', 'bpm', 'heartrate', 
                'hr', 'beats per minute', 'cardiac'
            ]),
            'distance': frozenset([
                'distance', 'km', 'kilometer', 'mile', 'kilometers', 'miles'
            ]),
            'active_minutes': frozenset([
                'active', 'activity', 'exercise', 'workout', 'active minutes', 'duration'
            ]),
            'blood_pressure': frozenset([
                'blood pressure', 'bp', 'systolic', 'diastolic', 'pressure'
            ]),
            'spo2': frozenset([
                'spo2', 'oxygen', 'oxygen saturation', 'o2', 'blood oxygen'
            ])
        }
        
        logger.info("EntityExtractor initialized with enhanced time-of-day support")
    
    def extract_entities(self, text: str, intent: str = None) -> Dict[str, Any]:
        """Extract all entities from text."""
        text_lower = text.lower()
        
        # Extract time of day FIRST (applies to all queries)
        time_of_day_info = self._extract_time_of_day(text_lower)
        
        # Extract dates and add time_of_day to them
        dates = self.extract_date_entities(text_lower)
        
        # Apply time_of_day to all date entities
        if time_of_day_info:
            for date_entity in dates:
                date_entity['time_range'] = time_of_day_info
        
        return {
            'dates': dates,
            'metrics': self.extract_metrics(text_lower, intent),
            'values': self.extract_values(text),
            'comparisons': [],
            'time_of_day': time_of_day_info  # Also return at top level
        }
    
    def _extract_time_of_day(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract time-of-day information from query.
        
        Returns:
            Dict with 'period', 'start', 'end' or None
        
        Examples:
            "steps yesterday morning" -> {'period': 'morning', 'start': '05:00', 'end': '11:59'}
            "heart rate between 2pm and 5pm" -> {'period': 'custom', 'start': '14:00', 'end': '17:00'}
        """
        # Check for specific time ranges first (e.g., "between 2pm and 5pm")
        time_range_match = self._PATTERNS['time_range_specific'].search(text)
        if time_range_match:
            start_hour = int(time_range_match.group(1))
            start_min = int(time_range_match.group(2)) if time_range_match.group(2) else 0
            start_period = time_range_match.group(3)
            
            end_hour = int(time_range_match.group(4))
            end_min = int(time_range_match.group(5)) if time_range_match.group(5) else 0
            end_period = time_range_match.group(6)
            
            # Convert to 24-hour format
            if start_period and start_period.lower() == 'pm' and start_hour < 12:
                start_hour += 12
            elif start_period and start_period.lower() == 'am' and start_hour == 12:
                start_hour = 0
                
            if end_period and end_period.lower() == 'pm' and end_hour < 12:
                end_hour += 12
            elif end_period and end_period.lower() == 'am' and end_hour == 12:
                end_hour = 0
            
            return {
                'period': 'custom',
                'start': f'{start_hour:02d}:{start_min:02d}',
                'end': f'{end_hour:02d}:{end_min:02d}'
            }
        
        # Check for named time periods (morning, evening, etc.)
        tod_match = self._PATTERNS['time_of_day'].search(text)
        if tod_match:
            period_name = tod_match.group(1).lower()
            if period_name in self._TIME_OF_DAY:
                return self._TIME_OF_DAY[period_name].copy()
        
        return None
    
    def extract_date_entities(self, text_lower: str) -> List[Dict[str, Any]]:
        """Extract date entities with priority order."""
        
        # 0. Check for comparison queries FIRST (e.g., "today vs yesterday")
        comparison_match = self._check_comparison_query(text_lower)
        if comparison_match:
            return comparison_match
        
        # 1. Check for "since" queries
        since_match = self._check_since_query(text_lower)
        if since_match:
            return [since_match]
        
        # 2. Check for explicit date ranges
        range_match = self._check_explicit_date_range(text_lower)
        if range_match:
            return [range_match]
        
        # 3. Check duration ranges
        duration_range = self._check_duration_range(text_lower)
        if duration_range:
            return [duration_range]
        
        # 4. Check real-time
        realtime = self._check_realtime(text_lower)
        if realtime:
            return [realtime]
        
        # 5. Check period ranges
        period_range = self._check_period_range(text_lower)
        if period_range:
            return [period_range]
        
        # 6. Check relative dates
        for phrase, offset in self._RELATIVE_DATES.items():
            if phrase in text_lower:
                target = self.today + timedelta(days=offset)
                return [{
                    'type': 'relative_date',
                    'value': target.strftime('%Y-%m-%d'),
                    'raw_text': phrase,
                    'weekday': target.strftime('%A').lower()
                }]
        
        # 7. Check weekday
        weekday = self._check_weekday(text_lower)
        if weekday:
            return [weekday]
        
        # 8. Check specific date
        date_match = self._check_specific_date(text_lower)
        if date_match:
            return [date_match]
        
        # 9. Default to today
        return [{
            'type': 'relative_date',
            'value': self.today.strftime('%Y-%m-%d'),
            'raw_text': 'today',
            'weekday': self.today.strftime('%A').lower()
        }]
    
    def _check_comparison_query(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract comparison queries like:
        - "steps today vs yesterday"
        - "heart rate this week compared to last week"
        - "calories today versus 5th October"
        
        Returns list of TWO date entities for comparison.
        """
        # Try the more flexible 'compare...with/to/and...' pattern first
        match_vs = self._PATTERNS['comparison_vs'].search(text)
        if match_vs:
            period1_str = match_vs.group(2).lower()
            period2_str = match_vs.group(4).lower()
        else:
            # Fallback to the simpler 'X and Y' pattern
            match_and = self._PATTERNS['comparison_and'].search(text)
            if not match_and:
                return None
            period1_str = match_and.group(1).lower()
            period2_str = match_and.group(2).lower()

        if period1_str == period2_str:
            logger.warning(f"Comparison periods are identical: '{period1_str}'. Skipping comparison.")
            return None

        logger.info(f"Detected comparison: '{period1_str}' vs '{period2_str}'")
        
        # Parse both periods
        entity1 = self._parse_period_string(period1_str)
        entity2 = self._parse_period_string(period2_str)
        
        if entity1 and entity2:
            # Mark as comparison
            entity1['is_comparison'] = True
            entity2['is_comparison'] = True
            return [entity1, entity2]
        
        return None
    
    def _parse_period_string(self, period_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse a period string into a date entity.
        
        Handles: today, yesterday, last week, this month, specific dates, etc.
        """
        # Check relative dates
        for phrase, offset in self._RELATIVE_DATES.items():
            if phrase in period_str:
                target = self.today + timedelta(days=offset)
                return {
                    'type': 'relative_date',
                    'value': target.strftime('%Y-%m-%d'),
                    'raw_text': period_str,
                    'weekday': target.strftime('%A').lower()
                }
        
        # Check period ranges
        if 'last week' in period_str:
            days_since_mon = self.current_weekday
            this_monday = self.today - timedelta(days=days_since_mon)
            last_sunday = this_monday - timedelta(days=1)
            last_monday = last_sunday - timedelta(days=6)
            return {
                'type': 'date_range',
                'start': last_monday.strftime('%Y-%m-%d'),
                'end': last_sunday.strftime('%Y-%m-%d'),
                'raw_text': period_str
            }
        
        if 'this week' in period_str:
            this_monday = self.today - timedelta(days=self.current_weekday)
            return {
                'type': 'date_range',
                'start': this_monday.strftime('%Y-%m-%d'),
                'end': self.today.strftime('%Y-%m-%d'),
                'raw_text': period_str
            }
        
        if 'last month' in period_str:
            first_of_this_month = self.today.replace(day=1)
            last_day_of_last_month = first_of_this_month - timedelta(days=1)
            first_of_last_month = last_day_of_last_month.replace(day=1)
            return {
                'type': 'date_range',
                'start': first_of_last_month.strftime('%Y-%m-%d'),
                'end': last_day_of_last_month.strftime('%Y-%m-%d'),
                'raw_text': period_str
            }
        
        if 'this month' in period_str:
            first_of_month = self.today.replace(day=1)
            return {
                'type': 'date_range',
                'start': first_of_month.strftime('%Y-%m-%d'),
                'end': self.today.strftime('%Y-%m-%d'),
                'raw_text': period_str
            }
        
        # Check specific date
        date_entity = self._check_specific_date(period_str)
        if date_entity:
            return date_entity
        
        # Check weekday
        for weekday_name, weekday_num in self._WEEKDAYS.items():
            if weekday_name in period_str:
                if 'last' in period_str:
                    days_back = (self.current_weekday - weekday_num) % 7 or 7
                    target = self.today - timedelta(days=days_back)
                else:
                    days_back = self.current_weekday - weekday_num
                    target = self.today - timedelta(days=days_back)
                
                return {
                    'type': 'relative_date',
                    'value': target.strftime('%Y-%m-%d'),
                    'raw_text': period_str,
                    'weekday': weekday_name
                }
        
        return None
    
    def _check_since_query(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract 'since' queries."""
        match = self._PATTERNS['since_date'].search(text)
        if not match:
            return None
        
        date_part = match.group(1).lower()
        end = self.today
        start = None
        
        # Check if it's a weekday
        for weekday_name, weekday_num in self._WEEKDAYS.items():
            if weekday_name in date_part:
                days_back = (self.current_weekday - weekday_num) % 7 or 7
                start = self.today - timedelta(days=days_back)
                break
        
        # Check if it's a relative date
        if not start:
            for phrase, offset in self._RELATIVE_DATES.items():
                if phrase in date_part:
                    start = self.today + timedelta(days=offset)
                    break
        
        # Check if it's a specific date
        if not start:
            date_entity = self._check_specific_date(date_part)
            if date_entity and date_entity.get('value'):
                start = datetime.strptime(date_entity['value'], '%Y-%m-%d').date()
        
        if not start:
            return None
        
        return {
            'type': 'date_range',
            'start': start.strftime('%Y-%m-%d'),
            'end': end.strftime('%Y-%m-%d'),
            'raw_text': match.group(0),
            'description': f'since {date_part}'
        }
    
    def _check_duration_range(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract duration ranges."""
        match = self._PATTERNS['duration_range'].search(text)
        if not match:
            return None
        
        amount_str = match.group(1).lower()
        amount = self._NUMBER_WORDS.get(amount_str, int(amount_str) if amount_str.isdigit() else 1)
        
        unit = match.group(2).lower()
        end = self.today
        
        if 'week' in unit:
            days = amount * 7
            start = end - timedelta(days=days - 1)
        elif 'month' in unit:
            days = amount * 30
            start = end - timedelta(days=days - 1)
        else:  # days
            days = amount
            start = end - timedelta(days=days - 1)
        
        return {
            'type': 'date_range',
            'start': start.strftime('%Y-%m-%d'),
            'end': end.strftime('%Y-%m-%d'),
            'raw_text': match.group(0),
            'description': f'{amount} {unit}'
        }
    
    def _check_explicit_date_range(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract explicit date ranges."""
        match = self._PATTERNS['date_range_explicit'].search(text)
        if not match:
            return None
        
        start_day = int(match.group(1))
        start_month_name = match.group(2).lower()
        end_day = int(match.group(3))
        end_month_name = match.group(4).lower()
        
        start_month = self._MONTHS.get(start_month_name)
        end_month = self._MONTHS.get(end_month_name)
        
        if not start_month or not end_month:
            return None
        
        start_date = self._create_date(start_month, start_day)
        end_date = self._create_date(end_month, end_day)
        
        if not start_date or not end_date:
            return None
        
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        
        return {
            'type': 'date_range',
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'raw_text': match.group(0)
        }
    
    def _check_period_range(self, text: str) -> Optional[Dict[str, Any]]:
        """Check for week/month ranges."""
        if 'last week' in text:
            days_since_mon = self.current_weekday
            this_monday = self.today - timedelta(days=days_since_mon)
            last_sunday = this_monday - timedelta(days=1)
            last_monday = last_sunday - timedelta(days=6)
            return {
                'type': 'date_range',
                'start': last_monday.strftime('%Y-%m-%d'),
                'end': last_sunday.strftime('%Y-%m-%d'),
                'raw_text': 'last week'
            }
        
        if 'this week' in text:
            this_monday = self.today - timedelta(days=self.current_weekday)
            return {
                'type': 'date_range',
                'start': this_monday.strftime('%Y-%m-%d'),
                'end': self.today.strftime('%Y-%m-%d'),
                'raw_text': 'this week'
            }
        
        if 'last month' in text:
            first_of_this_month = self.today.replace(day=1)
            last_day_of_last_month = first_of_this_month - timedelta(days=1)
            first_of_last_month = last_day_of_last_month.replace(day=1)
            return {
                'type': 'date_range',
                'start': first_of_last_month.strftime('%Y-%m-%d'),
                'end': last_day_of_last_month.strftime('%Y-%m-%d'),
                'raw_text': 'last month'
            }
        
        if 'this month' in text:
            first_of_month = self.today.replace(day=1)
            return {
                'type': 'date_range',
                'start': first_of_month.strftime('%Y-%m-%d'),
                'end': self.today.strftime('%Y-%m-%d'),
                'raw_text': 'this month'
            }
        
        return None
    
    def _check_realtime(self, text: str) -> Optional[Dict[str, Any]]:
        """Check for real-time queries."""
        for keyword in self._CURRENT_KEYWORDS:
            if keyword in text:
                return {
                    'type': 'realtime',
                    'subtype': 'current',
                    'timestamp': self.now.isoformat(),
                    'raw_text': keyword,
                    'description': 'current moment'
                }
        
        match = self._PATTERNS['past_duration'].search(text)
        if match:
            unit = match.group(2).lower()
            if any(u in unit for u in ['sec', 'min', 'h']):
                return self._build_duration_entity(match, 'past')
        
        match = self._PATTERNS['time_ago'].search(text)
        if match:
            return self._build_ago_entity(match)
        
        match = self._PATTERNS['since_ago'].search(text)
        if match:
            return self._build_duration_entity(match, 'since')
        
        match = self._PATTERNS['within_last'].search(text)
        if match:
            return self._build_duration_entity(match, 'within')
        
        return None
    
    def _build_duration_entity(self, match, prefix: str) -> Dict[str, Any]:
        """Build duration entity."""
        amount = int(match.group(1))
        unit_normalized, delta = self._normalize_time_unit(match.group(2), amount)
        start_time = self.now - delta
        
        return {
            'type': 'realtime',
            'subtype': 'duration',
            'start_timestamp': start_time.isoformat(),
            'end_timestamp': self.now.isoformat(),
            'duration': {'amount': amount, 'unit': unit_normalized},
            'raw_text': match.group(0),
            'description': f'{prefix} {amount} {unit_normalized}'
        }
    
    def _build_ago_entity(self, match) -> Dict[str, Any]:
        """Build point-in-time entity."""
        amount = int(match.group(1))
        unit_normalized, delta = self._normalize_time_unit(match.group(2), amount)
        target_time = self.now - delta
        
        return {
            'type': 'realtime',
            'subtype': 'point_in_time',
            'timestamp': target_time.isoformat(),
            'duration': {'amount': amount, 'unit': unit_normalized},
            'raw_text': match.group(0),
            'description': f'{amount} {unit_normalized} ago'
        }
    
    def _normalize_time_unit(self, unit: str, amount: int) -> Tuple[str, timedelta]:
        """Normalize time unit and return timedelta."""
        unit_lower = unit.lower()
        if unit_lower.startswith('s'):
            return 'seconds', timedelta(seconds=amount)
        elif unit_lower.startswith('m'):
            return 'minutes', timedelta(minutes=amount)
        elif unit_lower.startswith('h'):
            return 'hours', timedelta(hours=amount)
        else:
            return 'days', timedelta(days=amount)
    
    def _check_weekday(self, text: str) -> Optional[Dict[str, Any]]:
        """Check for weekday patterns."""
        match = self._PATTERNS['weekday'].search(text)
        if not match:
            return None
        
        modifier, weekday = match.groups()
        modifier = modifier.lower()
        weekday = weekday.lower()
        target_weekday = self._WEEKDAYS[weekday]
        
        if modifier in ['last', 'on']:
            days_back = (self.current_weekday - target_weekday) % 7 or 7
            target = self.today - timedelta(days=days_back)
        elif modifier == 'this':
            days_back = self.current_weekday - target_weekday
            target = self.today - timedelta(days=days_back)
        else:  # next
            days_forward = (target_weekday - self.current_weekday) % 7 or 7
            target = self.today + timedelta(days=days_forward)
        
        return {
            'type': 'relative_date',
            'value': target.strftime('%Y-%m-%d'),
            'raw_text': match.group(0),
            'weekday': weekday
        }
    
    def _check_specific_date(self, text: str) -> Optional[Dict[str, Any]]:
        """Check for specific dates."""
        # "5th October"
        match = self._PATTERNS['date1'].search(text)
        if match:
            day, month_name = int(match.group(1)), match.group(2).lower()
            month = self._MONTHS.get(month_name)
            if month:
                target = self._create_date(month, day)
                if target:
                    return {
                        'type': 'specific_date',
                        'value': target.strftime('%Y-%m-%d'),
                        'raw_text': match.group(0),
                        'weekday': target.strftime('%A').lower()
                    }
        
        # "October 5"
        match = self._PATTERNS['date2'].search(text)
        if match:
            month_name, day = match.group(1).lower(), int(match.group(2))
            month = self._MONTHS.get(month_name)
            if month:
                target = self._create_date(month, day)
                if target:
                    return {
                        'type': 'specific_date',
                        'value': target.strftime('%Y-%m-%d'),
                        'raw_text': match.group(0),
                        'weekday': target.strftime('%A').lower()
                    }
        
        # "10/5" or "10/5/2025"
        match = self._PATTERNS['date3'].search(text)
        if match:
            p1, p2 = int(match.group(1)), int(match.group(2))
            year = int(match.group(3)) if match.group(3) else self.current_year
            year = year + 2000 if year < 100 else year
            
            for month, day in [(p1, p2), (p2, p1)]:
                target = self._create_date(month, day, year)
                if target:
                    return {
                        'type': 'specific_date',
                        'value': target.strftime('%Y-%m-%d'),
                        'raw_text': match.group(0),
                        'weekday': target.strftime('%A').lower()
                    }
        
        return None
    
    def _create_date(self, month: int, day: int, year: int = None) -> Optional[datetime]:
        """Create date with validation."""
        year = year or self.current_year
        try:
            target = datetime(year, month, day).date()
            return target if target <= self.today else datetime(year - 1, month, day).date()
        except ValueError:
            return None
    
    def extract_metrics(self, text_lower: str, intent: str = None) -> List[str]:
        """Extract metrics using keyword sets."""
        found = []
        for metric, keywords in self._metric_sets.items():
            if any(kw in text_lower for kw in keywords):
                found.append(metric)
        return found
    
    def extract_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical values."""
        return [
            {'value': float(m.group(1).replace(',', '')), 'raw_text': m.group(1)}
            for m in self._PATTERNS['values'].finditer(text)
        ]


# Test
if __name__ == "__main__":
    extractor = EntityExtractor()

    test_queries = [
        # Time-of-day queries
        "How many steps did I take yesterday morning?",
        "What was my heart rate yesterday evening?",
        "Show me my steps from last week in the morning",
        "What's my heart rate between 2pm and 5pm yesterday?",
        "Steps during afternoon for the last 3 days",
        "Calories burned at night last week",

        # Regular queries
        "How many steps since last Friday?",
        "Show my steps for the past 2 weeks",
    ]

    comparison_queries = [
        "Compare my today's and yesterday's steps",
        "Compare my yesterday's step count to last Thursday's step count.",
        "steps today vs yesterday",
        "heart rate this week compared to last week",
        "calories today versus 5th October"
    ]

    print("="*80)
    print("Testing Enhanced Entity Extractor - Time-of-Day Support")
    print("="*80)

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        entities = extractor.extract_entities(query)

        if entities['dates']:
            date_entity = entities['dates'][0]
            print(f"  Date Type: {date_entity['type']}")

            if date_entity['type'] == 'date_range':
                print(f"  Range: {date_entity['start']} to {date_entity['end']}")
            elif date_entity['type'] == 'realtime':
                print(f"  Subtype: {date_entity.get('subtype')}")
            else:
                print(f"  Date: {date_entity.get('value')}")

            # Show time range if present
            if 'time_range' in date_entity:
                tr = date_entity['time_range']
                print(f"  Time Period: {tr.get('period', 'custom')}")
                print(f"  Time Range: {tr.get('start')} - {tr.get('end')}")

        print(f"  Metrics: {entities['metrics']}")

        if entities.get('time_of_day'):
            print(f"  Global Time-of-Day: {entities['time_of_day']}")

    print("\n" + "="*80)
    print("Testing Comparison Queries")
    print("="*80)

    for query in comparison_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        entities = extractor.extract_entities(query)

        if entities['dates'] and len(entities['dates']) > 1 and all(d.get('is_comparison') for d in entities['dates']):
            print("  Comparison detected:")
            for i, date_entity in enumerate(entities['dates']):
                print(f"    Period {i+1}:")
                print(f"      Raw Text: {date_entity['raw_text']}")
                if date_entity['type'] == 'date_range':
                    print(f"      Range: {date_entity['start']} to {date_entity['end']}")
                else:
                    print(f"      Date: {date_entity.get('value')}")
        else:
            print("  No comparison detected or something went wrong.")
            # Fallback to print regular date info
            if entities['dates']:
                date_entity = entities['dates'][0]
                print(f"  Date Type: {date_entity['type']}")
                if date_entity['type'] == 'date_range':
                    print(f"  Range: {date_entity['start']} to {date_entity['end']}")
                else:
                    print(f"  Date: {date_entity.get('value')}")


        print(f"  Metrics: {entities['metrics']}")


    print("\n" + "="*80)
    print("✓ Testing Complete!")
    print("="*80)