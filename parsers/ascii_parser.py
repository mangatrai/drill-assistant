"""
ASCII Parser
Handles ASCII files with header key-value pairs and tabular data
Based on real analysis: WLC_PETRO_COMPUTED_1_INF_1.ASC structure
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time
from .base_parser import BaseParser, ParsedData

class AsciiParser(BaseParser):
    """Parser for ASCII files with header metadata and tabular data"""
    
    def can_parse(self) -> bool:
        """Check if this is an ASCII file we can parse"""
        if not self.validate_file():
            return False
            
        # Check file extension
        if self.file_path.suffix.lower() not in ['.asc', '.txt']:
            return False
            
        # Check if file contains header patterns
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(10)]
                
            # Look for key-value header patterns
            header_pattern = re.compile(r'^[A-Z\s]+:\s*[A-Za-z0-9\s\-\.]+$')
            return any(header_pattern.match(line) for line in first_lines if line)
            
        except Exception:
            return False
    
    def parse(self) -> ParsedData:
        """Parse ASCII file with headers and data"""
        start_time = time.time()
        
        try:
            if not self.can_parse():
                return self.create_error_result("File cannot be parsed by ASCII parser")
            
            # Read file content
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse headers and data
            headers = self._extract_headers(content)
            data_sections = self._extract_data_sections(content)
            
            # Create structured result
            result_data = {
                'headers': headers,
                'data_sections': data_sections,
                'file_structure': self._analyze_file_structure(content)
            }
            
            processing_time = time.time() - start_time
            
            return ParsedData(
                file_path=str(self.file_path),
                file_type='ascii',
                parser_name=self.__class__.__name__,
                metadata=self.get_file_info(),
                data=result_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing {self.file_path}: {e}")
            return self.create_error_result(str(e))
    
    def _extract_headers(self, content: str) -> Dict[str, str]:
        """Extract key-value headers from content"""
        headers = {}
        
        # Pattern for key-value pairs like "OPERATOR : STATOIL PETROLEUM AS"
        header_pattern = re.compile(r'^([A-Z\s]+)\s*:\s*(.+)$', re.MULTILINE)
        
        for match in header_pattern.finditer(content):
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key and value:
                headers[key] = value
        
        return headers
    
    def _extract_data_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract tabular data sections"""
        sections = []
        
        # Split content into lines
        lines = content.split('\n')
        
        current_section = None
        in_data_section = False
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for section separators (lines with dashes)
            if re.match(r'^-+$', line):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    'start_line': line_num + 1,
                    'header': None,
                    'columns': [],
                    'data': []
                }
                in_data_section = False
                continue
            
            # Check for column headers
            if current_section and not in_data_section:
                if 'MNEM' in line or 'UNIT' in line or 'Curve' in line:
                    current_section['header'] = line
                    current_section['columns'] = self._parse_columns(line)
                    in_data_section = True
                    continue
            
            # Parse data rows
            if current_section and in_data_section:
                if line and not line.startswith('-'):
                    row_data = self._parse_data_row(line, current_section['columns'])
                    if row_data:
                        current_section['data'].append(row_data)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _parse_columns(self, header_line: str) -> List[str]:
        """Parse column names from header line"""
        # Split by whitespace and filter out empty strings
        columns = [col.strip() for col in re.split(r'\s+', header_line) if col.strip()]
        return columns
    
    def _parse_data_row(self, line: str, columns: List[str]) -> Optional[Dict[str, str]]:
        """Parse a data row into column-value pairs"""
        try:
            # Split by whitespace
            values = re.split(r'\s+', line.strip())
            
            if len(values) >= len(columns):
                row_data = {}
                for i, col in enumerate(columns):
                    if i < len(values):
                        row_data[col] = values[i]
                    else:
                        row_data[col] = ''
                return row_data
        except Exception:
            pass
        return None
    
    def _analyze_file_structure(self, content: str) -> Dict[str, Any]:
        """Analyze overall file structure"""
        lines = content.split('\n')
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len([l for l in lines if l.strip()]),
            'header_lines': len([l for l in lines if ':' in l and re.match(r'^[A-Z\s]+:', l)]),
            'data_lines': len([l for l in lines if l.strip() and not l.startswith('-') and ':' not in l]),
            'separator_lines': len([l for l in lines if re.match(r'^-+$', l)]),
            'estimated_columns': self._estimate_column_count(content)
        }
    
    def _estimate_column_count(self, content: str) -> int:
        """Estimate number of columns in data sections"""
        lines = content.split('\n')
        max_cols = 0
        
        for line in lines:
            if line.strip() and not line.startswith('-') and ':' not in line:
                cols = len(re.split(r'\s+', line.strip()))
                max_cols = max(max_cols, cols)
        
        return max_cols 