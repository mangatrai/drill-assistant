"""
LAS Parser
Handles LAS (Log ASCII Standard) well log files
Based on real analysis: NO_15_9-F-14_KLOGH_NEW.las structure
"""

import lasio
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from .base_parser import BaseParser, ParsedData

class LasParser(BaseParser):
    """Parser for LAS well log files"""
    
    def can_parse(self) -> bool:
        """Check if this is a LAS file we can parse"""
        if not self.validate_file():
            return False
            
        # Check file extension
        if self.file_path.suffix.lower() not in ['.las', '.las']:
            return False
            
        # Try to read with lasio to validate
        try:
            las = lasio.read(str(self.file_path))
            return True
        except Exception:
            return False
    
    def parse(self) -> ParsedData:
        """Parse LAS file and extract well data"""
        start_time = time.time()
        
        try:
            if not self.can_parse():
                return self.create_error_result("File cannot be parsed by LAS parser")
            
            # Read LAS file
            las = lasio.read(str(self.file_path))
            
            # Extract well information
            well_info = self._extract_well_info(las)
            
            # Extract curve information
            curve_info = self._extract_curve_info(las)
            
            # Extract data samples
            data_samples = self._extract_data_samples(las)
            
            # Create structured result
            result_data = {
                'well_info': well_info,
                'curve_info': curve_info,
                'data_samples': data_samples,
                'depth_range': self._get_depth_range(las),
                'sections': self._get_sections(las)
            }
            
            processing_time = time.time() - start_time
            
            return ParsedData(
                file_path=str(self.file_path),
                file_type='las',
                parser_name=self.__class__.__name__,
                metadata=self.get_file_info(),
                data=result_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing {self.file_path}: {e}")
            return self.create_error_result(str(e))
    
    def _extract_well_info(self, las: lasio.LASFile) -> Dict[str, str]:
        """Extract well information from LAS file"""
        well_info = {}
        
        # Standard well parameters
        well_params = [
            'STRT', 'STOP', 'STEP', 'NULL', 'COMP', 'WELL', 'FLD', 
            'LOC', 'CTRY', 'STAT', 'CNTY', 'SRVC', 'DATE', 'API', 'UWI'
        ]
        
        for param in well_params:
            try:
                if hasattr(las.well, param):
                    value = getattr(las.well, param)
                    if value and str(value).strip():
                        well_info[param] = str(value).strip()
            except Exception:
                continue
        
        return well_info
    
    def _extract_curve_info(self, las: lasio.LASFile) -> Dict[str, Dict[str, Any]]:
        """Extract information about each curve"""
        curve_info = {}
        
        for curve in las.curves:
            try:
                curve_data = {
                    'mnemonic': curve.mnemonic,
                    'unit': curve.unit if hasattr(curve, 'unit') else '',
                    'description': curve.descr if hasattr(curve, 'descr') else '',
                    'data_type': str(type(curve.data[0])) if len(curve.data) > 0 else 'unknown',
                    'data_count': len(curve.data),
                    'null_value': curve.null if hasattr(curve, 'null') else None
                }
                
                # Add data statistics if available
                if len(curve.data) > 0:
                    valid_data = [x for x in curve.data if x is not None and str(x) != str(curve.null)]
                    if valid_data:
                        curve_data['statistics'] = {
                            'min': min(valid_data),
                            'max': max(valid_data),
                            'mean': sum(valid_data) / len(valid_data),
                            'valid_count': len(valid_data)
                        }
                
                curve_info[curve.mnemonic] = curve_data
                
            except Exception as e:
                self.logger.warning(f"Error extracting curve info for {curve.mnemonic}: {e}")
                continue
        
        return curve_info
    
    def _extract_data_samples(self, las: lasio.LASFile) -> Dict[str, List]:
        """Extract sample data from curves"""
        data_samples = {}
        
        for curve in las.curves:
            try:
                # Get first 10 non-null values as samples
                samples = []
                null_val = curve.null if hasattr(curve, 'null') else None
                
                for value in curve.data[:50]:  # Check first 50 values
                    if value is not None and str(value) != str(null_val):
                        samples.append(value)
                        if len(samples) >= 10:  # Get up to 10 samples
                            break
                
                data_samples[curve.mnemonic] = samples
                
            except Exception as e:
                self.logger.warning(f"Error extracting data samples for {curve.mnemonic}: {e}")
                continue
        
        return data_samples
    
    def _get_depth_range(self, las: lasio.LASFile) -> Dict[str, float]:
        """Get depth range information"""
        depth_range = {}
        
        try:
            # Find depth curve (usually first curve)
            depth_curve = None
            for curve in las.curves:
                if curve.mnemonic.upper() in ['DEPTH', 'MD', 'TVD', 'TVDSS']:
                    depth_curve = curve
                    break
            
            if depth_curve is None and len(las.curves) > 0:
                depth_curve = las.curves[0]  # Assume first curve is depth
            
            if depth_curve and len(depth_curve.data) > 0:
                valid_data = [x for x in depth_curve.data if x is not None]
                if valid_data:
                    depth_range = {
                        'min_depth': min(valid_data),
                        'max_depth': max(valid_data),
                        'depth_units': depth_curve.unit if hasattr(depth_curve, 'unit') else '',
                        'depth_curve': depth_curve.mnemonic
                    }
        
        except Exception as e:
            self.logger.warning(f"Error getting depth range: {e}")
        
        return depth_range
    
    def _get_sections(self, las: lasio.LASFile) -> List[str]:
        """Get available LAS sections"""
        sections = []
        
        try:
            # Check which sections are available
            if hasattr(las, 'well') and las.well:
                sections.append('W')
            if hasattr(las, 'params') and las.params:
                sections.append('P')
            if hasattr(las, 'curves') and las.curves:
                sections.append('C')
            if hasattr(las, 'data') and las.data is not None:
                sections.append('A')
            if hasattr(las, 'other') and las.other:
                sections.append('O')
            if hasattr(las, 'version') and las.version:
                sections.append('V')
                
        except Exception as e:
            self.logger.warning(f"Error getting sections: {e}")
        
        return sections 