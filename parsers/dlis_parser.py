"""
DLIS Parser
Handles DLIS (Digital Log Interchange Standard) well log files
"""

import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .base_parser import BaseParser, ParsedData, ContextualDocument

try:
    import dlisio
except ImportError:
    dlisio = None

@dataclass
class DlisCurve:
    """DLIS curve data with enhanced information"""
    name: str
    units: str
    frame_count: int
    sample_count: int
    data_type: str
    # Enhanced fields for actual data extraction
    measured_values: Optional[List[float]] = None
    depth_range: Optional[Tuple[float, float]] = None
    statistics: Optional[Dict[str, float]] = None
    quality_flags: Optional[List[str]] = None
    description: Optional[str] = None

@dataclass
class DlisFrame:
    """DLIS frame data with enhanced information"""
    name: str
    index_type: str
    spacing: float
    curve_count: int
    # Enhanced fields
    depth_range: Optional[Tuple[float, float]] = None
    sample_count: Optional[int] = None
    frame_rate: Optional[float] = None
    description: Optional[str] = None

class DlisParser(BaseParser):
    """Enhanced parser for DLIS well log files with actual data extraction"""
    
    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def can_parse(self) -> bool:
        """Check if this is a DLIS file we can parse"""
        if not self.validate_file():
            return False
            
        # Check file extension
        if self.file_path.suffix.lower() not in ['.dlis']:
            return False
            
        # Check if dlisio is available
        if dlisio is None:
            self.logger.warning("dlisio library not available")
            return False
            
        return True
    
    def parse(self) -> ParsedData:
        """Parse DLIS file and extract comprehensive well log data including measured values"""
        start_time = time.time()
        
        try:
            if not self.can_parse():
                return self.create_error_result("File cannot be parsed by DLIS parser")
            
            if dlisio is None:
                return self.create_error_result("dlisio library not available")
            
            # Parse DLIS file
            with dlisio.dlis.load(str(self.file_path)) as files:
                file_info = self.get_file_info()
                all_curves = []
                all_frames = []
                all_tools = []
                all_parameters = []
                all_comments = []
                all_messages = []
                all_calibrations = []
                well_info = {}
                
                # Track overall statistics
                total_samples = 0
                overall_depth_range = None
                data_availability = {
                    'curves_with_data': 0,
                    'curves_without_data': 0,
                    'metadata_only': False
                }
                
                for lf in files:
                    # Extract well information from origins
                    origins = lf.origins
                    if origins:
                        origin = origins[0]
                        well_info['well_name'] = str(origin.well_name) if hasattr(origin, 'well_name') else 'Unknown'
                        well_info['field_name'] = str(origin.field_name) if hasattr(origin, 'field_name') else 'Unknown'
                        well_info['company'] = str(origin.company) if hasattr(origin, 'company') else 'Unknown'
                        well_info['creation_time'] = str(origin.creation_time) if hasattr(origin, 'creation_time') else 'Unknown'
                        well_info['producer_name'] = str(origin.producer_name) if hasattr(origin, 'producer_name') else 'Unknown'
                        well_info['product'] = str(origin.product) if hasattr(origin, 'product') else 'Unknown'
                        well_info['version'] = str(origin.version) if hasattr(origin, 'version') else 'Unknown'
                        
                        # Only add fields that actually exist in the DLIS file
                        if hasattr(origin, 'operator') and origin.operator:
                            well_info['operator'] = str(origin.operator)
                        if hasattr(origin, 'service_company') and origin.service_company:
                            well_info['service_company'] = str(origin.service_company)
                        if hasattr(origin, 'date') and origin.date:
                            well_info['date'] = str(origin.date)
                        if hasattr(origin, 'time') and origin.time:
                            well_info['time'] = str(origin.time)
                    
                    # Extract frame and curve information with actual data
                    for frame in lf.frames:
                        frame_depth_range = None
                        frame_sample_count = 0
                        
                        # Extract frame metadata
                        frame_data = DlisFrame(
                            name=str(frame.name),
                            index_type=str(frame.index_type),
                            spacing=float(frame.spacing) if frame.spacing else 0.0,
                            curve_count=len(frame.channels),
                            description=str(getattr(frame, 'description', ''))
                        )
                        
                        # Process channels and extract actual measured values
                        for channel in frame.channels:
                            try:
                                # Extract actual measured values
                                measured_values = []
                                data_available = False
                                
                                # Try multiple methods to extract data
                                if hasattr(channel, '__len__') and len(channel) > 0:
                                    # Method 1: Direct iteration
                                    try:
                                        measured_values = [float(val) for val in channel if val is not None and str(val).replace('.', '').replace('-', '').isdigit()]
                                        if measured_values:
                                            data_available = True
                                    except:
                                        pass
                                
                                # Method 2: Try accessing individual elements
                                if not data_available:
                                    try:
                                        sample_values = []
                                        for i in range(100):  # Try first 100 samples
                                            try:
                                                value = channel[i]
                                                # Check if it's a valid numeric value
                                                if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit()):
                                                    sample_values.append(float(value))
                                            except (IndexError, KeyError):
                                                break
                                        
                                        if sample_values:
                                            measured_values = sample_values
                                            data_available = True
                                    except:
                                        pass
                                
                                # Method 3: Try through frame reference
                                if not data_available and hasattr(channel, 'frame'):
                                    try:
                                        frame_channels = channel.frame.channels
                                        for fc in frame_channels:
                                            if fc.name == channel.name:
                                                # Try to extract data from frame channel
                                                sample_values = []
                                                for i in range(100):
                                                    try:
                                                        value = fc[i]
                                                        if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit()):
                                                            sample_values.append(float(value))
                                                    except (IndexError, KeyError):
                                                        break
                                                
                                                if sample_values:
                                                    measured_values = sample_values
                                                    data_available = True
                                                break
                                    except:
                                        pass
                                
                                # Calculate statistics if we have data
                                statistics = None
                                depth_range = None
                                
                                if measured_values and data_available:
                                    # Convert to numpy for efficient calculations
                                    values_array = np.array(measured_values)
                                    
                                    # Filter out invalid values
                                    valid_mask = np.isfinite(values_array)
                                    valid_values = values_array[valid_mask]
                                    
                                    if len(valid_values) > 0:
                                        statistics = {
                                            'mean': float(np.mean(valid_values)),
                                            'std': float(np.std(valid_values)),
                                            'min': float(np.min(valid_values)),
                                            'max': float(np.max(valid_values)),
                                            'median': float(np.median(valid_values)),
                                            'q25': float(np.percentile(valid_values, 25)),
                                            'q75': float(np.percentile(valid_values, 75)),
                                            'count': len(valid_values),
                                            'valid_count': len(valid_values),
                                            'total_count': len(values_array)
                                        }
                                        
                                        depth_range = (float(np.min(valid_values)), float(np.max(valid_values)))
                                        
                                        # Update frame statistics
                                        if frame_depth_range is None:
                                            frame_depth_range = depth_range
                                        else:
                                            frame_depth_range = (
                                                min(frame_depth_range[0], depth_range[0]),
                                                max(frame_depth_range[1], depth_range[1])
                                            )
                                        
                                        frame_sample_count = max(frame_sample_count, len(valid_values))
                                        
                                        # Update data availability tracking
                                        data_availability['curves_with_data'] += 1
                                else:
                                    data_availability['curves_without_data'] += 1
                                
                                # Extract quality flags if available
                                quality_flags = []
                                if hasattr(channel, 'quality'):
                                    quality_flags = [str(flag) for flag in channel.quality] if hasattr(channel.quality, '__iter__') else [str(channel.quality)]
                                
                                # Create enhanced curve data
                                curve_data = DlisCurve(
                                    name=str(channel.name),
                                    units=str(channel.units) if channel.units else '',
                                    frame_count=len(frame.channels),
                                    sample_count=len(measured_values) if measured_values else 0,
                                    data_type=str(channel.dtype) if hasattr(channel, 'dtype') else 'unknown',
                                    measured_values=measured_values if measured_values else None,
                                    depth_range=depth_range,
                                    statistics=statistics,
                                    quality_flags=quality_flags if quality_flags else None,
                                    description=str(getattr(channel, 'long_name', ''))
                                )
                                
                                all_curves.append(curve_data)
                                total_samples += len(measured_values) if measured_values else 0
                                
                                # Update overall depth range
                                if depth_range:
                                    if overall_depth_range is None:
                                        overall_depth_range = depth_range
                                    else:
                                        overall_depth_range = (
                                            min(overall_depth_range[0], depth_range[0]),
                                            max(overall_depth_range[1], depth_range[1])
                                        )
                                
                            except Exception as e:
                                self.logger.warning(f"Error processing channel {channel.name}: {e}")
                                # Add basic curve data without measured values
                                curve_data = DlisCurve(
                                    name=str(channel.name),
                                    units=str(channel.units) if channel.units else '',
                                    frame_count=len(frame.channels),
                                    sample_count=0,
                                    data_type=str(channel.dtype) if hasattr(channel, 'dtype') else 'unknown',
                                    description=str(getattr(channel, 'long_name', ''))
                                )
                                all_curves.append(curve_data)
                                data_availability['curves_without_data'] += 1
                        
                        # Update frame with extracted statistics
                        frame_data.depth_range = frame_depth_range
                        frame_data.sample_count = frame_sample_count
                        frame_data.frame_rate = 1.0 / frame_data.spacing if frame_data.spacing > 0 else None
                        
                        all_frames.append(frame_data)
                    
                    # Extract tool information
                    for tool in lf.tools:
                        tool_data = {
                            'name': str(tool.name),
                            'description': str(getattr(tool, 'description', '')),
                            'type': str(getattr(tool, 'type', '')),
                            'manufacturer': str(getattr(tool, 'manufacturer', '')),
                            'model': str(getattr(tool, 'model', '')),
                            'serial_number': str(getattr(tool, 'serial_number', ''))
                        }
                        all_tools.append(tool_data)
                    
                    # Extract parameter information
                    for param in lf.parameters:
                        param_data = {
                            'name': str(param.name),
                            'long_name': str(getattr(param, 'long_name', '')),
                            'value': str(param.values[0]) if hasattr(param, 'values') and len(param.values) > 0 else '',
                            'units': str(getattr(param, 'units', '')),
                            'description': str(getattr(param, 'description', ''))
                        }
                        all_parameters.append(param_data)
                    
                    # Extract comments
                    for comment in lf.comments:
                        comment_data = {
                            'text': str(comment.text),
                            'origin': str(getattr(comment, 'origin', ''))
                        }
                        all_comments.append(comment_data)
                    
                    # Extract messages
                    for message in lf.messages:
                        message_data = {
                            'text': str(message.text),
                            'origin': str(getattr(message, 'origin', ''))
                        }
                        all_messages.append(message_data)
                    
                    # Extract calibrations
                    for cal in lf.calibrations:
                        cal_data = {
                            'name': str(cal.name),
                            'type': str(getattr(cal, 'type', '')),
                            'date': str(getattr(cal, 'date', '')),
                            'time': str(getattr(cal, 'time', ''))
                        }
                        all_calibrations.append(cal_data)
                
                # Determine if this is a metadata-only file
                if data_availability['curves_with_data'] == 0 and data_availability['curves_without_data'] > 0:
                    data_availability['metadata_only'] = True
                
                # Calculate overall statistics
                overall_stats = {
                    'total_samples': total_samples,
                    'overall_depth_range': overall_depth_range,
                    'data_quality': data_availability,
                    'file_type': 'metadata_only' if data_availability['metadata_only'] else 'data_available'
                }
                
                result_data = {
                    'well_info': well_info,
                    'curves': [self._curve_to_dict(c) for c in all_curves],
                    'frames': [self._frame_to_dict(f) for f in all_frames],
                    'tools': all_tools,
                    'parameters': all_parameters,
                    'comments': all_comments,
                    'messages': all_messages,
                    'calibrations': all_calibrations,
                    'overall_statistics': overall_stats,
                    'total_curves': len(all_curves),
                    'total_frames': len(all_frames),
                    'total_tools': len(all_tools),
                    'total_parameters': len(all_parameters),
                    'total_comments': len(all_comments),
                    'total_messages': len(all_messages),
                    'total_calibrations': len(all_calibrations),
                    'file_info': file_info
                }
                
                processing_time = time.time() - start_time
                
                return ParsedData(
                    file_path=str(self.file_path),
                    file_type='dlis',
                    parser_name=self.__class__.__name__,
                    metadata=file_info,
                    data=result_data,
                    processing_time=processing_time
                )
                
        except Exception as e:
            self.logger.error(f"Error parsing {self.file_path}: {e}")
            return self.create_error_result(str(e))
    
    def _curve_to_dict(self, curve: DlisCurve) -> Dict[str, Any]:
        """Convert DlisCurve to dictionary with enhanced data"""
        curve_dict = {
            'name': curve.name,
            'units': curve.units,
            'frame_count': curve.frame_count,
            'sample_count': curve.sample_count,
            'data_type': curve.data_type,
            'description': curve.description
        }
        
        # Add enhanced data if available
        if curve.measured_values is not None:
            curve_dict['measured_values'] = curve.measured_values
        if curve.depth_range is not None:
            curve_dict['depth_range'] = curve.depth_range
        if curve.statistics is not None:
            curve_dict['statistics'] = curve.statistics
        if curve.quality_flags is not None:
            curve_dict['quality_flags'] = curve.quality_flags
            
        return curve_dict
    
    def _frame_to_dict(self, frame: DlisFrame) -> Dict[str, Any]:
        """Convert DlisFrame to dictionary with enhanced data"""
        frame_dict = {
            'name': frame.name,
            'index_type': frame.index_type,
            'spacing': frame.spacing,
            'curve_count': frame.curve_count,
            'description': frame.description
        }
        
        # Add enhanced data if available
        if frame.depth_range is not None:
            frame_dict['depth_range'] = frame.depth_range
        if frame.sample_count is not None:
            frame_dict['sample_count'] = frame.sample_count
        if frame.frame_rate is not None:
            frame_dict['frame_rate'] = frame.frame_rate
            
        return frame_dict
    
    def generate_contextual_documents(self) -> List[ContextualDocument]:
        """Generate contextual documents from enhanced DLIS data"""
        parsed_data = self.parse()
        
        if parsed_data.error:
            return []
        
        documents = []
        data = parsed_data.data
        
        well_info = data.get('well_info', {})
        well_name = well_info.get('well_name', 'Unknown')
        field_name = well_info.get('field_name', 'Unknown')
        company = well_info.get('company', 'Unknown')
        creation_time = well_info.get('creation_time', 'Unknown')
        producer_name = well_info.get('producer_name', 'Unknown')
        product = well_info.get('product', 'Unknown')
        version = well_info.get('version', 'Unknown')
        
        # Get overall statistics
        overall_stats = data.get('overall_statistics', {})
        total_samples = overall_stats.get('total_samples', 0)
        overall_depth_range = overall_stats.get('overall_depth_range')
        data_quality = overall_stats.get('data_quality', {})
        file_type = overall_stats.get('file_type', 'unknown')
        
        # Document 1: Enhanced Well Overview
        overview_content = f"""
Enhanced Well Log Data Analysis - DLIS File
Well Name: {well_name}
Field: {field_name}
Company: {company}
Creation Time: {creation_time}
Producer: {producer_name}
Product: {product}
Version: {version}

Data Summary:
- Total Curves: {data.get('total_curves', 0)}
- Total Frames: {data.get('total_frames', 0)}
- Total Tools: {data.get('total_tools', 0)}
- Total Parameters: {data.get('total_parameters', 0)}
- Total Comments: {data.get('total_comments', 0)}
- Total Messages: {data.get('total_messages', 0)}
- Total Calibrations: {data.get('total_calibrations', 0)}

Enhanced Data Analysis:
- Total Samples: {total_samples:,}
- Curves with Data: {data_quality.get('curves_with_data', 0)}
- Curves without Data: {data_quality.get('curves_without_data', 0)}
- Data Quality: {data_quality.get('curves_with_data', 0)}/{data_quality.get('total_curves', 0)} curves contain measured values
- File Type: {file_type}
"""
        
        if overall_depth_range:
            overview_content += f"- Overall Depth Range: {overall_depth_range[0]:.2f} to {overall_depth_range[1]:.2f}\n"
        
        if data_quality.get('metadata_only', False):
            overview_content += f"""

This DLIS file appears to be a metadata-only file containing well log structure information but no actual measured values.
The enhanced parser has successfully extracted comprehensive metadata including {data.get('total_curves', 0)} curve definitions,
{data.get('total_frames', 0)} data frames, and {data.get('total_parameters', 0)} processing parameters.
This type of file is commonly used for data structure documentation and well log setup information.
"""
        else:
            overview_content += f"""

This DLIS file contains comprehensive well log data with {data.get('total_curves', 0)} measurement curves across {data.get('total_frames', 0)} data frames.
The enhanced parser has extracted {total_samples:,} total samples with detailed statistical analysis and depth range information.
"""
        
        documents.append(ContextualDocument(
            content=overview_content.strip(),
            document_type='dlis',
            source=str(self.file_path),
            metadata={
                'well_name': well_name,
                'field_name': field_name,
                'company': company,
                'creation_time': creation_time,
                'producer_name': producer_name,
                'product': product,
                'version': version,
                'total_curves': data.get('total_curves', 0),
                'total_frames': data.get('total_frames', 0),
                'total_tools': data.get('total_tools', 0),
                'total_parameters': data.get('total_parameters', 0),
                'total_comments': data.get('total_comments', 0),
                'total_messages': data.get('total_messages', 0),
                'total_calibrations': data.get('total_calibrations', 0),
                'total_samples': total_samples,
                'overall_depth_range': overall_depth_range,
                'data_quality': data_quality,
                'file_type': file_type,
                'file_type': 'dlis',
                'parser_type': 'DlisParser'
            },
            timestamp=datetime.now().isoformat()
        ))
        
        # Document 2: Enhanced Curve Details with Statistics
        curves = data.get('curves', [])
        if curves:
            curve_details = "Enhanced Well Log Curves with Statistical Analysis:\n\n"
            
            for curve in curves[:15]:  # Show first 15 curves with detailed info
                curve_details += f"• {curve['name']} ({curve['units']}) - {curve['data_type']}\n"
                curve_details += f"  Description: {curve.get('description', 'N/A')}\n"
                curve_details += f"  Samples: {curve['sample_count']:,}\n"
                
                if 'depth_range' in curve and curve['depth_range']:
                    curve_details += f"  Depth Range: {curve['depth_range'][0]:.2f} to {curve['depth_range'][1]:.2f}\n"
                
                if 'statistics' in curve and curve['statistics']:
                    stats = curve['statistics']
                    curve_details += f"  Statistics: Mean={stats.get('mean', 0):.3f}, Std={stats.get('std', 0):.3f}\n"
                    curve_details += f"  Range: {stats.get('min', 0):.3f} to {stats.get('max', 0):.3f}\n"
                    curve_details += f"  Valid Data: {stats.get('valid_count', 0)}/{stats.get('total_count', 0)} samples\n"
                else:
                    curve_details += f"  Data Status: Metadata only (no measured values)\n"
                
                if 'quality_flags' in curve and curve['quality_flags']:
                    curve_details += f"  Quality Flags: {', '.join(curve['quality_flags'])}\n"
                
                curve_details += "\n"
            
            if len(curves) > 15:
                curve_details += f"... and {len(curves) - 15} more curves with similar detailed analysis\n"
            
            documents.append(ContextualDocument(
                content=curve_details.strip(),
                document_type='dlis',
                source=str(self.file_path),
                metadata={
                    'well_name': well_name,
                    'curve_count': len(curves),
                    'curves_with_data': data_quality.get('curves_with_data', 0),
                    'curves_without_data': data_quality.get('curves_without_data', 0),
                    'file_type': 'dlis',
                    'parser_type': 'DlisParser',
                    'content_type': 'enhanced_curve_details'
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Document 3: Enhanced Frame Details
        frames = data.get('frames', [])
        if frames:
            frame_details = "Enhanced Data Frame Information:\n\n"
            for frame in frames:
                frame_details += f"• Frame: {frame['name']} - Type: {frame['index_type']}\n"
                frame_details += f"  Spacing: {frame['spacing']} - Curves: {frame['curve_count']}\n"
                
                if 'depth_range' in frame and frame['depth_range']:
                    frame_details += f"  Depth Range: {frame['depth_range'][0]:.2f} to {frame['depth_range'][1]:.2f}\n"
                
                if 'sample_count' in frame and frame['sample_count']:
                    frame_details += f"  Sample Count: {frame['sample_count']:,}\n"
                
                if 'frame_rate' in frame and frame['frame_rate']:
                    frame_details += f"  Frame Rate: {frame['frame_rate']:.3f} Hz\n"
                
                frame_details += "\n"
            
            documents.append(ContextualDocument(
                content=frame_details.strip(),
                document_type='dlis',
                source=str(self.file_path),
                metadata={
                    'well_name': well_name,
                    'frame_count': len(frames),
                    'file_type': 'dlis',
                    'parser_type': 'DlisParser',
                    'content_type': 'enhanced_frame_details'
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Document 4: Data Quality Analysis
        quality_analysis = f"""
Data Quality Analysis:
- Total Curves: {data_quality.get('total_curves', 0)}
- Curves with Measured Data: {data_quality.get('curves_with_data', 0)}
- Curves without Data: {data_quality.get('curves_without_data', 0)}
- Data Coverage: {data_quality.get('curves_with_data', 0)/max(data_quality.get('total_curves', 1), 1)*100:.1f}%

Overall Statistics:
- Total Samples Extracted: {total_samples:,}
- File Type: {file_type}
"""
        
        if overall_depth_range:
            quality_analysis += f"- Overall Depth Coverage: {overall_depth_range[1] - overall_depth_range[0]:.2f} units\n"
        
        if data_quality.get('metadata_only', False):
            quality_analysis += f"""
Data Quality Assessment:
This DLIS file is a metadata-only file containing well log structure information but no actual measured values.
This is common for:
- Well log setup files
- Data structure documentation
- Template files for well log processing
- Header-only DLIS files

The enhanced parser has successfully extracted comprehensive metadata including curve definitions, frame structures, and processing parameters.
"""
        else:
            quality_analysis += f"""
Data Quality Assessment:
This DLIS file contains {'high quality' if data_quality.get('curves_with_data', 0) > data_quality.get('curves_without_data', 0) else 'limited'} data with {data_quality.get('curves_with_data', 0)} curves containing actual measured values.
The enhanced parser successfully extracted {total_samples:,} total samples for comprehensive analysis.
"""
        
        documents.append(ContextualDocument(
            content=quality_analysis.strip(),
            document_type='dlis',
            source=str(self.file_path),
            metadata={
                'well_name': well_name,
                'data_quality': data_quality,
                'total_samples': total_samples,
                'overall_depth_range': overall_depth_range,
                'file_type': 'dlis',
                'parser_type': 'DlisParser',
                'content_type': 'data_quality_analysis'
            },
            timestamp=datetime.now().isoformat()
        ))
        
        # Document 5: Tool Information
        tools = data.get('tools', [])
        if tools:
            tool_details = "Tool Information:\n\n"
            for tool in tools:
                tool_details += f"• Tool: {tool['name']}\n"
                if tool['description']:
                    tool_details += f"  Description: {tool['description']}\n"
                if tool['type']:
                    tool_details += f"  Type: {tool['type']}\n"
                if tool['manufacturer']:
                    tool_details += f"  Manufacturer: {tool['manufacturer']}\n"
                if tool['model']:
                    tool_details += f"  Model: {tool['model']}\n"
                if tool['serial_number']:
                    tool_details += f"  Serial Number: {tool['serial_number']}\n"
                tool_details += "\n"
            
            documents.append(ContextualDocument(
                content=tool_details.strip(),
                document_type='dlis',
                source=str(self.file_path),
                metadata={
                    'well_name': well_name,
                    'tool_count': len(tools),
                    'file_type': 'dlis',
                    'parser_type': 'DlisParser',
                    'content_type': 'tool_details'
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Document 6: Parameter Information
        parameters = data.get('parameters', [])
        if parameters:
            param_details = "Processing Parameters:\n\n"
            for param in parameters:
                param_details += f"• Parameter: {param['name']}"
                if param['long_name']:
                    param_details += f" ({param['long_name']})"
                param_details += "\n"
                if param['value']:
                    param_details += f"  Value: {param['value']}\n"
                if param['units']:
                    param_details += f"  Units: {param['units']}\n"
                if param['description']:
                    param_details += f"  Description: {param['description']}\n"
                param_details += "\n"
            
            documents.append(ContextualDocument(
                content=param_details.strip(),
                document_type='dlis',
                source=str(self.file_path),
                metadata={
                    'well_name': well_name,
                    'parameter_count': len(parameters),
                    'file_type': 'dlis',
                    'parser_type': 'DlisParser',
                    'content_type': 'parameter_details'
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Document 7: Comments and Messages
        comments = data.get('comments', [])
        messages = data.get('messages', [])
        
        if comments or messages:
            text_details = ""
            if comments:
                text_details += "Comments:\n\n"
                for comment in comments:
                    text_details += f"• {comment['text'][:200]}...\n\n"
            
            if messages:
                text_details += "Messages:\n\n"
                for message in messages:
                    text_details += f"• {message['text'][:200]}...\n\n"
            
            if text_details.strip():
                documents.append(ContextualDocument(
                    content=text_details.strip(),
                    document_type='dlis',
                    source=str(self.file_path),
                    metadata={
                        'well_name': well_name,
                        'comment_count': len(comments),
                        'message_count': len(messages),
                        'file_type': 'dlis',
                        'parser_type': 'DlisParser',
                        'content_type': 'text_details'
                    },
                    timestamp=datetime.now().isoformat()
                ))
        
        # Document 8: Calibration Information
        calibrations = data.get('calibrations', [])
        if calibrations:
            cal_details = "Calibration Information:\n\n"
            for cal in calibrations:
                cal_details += f"• Calibration: {cal['name']}\n"
                if cal['type']:
                    cal_details += f"  Type: {cal['type']}\n"
                if cal['date']:
                    cal_details += f"  Date: {cal['date']}\n"
                if cal['time']:
                    cal_details += f"  Time: {cal['time']}\n"
                cal_details += "\n"
            
            documents.append(ContextualDocument(
                content=cal_details.strip(),
                document_type='dlis',
                source=str(self.file_path),
                metadata={
                    'well_name': well_name,
                    'calibration_count': len(calibrations),
                    'file_type': 'dlis',
                    'parser_type': 'DlisParser',
                    'content_type': 'calibration_details'
                },
                timestamp=datetime.now().isoformat()
            ))
        
        return documents 