"""
Base Parser Class
Abstract base class for all oil & gas data parsers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

@dataclass
class ParsedData:
    """Standardized output structure for parsed data"""
    file_path: str
    file_type: str
    parser_name: str
    metadata: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class BaseParser(ABC):
    """Abstract base class for all parsers"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def can_parse(self) -> bool:
        """Check if this parser can handle the given file"""
        pass
        
    @abstractmethod
    def parse(self) -> ParsedData:
        """Parse the file and return structured data"""
        pass
        
    def validate_file(self) -> bool:
        """Basic file validation"""
        if not self.file_path.exists():
            self.logger.error(f"File does not exist: {self.file_path}")
            return False
            
        if not self.file_path.is_file():
            self.logger.error(f"Path is not a file: {self.file_path}")
            return False
            
        return True
        
    def get_file_info(self) -> Dict[str, Any]:
        """Get basic file information"""
        stat = self.file_path.stat()
        return {
            'name': self.file_path.name,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'extension': self.file_path.suffix.lower(),
            'modified_time': stat.st_mtime
        }
        
    def create_error_result(self, error_msg: str) -> ParsedData:
        """Create a standardized error result"""
        return ParsedData(
            file_path=str(self.file_path),
            file_type=self.file_path.suffix.lower(),
            parser_name=self.__class__.__name__,
            metadata=self.get_file_info(),
            error=error_msg
        ) 