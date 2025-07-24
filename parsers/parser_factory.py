"""
Parser Factory
Automatically selects and creates appropriate parsers for different file types
"""

from pathlib import Path
from typing import Dict, Type, Optional, List
import logging
from .base_parser import BaseParser
from .ascii_parser import AsciiParser
from .las_parser import LasParser
from .dlis_parser import DlisParser
from .segy_parser import SegyParser
from .pdf_parser import PdfParser
from .excel_parser import ExcelParser
from .text_parser import TextParser
from .dat_parser import DatParser

class ParserFactory:
    """Factory for creating appropriate parsers based on file type"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # File extension to parser mapping
        self.extension_parsers: Dict[str, Type[BaseParser]] = {
            '.asc': AsciiParser,
            '.txt': TextParser,
            '.las': LasParser,
            '.dlis': DlisParser,
            '.segy': SegyParser,
            '.sgy': SegyParser,
            '.pdf': PdfParser,
            '.xlsx': ExcelParser,
            '.xls': ExcelParser,
            '.dat': DatParser
        }
        
        # Parser priority order (for files without extensions)
        self.priority_parsers = [
            AsciiParser,  # Well survey files often have no extension
            TextParser,
            DatParser
        ]
    
    def create_parser(self, file_path: str) -> Optional[BaseParser]:
        """Create appropriate parser for the given file"""
        try:
            file_path = Path(file_path)
            
            # Get file extension
            extension = file_path.suffix.lower()
            
            # Try extension-based parser first
            if extension in self.extension_parsers:
                parser_class = self.extension_parsers[extension]
                parser = parser_class(str(file_path))
                
                # Validate that the parser can actually handle this file
                if parser.can_parse():
                    self.logger.info(f"Selected {parser_class.__name__} for {file_path.name}")
                    return parser
                else:
                    self.logger.warning(f"{parser_class.__name__} cannot parse {file_path.name}")
            
            # For files without extensions, try priority parsers
            if not extension:
                for parser_class in self.priority_parsers:
                    parser = parser_class(str(file_path))
                    if parser.can_parse():
                        self.logger.info(f"Selected {parser_class.__name__} for {file_path.name} (no extension)")
                        return parser
            
            # No suitable parser found
            self.logger.warning(f"No suitable parser found for {file_path.name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating parser for {file_path}: {e}")
            return None
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self.extension_parsers.keys())
    
    def get_parser_for_extension(self, extension: str) -> Optional[Type[BaseParser]]:
        """Get parser class for specific extension"""
        return self.extension_parsers.get(extension.lower())
    
    def register_parser(self, extension: str, parser_class: Type[BaseParser]):
        """Register a new parser for an extension"""
        self.extension_parsers[extension.lower()] = parser_class
        self.logger.info(f"Registered {parser_class.__name__} for extension {extension}")
    
    def get_parser_info(self) -> Dict[str, str]:
        """Get information about available parsers"""
        return {
            ext: parser_class.__name__ 
            for ext, parser_class in self.extension_parsers.items()
        } 