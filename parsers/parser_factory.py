"""
Parser Factory
Automatically selects and creates appropriate parsers for different file types
"""

from pathlib import Path
from typing import Dict, Type, Optional, List
import logging
from .base_parser import BaseParser
from .dlis_parser import DlisParser
from .segy_parser import SegyParser
from .unstructured_parser import UnstructuredParser

class ParserFactory:
    """Factory for creating appropriate parsers based on file type"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # File extension to parser mapping
        self.extension_parsers: Dict[str, Type[BaseParser]] = {
            # Industry-specific formats (keep custom parsers)
            '.dlis': DlisParser,
            '.segy': SegyParser,
            '.sgy': SegyParser,
            
            # Plain text files (use Unstructured SDK)
            '.asc': UnstructuredParser,
            '.las': UnstructuredParser,
            '.dat': UnstructuredParser,
            '.txt': UnstructuredParser,
            
            # General document formats (use Unstructured SDK)
            '.pdf': UnstructuredParser,
            '.docx': UnstructuredParser,
            '.doc': UnstructuredParser,
            '.pptx': UnstructuredParser,
            '.ppt': UnstructuredParser,
            '.xlsx': UnstructuredParser,
            '.xls': UnstructuredParser,
            '.csv': UnstructuredParser,
            '.html': UnstructuredParser,
            '.htm': UnstructuredParser,
            '.md': UnstructuredParser,
            '.png': UnstructuredParser,
            '.jpg': UnstructuredParser,
            '.jpeg': UnstructuredParser,
            '.tiff': UnstructuredParser,
            '.bmp': UnstructuredParser,
            '.gif': UnstructuredParser,
            '.eml': UnstructuredParser,
            '.msg': UnstructuredParser
        }
        
        # Parser priority order (for files without extensions)
        self.priority_parsers = [
            UnstructuredParser  # Universal document parser for plain text files
        ]
    
    def create_parser(self, file_path: str) -> Optional[BaseParser]:
        """Create appropriate parser for the given file
        
        If the file extension is not in the mapping, try all priority parsers (even if the file has an extension).
        Only if none of the priority parsers can handle it, log a warning and return None.
        """
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()

            # Try extension-based parser first
            if extension in self.extension_parsers:
                parser_class = self.extension_parsers[extension]
                parser = parser_class(str(file_path))
                if parser.can_parse():
                    self.logger.info(f"Selected {parser_class.__name__} for {file_path.name}")
                    return parser
                else:
                    self.logger.warning(f"{parser_class.__name__} cannot parse {file_path.name}")

            # For files with no extension OR extension not in mapping, try priority parsers
            # (This is the new behavior: always try priority parsers if extension is not mapped)
            for parser_class in self.priority_parsers:
                parser = parser_class(str(file_path))
                if parser.can_parse():
                    self.logger.info(f"Selected {parser_class.__name__} for {file_path.name} (priority parser)")
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