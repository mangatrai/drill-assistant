"""
Modular Data Parser
Main orchestration class for parsing oil & gas data using modular parser system
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time
from datetime import datetime

from parsers import ParserFactory
from parsers.base_parser import ParsedData, ContextualDocument

class ModularDataParser:
    """Main parser class using modular architecture"""
    
    def __init__(self, data_path: str, output_dir: str = "parsed_data"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.parser_factory = ParserFactory()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Results storage
        self.parsed_results = []
        self.contextual_documents = []
        self.parser_stats = defaultdict(int)
        self.error_files = []
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def parse_all_data(self) -> Dict[str, Any]:
        """Parse all data files using modular parser system"""
        start_time = time.time()
        
        self.logger.info("🚀 Starting modular data parsing...")
        self.logger.info(f"📁 Data path: {self.data_path}")
        self.logger.info(f"📊 Output directory: {self.output_dir}")
        
        # Discover all files
        all_files = self._discover_files()
        self.logger.info(f"📋 Found {len(all_files)} files to process")
        
        # Parse each file and generate contextual documents
        for file_path in all_files:
            self._parse_single_file(file_path)
        
        # Generate summary
        summary = self._generate_summary(start_time)
        
        # Save results
        self._save_results(summary)
        
        # Save contextual documents
        self._save_contextual_documents()
        
        return summary
    
    def _discover_files(self) -> List[Path]:
        """Discover all files in the data directory"""
        files = []
        
        if not self.data_path.exists():
            self.logger.error(f"Data path does not exist: {self.data_path}")
            return files
        
        # Recursively find all files
        for file_path in self.data_path.rglob('*'):
            if file_path.is_file():
                files.append(file_path)
        
        # Sort by priority (based on analysis findings)
        priority_order = ['.asc', '.dlis', '.pdf', '', '.xlsx', '.las', '.txt', '.dat', '.segy']
        
        def get_priority(file_path):
            ext = file_path.suffix.lower()
            try:
                return priority_order.index(ext)
            except ValueError:
                return len(priority_order)  # Lowest priority
        
        files.sort(key=get_priority)
        
        return files
    
    def _parse_single_file(self, file_path: Path):
        """Parse a single file using appropriate parser"""
        self.logger.info(f"🔍 Processing: {file_path.name}")
        
        try:
            # Create appropriate parser
            parser = self.parser_factory.create_parser(str(file_path))
            
            if parser is None:
                self.logger.warning(f"❌ No parser found for: {file_path.name}")
                self.error_files.append({
                    'file': str(file_path),
                    'error': 'No suitable parser found'
                })
                return
            
            # Parse the file and generate contextual documents
            result = parser.parse()
            
            # Store result
            self.parsed_results.append(result)
            self.parser_stats[result.parser_name] += 1
            
            # Log result
            if result.error:
                self.logger.error(f"❌ Error parsing {file_path.name}: {result.error}")
                self.error_files.append({
                    'file': str(file_path),
                    'error': result.error
                })
            else:
                self.logger.info(f"✅ Successfully parsed {file_path.name} with {result.parser_name}")
                
                # Generate contextual documents
                contextual_docs = parser.generate_contextual_documents()
                self.contextual_documents.extend(contextual_docs)
                self.logger.info(f"📄 Generated {len(contextual_docs)} contextual documents from {file_path.name}")
                
        except Exception as e:
            self.logger.error(f"❌ Unexpected error parsing {file_path.name}: {e}")
            self.error_files.append({
                'file': str(file_path),
                'error': str(e)
            })
    
    def _generate_summary(self, start_time: float) -> Dict[str, Any]:
        """Generate parsing summary"""
        total_time = time.time() - start_time
        
        # Count successful vs failed
        successful = len([r for r in self.parsed_results if not r.error])
        failed = len(self.error_files)
        
        # Group by file type
        file_type_stats = defaultdict(int)
        for result in self.parsed_results:
            if not result.error:
                file_type_stats[result.file_type] += 1
        
        # Group by parser
        parser_stats = dict(self.parser_stats)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files_processed': len(self.parsed_results) + len(self.error_files),
            'successful_parses': successful,
            'failed_parses': failed,
            'success_rate': successful / (successful + failed) if (successful + failed) > 0 else 0,
            'total_processing_time': total_time,
            'average_time_per_file': total_time / (successful + failed) if (successful + failed) > 0 else 0,
            'file_type_breakdown': dict(file_type_stats),
            'parser_usage': parser_stats,
            'error_summary': self._summarize_errors(),
            'contextual_documents': {
                'total_documents': len(self.contextual_documents),
                'document_types': self._count_document_types()
            }
        }
        
        return summary
    
    def _summarize_errors(self) -> Dict[str, int]:
        """Summarize error types"""
        error_types = defaultdict(int)
        for error_info in self.error_files:
            error_msg = error_info['error']
            if 'No suitable parser' in error_msg:
                error_types['no_parser'] += 1
            elif 'cannot be parsed' in error_msg:
                error_types['parser_rejection'] += 1
            else:
                error_types['other'] += 1
        return dict(error_types)
    
    def _count_document_types(self) -> Dict[str, int]:
        """Count contextual documents by type"""
        doc_types = defaultdict(int)
        for doc in self.contextual_documents:
            doc_types[doc.document_type] += 1
        return dict(doc_types)
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save parsing results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"parsing_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'parsed_data': [self._serialize_result(r) for r in self.parsed_results],
                'errors': self.error_files
            }, f, indent=2, default=str)
        
        # Save summary only
        summary_file = self.output_dir / f"parsing_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to:")
        self.logger.info(f"   📄 Detailed: {results_file}")
        self.logger.info(f"   📊 Summary: {summary_file}")
    
    def _save_contextual_documents(self):
        """Save contextual documents to file"""
        if not self.contextual_documents:
            self.logger.info("📄 No contextual documents to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        documents_file = self.output_dir / f"contextual_documents_{timestamp}.json"
        
        # Serialize contextual documents
        serialized_docs = []
        for doc in self.contextual_documents:
            serialized_docs.append({
                'content': doc.content,
                'metadata': doc.metadata,
                'document_type': doc.document_type,
                'source': doc.source,
                'timestamp': doc.timestamp
            })
        
        with open(documents_file, 'w') as f:
            json.dump(serialized_docs, f, indent=2, default=str)
        
        self.logger.info(f"📄 Contextual documents saved to: {documents_file}")
        self.logger.info(f"📊 Total contextual documents: {len(self.contextual_documents)}")
    
    def _serialize_result(self, result: ParsedData) -> Dict[str, Any]:
        """Serialize ParsedData object for JSON output"""
        return {
            'file_path': result.file_path,
            'file_type': result.file_type,
            'parser_name': result.parser_name,
            'metadata': result.metadata,
            'data': result.data,
            'error': result.error,
            'processing_time': result.processing_time
        }
    
    def get_parser_info(self) -> Dict[str, str]:
        """Get information about available parsers"""
        return self.parser_factory.get_parser_info()
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return self.parser_factory.get_supported_extensions()

def main():
    """Main function to run the modular parser"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Modular Oil & Gas Data Parser")
    parser.add_argument("data_path", help="Path to data directory")
    parser.add_argument("--output", "-o", default="parsed_data", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run parser
    data_parser = ModularDataParser(args.data_path, args.output)
    
    print("🔧 Modular Data Parser")
    print("=" * 50)
    print(f"📁 Data path: {args.data_path}")
    print(f"📊 Output: {args.output}")
    print()
    
    # Show available parsers
    parser_info = data_parser.get_parser_info()
    print("🔧 Available parsers:")
    for ext, parser_name in parser_info.items():
        print(f"   {ext:8} → {parser_name}")
    print()
    
    # Run parsing
    summary = data_parser.parse_all_data()
    
    # Display results
    print("📊 Parsing Results:")
    print("=" * 50)
    print(f"✅ Successful: {summary['successful_parses']}")
    print(f"❌ Failed: {summary['failed_parses']}")
    print(f"📈 Success rate: {summary['success_rate']:.1%}")
    print(f"⏱️  Total time: {summary['total_processing_time']:.2f}s")
    print(f"📁 Files processed: {summary['total_files_processed']}")
    print()
    
    print("📋 File type breakdown:")
    for file_type, count in summary['file_type_breakdown'].items():
        print(f"   {file_type:8} → {count}")
    print()
    
    if summary['failed_parses'] > 0:
        print("❌ Error summary:")
        for error_type, count in summary['error_summary'].items():
            print(f"   {error_type:15} → {count}")

if __name__ == "__main__":
    main() 