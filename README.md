# Oil & Gas Data Processing Pipeline

A comprehensive **modular data processing and vectorization pipeline** for oil & gas exploration data, designed to support generative AI agentic workflows with DataStax Astra DB.

## 🎯 Overview

This project provides a complete solution for parsing, contextualizing, and vectorizing oil & gas exploration data from the Volve field. It transforms raw data files into structured, searchable documents that can be used by AI agents for intelligent Q&A and analysis.

**🆕 NEW: Astra DB Integration** - Single collection approach with built-in vectorize service
**🆕 NEW: Unstructured SDK Integration** - Intelligent document parsing and chunking
**🆕 NEW: Modular Parser Architecture** - Based on real data analysis with robust, maintainable design

## 📊 Data Types Supported

- **SEG-Y Files**: 3D seismic reflection data (870MB each)
- **DLIS Files**: Digital Log Interchange Standard (well logs)
- **LAS Files**: Log ASCII Standard (well logs with permeability data) - *Handled by Unstructured SDK*
- **ASCII Files**: Petrophysical info with header metadata - *Handled by Unstructured SDK*
- **PDF Files**: Reports and documentation - *Handled by Unstructured SDK*
- **Excel Files**: Facies and tabular data - *Handled by Unstructured SDK*
- **Text Files**: Content descriptions - *Handled by Unstructured SDK*
- **DAT Files**: Formation picks - *Handled by Unstructured SDK*
- **Images**: PNG, JPG, TIFF, BMP - *Handled by Unstructured SDK*

## 🏗️ Architecture

```
Raw Data Files → Modular Parser → Contextual Documents → Astra DB → Query Engine
     ↓              ↓                    ↓                ↓              ↓
  SEG-Y, DLIS   ParsedData          Rich Context    Single Collection  AI Agent
  LAS, ASCII    Objects            Documents       with Vectorize      Interface
  PDF, Excel    (Direct Context)   (Direct Output) Service
```

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
# Astra DB Configuration
ASTRA_DB_API_ENDPOINT=your_api_endpoint_here
ASTRA_DB_TOKEN=your_application_token_here
ASTRA_DB_KEYSPACE=your_keyspace_name
ASTRA_DB_COLLECTION=your_collection_name

```

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd drill-assistant

# Set up Python virtual environment (recommended: use uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install system dependencies for Unstructured SDK
brew install poppler tesseract  # macOS
# sudo apt-get install poppler-utils tesseract-ocr  # Ubuntu/Debian
```

### 3. Data Setup

Ensure your oil & gas data is in the `data/` directory. The system automatically discovers and processes all supported file types.

### 4. Run the Pipeline

```bash
# Analyze file structure first
python file_structure_analyzer.py data

# Run oil & gas parser
python oil_gas_parser.py data

# Run with custom output directory
python oil_gas_parser.py data --output my_parsed_data

# Run with verbose logging
python oil_gas_parser.py data --verbose
```

### 5. Load Data to Astra DB

```bash
# Load contextual documents to Astra DB
python astra_vector_store.py --load-documents

# Test queries
python astra_vector_store.py --query "What is the maximum inclination of well F-12?"
python astra_vector_store.py --query "Tell me about the Hugin Formation" --well "15_9-F-11"
```

## 📁 Generated Files

After running the oil_gas parser, you'll get:

- `parsed_data/parsing_results_YYYYMMDD_HHMMSS.json`: Detailed parsing results
- `parsed_data/parsing_summary_YYYYMMDD_HHMMSS.json`: Summary statistics
- `parsed_data/contextual_documents_YYYYMMDD_HHMMSS.json`: Contextual documents for vector storage
- `file_analysis_report.json`: Comprehensive file structure analysis
- `file_analysis_summary.json`: Summary of file types and sizes

## 🔧 Core Components

### 1. Oil & Gas Parser (`oil_gas_parser.py`)

**NEW**: Modular parser system with factory pattern and direct contextual document generation:

```python
from oil_gas_parser import ModularDataParser

parser = ModularDataParser("data", "output_dir")
summary = parser.parse_all_data()
```

**Features:**
- **Extension-based parsing** (no more fragile regex)
- **Real data structures** (based on actual file analysis)
- **Modular design** (each parser is independent)
- **Direct contextual document generation** (no separate context processor)
- **Unstructured SDK integration** for general document types
- **Comprehensive error handling**
- **Performance metrics** and statistics

### 2. Parser Factory (`parsers/parser_factory.py`)

Automatically selects appropriate parsers for each file type:

```python
from parsers import ParserFactory

factory = ParserFactory()
parser = factory.create_parser("path/to/file.asc")
result = parser.parse()
```

**Available Parsers:**
- `UnstructuredParser`: Universal document parser (PDF, Excel, Word, Images, LAS, ASCII, DAT, etc.)
- `DlisParser`: Digital log data (custom parser)
- `SegyParser`: Seismic data (custom parser)

**Parser Priority:**
1. **Priority Parsers**: Custom parsers for industry-specific formats (DLIS, SEG-Y)
2. **Extension Mapping**: File extensions mapped to UnstructuredParser
3. **Fallback**: UnstructuredParser for unknown extensions

### 3. Astra Vector Store (`astra_vector_store.py`)

**NEW**: Astra DB-based vector storage with single collection approach:

```python
from astra_vector_store import AstraVectorStoreSingle, AstraQueryEngineSingle

vector_store = AstraVectorStoreSingle()
query_engine = AstraQueryEngineSingle(vector_store)

# Query the data
result = query_engine.query("What is the maximum inclination of well F-12?")
```

**Features:**
- **Single collection** with rich metadata and tags
- **Built-in vectorize service** (NVIDIA NV-Embed-QA model)
- **Automatic collection creation** if it doesn't exist
- **Semantic search capabilities** using `$vectorize` operator
- **Well-specific filtering**
- **Cross-document type queries**
- **Configurable collection name** via environment variables

### 4. File Structure Analyzer (`file_structure_analyzer.py`)

Analyzes the actual file structure in your data directory:

```bash
python file_structure_analyzer.py data
```

**Output:**
- Comprehensive file type analysis
- Size and count statistics
- Parser recommendations
- Data quality assessment

### 5. Cleanup Utility (`cleanup_old_collections.py`)

Utility to delete any Astra DB collection:

```bash
python cleanup_old_collections.py --collection my_collection_name [--keyspace my_keyspace]
```

## 🔍 Query Examples

### Well Information
```python
# Get well details
result = query_engine.query("Tell me about well F-12")

# Well trajectory analysis
result = query_engine.query("What is the maximum inclination of well F-12?")

# Survey data
result = query_engine.query("Show me survey data for all wells")
```

### Geological Analysis
```python
# Formation information
result = query_engine.query("What formations are present in the field?")

# Formation depths
result = query_engine.query("What is the average depth of the Hugin Formation?")

# Geological context
result = query_engine.query("Tell me about the Draupne Formation")
```

### Petrophysical Data
```python
# Available curves
result = query_engine.query("What petrophysical curves are available?")

# Porosity analysis
result = query_engine.query("Show me porosity data for well F-15")

# Permeability information
result = query_engine.query("What is the permeability range in the field?")
```

### Seismic Information
```python
# Seismic data types
result = query_engine.query("What type of seismic data is available?")

# Migration information
result = query_engine.query("What migration algorithm was used?")

# Seismic interpretation
result = query_engine.query("How can I interpret the seismic data?")
```

### Cross-Document Queries
```python
# Query across multiple document types
result = query_engine.query("What is the relationship between seismic data and well logs?")

# Well-specific comprehensive analysis
result = query_engine.query_cross_reference("Analyze well F-11", "15_9-F-11")
```

## 📈 Data Quality Assessment

The pipeline provides comprehensive data quality metrics:

- **File Structure Analysis**: Complete breakdown of file types and sizes
- **Parser Success Rates**: Performance metrics for each parser type
- **Data Completeness**: Missing vs. available data
- **Format Validation**: Ensures data integrity
- **Collection Statistics**: Document counts and type distribution

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ASTRA_DB_API_ENDPOINT` | Astra DB API endpoint | Required |
| `ASTRA_DB_TOKEN` | Astra DB application token | Required |
| `ASTRA_DB_KEYSPACE` | Astra DB keyspace name | Optional |
| `ASTRA_DB_COLLECTION` | Collection name | `oil_gas_documents` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 🙏 Acknowledgments

- DataStax Astra DB for vector storage
- Unstructured SDK for intelligent document parsing
- Open source oil & gas libraries

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example usage

---

**Ready to build intelligent oil & gas AI agents with Astra DB! 🚀**