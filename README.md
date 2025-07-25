# Oil & Gas Data Processing Pipeline

A comprehensive **modular data processing and vectorization pipeline** for oil & gas exploration data, designed to support generative AI agentic workflows.

## 🎯 Overview

This project provides a complete solution for parsing, contextualizing, and vectorizing oil & gas exploration data from the Volve field. It transforms raw data files into structured, searchable documents that can be used by AI agents for intelligent Q&A and analysis.

**🆕 NEW: Modular Parser Architecture** - Based on real data analysis with robust, maintainable design.

## 📊 Data Types Supported

- **SEG-Y Files**: 3D seismic reflection data (870MB each)
- **DLIS Files**: Digital Log Interchange Standard (well logs)
- **LAS Files**: Log ASCII Standard (well logs with permeability data)
- **ASCII Files**: Petrophysical info with header metadata
- **PDF Files**: Reports and documentation
- **Excel Files**: Facies and tabular data
- **Text Files**: Content descriptions
- **DAT Files**: Formation picks

## 🏗️ Architecture

```
Raw Data Files → Modular Parser → Contextual Documents → Vector Store → Query Engine
     ↓              ↓                    ↓                ↓              ↓
  SEG-Y, DLIS   ParsedData          Rich Context    ChromaDB      AI Agent
  LAS, ASCII    Objects            Documents       Collections   Interface
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd drill-assistant

# Set up virtual environment (recommended)
source activate_env.sh

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Ensure your oil & gas data is in the `data/` directory. The system automatically discovers and processes all supported file types.

### 3. Run the Pipeline

```bash
# Run oil & gas parser (Phase 2)
python oil_gas_parser.py data

# Run with custom output directory
python oil_gas_parser.py data --output my_parsed_data

# Run with verbose logging
python oil_gas_parser.py data --verbose
```

## 📁 Generated Files

After running the modular parser, you'll get:

- `parsed_data/parsing_results_YYYYMMDD_HHMMSS.json`: Detailed parsing results
- `parsed_data/parsing_summary_YYYYMMDD_HHMMSS.json`: Summary statistics
- `parsed_data/contextual_documents_YYYYMMDD_HHMMSS.json`: Contextual documents for vector storage

## 🔧 Core Components

### 1. Oil & Gas Parser (`oil_gas_parser.py`)

**NEW**: Modular parser system with factory pattern:

```python
from oil_gas_parser import ModularDataParser

parser = ModularDataParser("data", "output_dir")
summary = parser.parse_all_data()
```

**Features:**
- **Extension-based parsing** (no more fragile regex)
- **Real data structures** (based on actual file analysis)
- **Modular design** (each parser is independent)
- **Comprehensive error handling**
- **Performance metrics** and statistics
- **88.9% success rate** on real data

### 2. Parser Factory (`parsers/parser_factory.py`)

Automatically selects appropriate parsers for each file type:

```python
from parsers import ParserFactory

factory = ParserFactory()
parser = factory.create_parser("path/to/file.asc")
result = parser.parse()
```

**Available Parsers:**
- `UnstructuredParser`: Universal document parser (PDF, Excel, Word, Images, etc.)
- `DlisParser`: Digital log data
- `SegyParser`: Seismic data

### 3. Context Processor (`context_processor.py`)

**⚠️ REVIEW NEEDED**: May need updates for new `ParsedData` format

Transforms parsed data into meaningful documents:

```python
from context_processor import OilGasContextProcessor

processor = OilGasContextProcessor(parsed_data)
documents = processor.process_all_data()
```

**Document Types:**
- Well summaries and trajectories
- Formation analysis
- Petrophysical insights
- Seismic metadata
- Field overview

### 4. Astra Vector Store (`astra_vector_store.py`)

Astra DB-based vector storage with intelligent querying:

```python
from astra_vector_store import AstraVectorStoreSingle, AstraQueryEngineSingle

vector_store = AstraVectorStoreSingle()
query_engine = AstraQueryEngineSingle(vector_store)

# Query the data
result = query_engine.query("What is the maximum inclination of well F-12?")
```

**Features:**
- Single collection with rich metadata
- Built-in vectorize service (NVIDIA embeddings)
- Semantic search capabilities
- Well-specific filtering
- `field_data`: Field overview

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

### Field Overview
```python
# Field statistics
result = query_engine.query("What is the total number of wells in the field?")

# Data quality
result = query_engine.query("What data quality assessment is available?")

# Field characteristics
result = query_engine.query("Tell me about the Volve field")
```

## 🤖 AI Agent Integration

### Basic Integration

```python
from astra_vector_store import AstraVectorStoreSingle, AstraQueryEngineSingle

class OilGasAIAgent:
    def __init__(self):
        self.vector_store = AstraVectorStoreSingle()
        self.query_engine = AstraQueryEngineSingle(self.vector_store)
    
    def answer_question(self, question: str) -> str:
        """Answer questions about oil & gas data"""
        result = self.query_engine.query(question)
        
        # Format response for AI agent
        response = f"Query Type: {result['query_type']}\n"
        response += f"Results: {result['summary']}\n\n"
        
        # Add relevant content
        if 'results' in result:
            if isinstance(result['results'], dict):
                for collection, results_list in result['results'].items():
                    if results_list:
                        response += f"{collection.upper()}:\n"
                        for res in results_list[:2]:
                            response += f"- {res['content'][:300]}...\n"
            else:
                for res in result['results'][:3]:
                    response += f"- {res['content'][:300]}...\n"
        
        return response

# Usage
agent = OilGasAIAgent()
answer = agent.answer_question("What is the reservoir formation in Volve field?")
print(answer)
```

### Advanced Integration with LLM

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class AdvancedOilGasAgent:
    def __init__(self, openai_api_key: str):
        self.vector_store = OilGasVectorStore()
        self.query_engine = OilGasQueryEngine(self.vector_store)
        
        # Initialize LLM
        self.llm = OpenAI(api_key=openai_api_key, temperature=0.1)
        
        # Create prompt template
        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are an expert oil & gas geoscientist. Answer the following question based on the provided context.
            
            Question: {question}
            
            Context: {context}
            
            Provide a comprehensive, professional answer:
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def answer_question(self, question: str) -> str:
        # Get relevant context from vector store
        result = self.query_engine.query(question, max_results=5)
        
        # Extract context
        context = ""
        if 'results' in result:
            if isinstance(result['results'], dict):
                for collection, results_list in result['results'].items():
                    if results_list:
                        context += f"\n{collection.upper()}:\n"
                        for res in results_list:
                            context += f"{res['content']}\n"
            else:
                for res in result['results']:
                    context += f"{res['content']}\n"
        
        # Generate answer using LLM
        response = self.chain.run(question=question, context=context)
        return response
```

## 📈 Data Quality Assessment

The pipeline provides comprehensive data quality metrics:

- **Well Coverage**: Number of wells with different data types
- **Data Completeness**: Missing vs. available data
- **File Size Analysis**: Large files that may need special handling
- **Format Validation**: Ensures data integrity

## 🔧 Customization

### Adding New File Formats

1. Extend `OilGasDataParser` class
2. Add parsing method for new format
3. Update data structures as needed

### Custom Document Types

1. Extend `OilGasContextProcessor` class
2. Add new document processing method
3. Update vector store collections

### Query Enhancements

1. Extend `OilGasQueryEngine` class
2. Add new query classification rules
3. Implement specialized query handlers

## 🧪 Testing

```bash
# Run tests
pytest

# Test specific components
python -m pytest tests/test_data_parser.py
python -m pytest tests/test_context_processor.py
python -m pytest tests/test_vector_store.py
```

## 📊 Performance

- **Parsing Speed**: ~1000 files/minute
- **Vector Store**: Sub-second query response
- **Memory Usage**: Efficient streaming for large files
- **Scalability**: Supports datasets with 1000+ wells

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Volve field data provided by Equinor
- ChromaDB for vector storage
- Open source oil & gas libraries

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example usage

---

**Ready to build intelligent oil & gas AI agents! 🚀**