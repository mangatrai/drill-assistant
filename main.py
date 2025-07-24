"""
Main Orchestration Script for Oil & Gas Data Processing Pipeline
Runs the complete workflow from data parsing to vector store setup
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Import our custom modules
from data_parser import OilGasDataParser
from context_processor import OilGasContextProcessor
from vector_store import OilGasVectorStore, OilGasQueryEngine

def main():
    """Main orchestration function"""
    print("=" * 60)
    print("OIL & GAS DATA PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if data directory exists
    data_path = "data"
    if not os.path.exists(data_path):
        print(f"❌ Data directory '{data_path}' not found!")
        print("Please ensure the data directory exists with your oil & gas files.")
        return False
    
    try:
        # Step 1: Parse all data files
        print("📁 STEP 1: Parsing Data Files")
        print("-" * 40)
        
        parser = OilGasDataParser(data_path)
        parsed_data = parser.parse_all_data()
        
        # Save parsed data
        with open("parsed_data.json", "w") as f:
            json.dump(parsed_data, f, indent=2, default=str)
        
        print(f"✅ Parsed data saved to 'parsed_data.json'")
        print(f"📊 Summary: {parsed_data['summary']}")
        print()
        
        # Step 2: Process into contextual documents
        print("📝 STEP 2: Creating Contextual Documents")
        print("-" * 40)
        
        processor = OilGasContextProcessor(parsed_data)
        documents = processor.process_all_data()
        
        # Save contextual documents
        with open("contextual_documents.json", "w") as f:
            json.dump([{
                'content': doc.content,
                'metadata': doc.metadata,
                'document_type': doc.document_type,
                'source': doc.source,
                'timestamp': doc.timestamp
            } for doc in documents], f, indent=2)
        
        print(f"✅ Generated {len(documents)} contextual documents")
        print(f"📄 Documents saved to 'contextual_documents.json'")
        print()
        
        # Step 3: Set up vector store
        print("🔍 STEP 3: Setting Up Vector Store")
        print("-" * 40)
        
        vector_store = OilGasVectorStore()
        vector_store.load_contextual_documents()
        
        # Get collection statistics
        stats = vector_store.get_collection_stats()
        print("📊 Vector Store Statistics:")
        for collection, count in stats.items():
            if collection != 'total_documents':
                print(f"   - {collection}: {count} documents")
        print(f"   - Total: {stats.get('total_documents', 'N/A')} documents")
        print()
        
        # Step 4: Test query engine
        print("🔎 STEP 4: Testing Query Engine")
        print("-" * 40)
        
        query_engine = OilGasQueryEngine(vector_store)
        
        # Test queries
        test_queries = [
            "What is the total number of wells in the field?",
            "What formations are present in the field?",
            "What petrophysical data is available?",
            "What type of seismic data do we have?",
            "Tell me about well F-12"
        ]
        
        print("🧪 Running test queries:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            result = query_engine.query(query, max_results=3)
            print(f"   Type: {result['query_type']}")
            print(f"   Results: {result['summary']}")
        
        print()
        
        # Step 5: Save pipeline status
        print("💾 STEP 5: Saving Pipeline Status")
        print("-" * 40)
        
        pipeline_status = {
            'timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'parsed_data_file': 'parsed_data.json',
            'contextual_documents_file': 'contextual_documents.json',
            'vector_store_path': './chroma_db',
            'statistics': stats,
            'test_queries': len(test_queries),
            'status': 'completed_successfully'
        }
        
        with open("pipeline_status.json", "w") as f:
            json.dump(pipeline_status, f, indent=2)
        
        print("✅ Pipeline status saved to 'pipeline_status.json'")
        print()
        
        # Final summary
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Generated Files:")
        print("   - parsed_data.json: Raw parsed data")
        print("   - contextual_documents.json: Processed documents")
        print("   - pipeline_status.json: Pipeline status")
        print("   - chroma_db/: Vector store database")
        print()
        print("🔧 Next Steps:")
        print("   1. Use the query engine for data exploration")
        print("   2. Integrate with your AI agent workflow")
        print("   3. Add more sophisticated analysis capabilities")
        print()
        print("💡 Example Usage:")
        print("   from vector_store import OilGasVectorStore, OilGasQueryEngine")
        print("   vector_store = OilGasVectorStore()")
        print("   query_engine = OilGasQueryEngine(vector_store)")
        print("   result = query_engine.query('Your question here')")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        print(f"📋 Error details: {type(e).__name__}: {str(e)}")
        return False

def interactive_mode():
    """Interactive mode for testing queries"""
    print("\n" + "=" * 60)
    print("INTERACTIVE QUERY MODE")
    print("=" * 60)
    print("Type 'quit' to exit, 'help' for available commands")
    print()
    
    try:
        vector_store = OilGasVectorStore()
        query_engine = OilGasQueryEngine(vector_store)
        
        while True:
            try:
                query = input("🔍 Enter your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\n📚 Available Commands:")
                    print("   - 'stats': Show collection statistics")
                    print("   - 'quit': Exit interactive mode")
                    print("   - Any question about the oil & gas data")
                    print()
                    continue
                
                if query.lower() == 'stats':
                    stats = vector_store.get_collection_stats()
                    print("\n📊 Collection Statistics:")
                    for collection, count in stats.items():
                        print(f"   - {collection}: {count}")
                    print()
                    continue
                
                if not query:
                    continue
                
                print(f"\n🔍 Searching for: {query}")
                result = query_engine.query(query, max_results=5)
                
                print(f"📋 Query Type: {result['query_type']}")
                print(f"📊 {result['summary']}")
                
                # Show top results
                if 'results' in result:
                    if isinstance(result['results'], dict):
                        # Multiple collections
                        for collection, results_list in result['results'].items():
                            if results_list:
                                print(f"\n📁 {collection.upper()} Results:")
                                for i, res in enumerate(results_list[:2], 1):
                                    print(f"   {i}. {res['content'][:200]}...")
                    else:
                        # Single collection
                        print(f"\n📁 Results:")
                        for i, res in enumerate(result['results'][:2], 1):
                            print(f"   {i}. {res['content'][:200]}...")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
                
    except Exception as e:
        print(f"❌ Failed to initialize query engine: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        success = main()
        if success:
            print("\n🚀 Ready for AI agent integration!")
        else:
            print("\n❌ Pipeline failed. Check the error messages above.")
            sys.exit(1) 