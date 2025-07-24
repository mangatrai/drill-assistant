"""
Vector Store Setup for Oil & Gas Data
Uses ChromaDB for efficient storage and retrieval of contextual documents
"""

import json
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import numpy as np
from datetime import datetime

class OilGasVectorStore:
    """Vector store for oil & gas exploration data"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collections
        self.well_collection = self.client.get_or_create_collection(
            name="well_data",
            metadata={"description": "Well summaries and trajectory data"}
        )
        
        self.formation_collection = self.client.get_or_create_collection(
            name="formation_data", 
            metadata={"description": "Geological formation analysis"}
        )
        
        self.petrophysical_collection = self.client.get_or_create_collection(
            name="petrophysical_data",
            metadata={"description": "Petrophysical interpretations and well logs"}
        )
        
        self.seismic_collection = self.client.get_or_create_collection(
            name="seismic_data",
            metadata={"description": "Seismic data and interpretation guidance"}
        )
        
        self.field_collection = self.client.get_or_create_collection(
            name="field_data",
            metadata={"description": "Field overview and general information"}
        )
        
        print("Vector store initialized with collections:")
        print("- well_data: Well summaries and trajectories")
        print("- formation_data: Geological formations")
        print("- petrophysical_data: Petrophysical data")
        print("- seismic_data: Seismic information")
        print("- field_data: Field overview")
    
    def load_contextual_documents(self, documents_file: str = "contextual_documents.json"):
        """Load and index contextual documents"""
        print(f"Loading documents from {documents_file}...")
        
        if not os.path.exists(documents_file):
            print(f"Document file {documents_file} not found. Please run context_processor.py first.")
            return
        
        with open(documents_file, 'r') as f:
            documents = json.load(f)
        
        print(f"Found {len(documents)} documents to index")
        
        # Separate documents by type
        well_docs = []
        formation_docs = []
        petrophysical_docs = []
        seismic_docs = []
        field_docs = []
        
        for doc in documents:
            doc_type = doc['document_type']
            
            if doc_type == 'well_summary' or doc_type == 'trajectory_analysis':
                well_docs.append(doc)
            elif doc_type == 'formation_analysis':
                formation_docs.append(doc)
            elif doc_type == 'petrophysical_analysis':
                petrophysical_docs.append(doc)
            elif doc_type == 'seismic_analysis':
                seismic_docs.append(doc)
            elif doc_type == 'field_overview':
                field_docs.append(doc)
        
        # Index documents in appropriate collections
        self._index_documents(self.well_collection, well_docs, "well")
        self._index_documents(self.formation_collection, formation_docs, "formation")
        self._index_documents(self.petrophysical_collection, petrophysical_docs, "petrophysical")
        self._index_documents(self.seismic_collection, seismic_docs, "seismic")
        self._index_documents(self.field_collection, field_docs, "field")
        
        print("Document indexing completed!")
        print(f"Indexed {len(well_docs)} well documents")
        print(f"Indexed {len(formation_docs)} formation documents")
        print(f"Indexed {len(petrophysical_docs)} petrophysical documents")
        print(f"Indexed {len(seismic_docs)} seismic documents")
        print(f"Indexed {len(field_docs)} field documents")
    
    def _index_documents(self, collection, documents: List[Dict], doc_type: str):
        """Index documents in a specific collection"""
        if not documents:
            return
        
        # Prepare data for indexing
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = f"{doc_type}_{i}_{doc['source']}"
            ids.append(doc_id)
            texts.append(doc['content'])
            
            # Prepare metadata
            metadata = {
                'document_type': doc['document_type'],
                'source': doc['source'],
                'timestamp': doc['timestamp']
            }
            
            # Add specific metadata based on document type
            if 'well_name' in doc['metadata']:
                metadata['well_name'] = doc['metadata']['well_name']
            if 'formation_name' in doc['metadata']:
                metadata['formation_name'] = doc['metadata']['formation_name']
            if 'filename' in doc['metadata']:
                metadata['filename'] = doc['metadata']['filename']
            
            metadatas.append(metadata)
        
        # Add documents to collection
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_wells(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for well-related information"""
        results = self.well_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_search_results(results, "well")
    
    def search_formations(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for formation-related information"""
        results = self.formation_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_search_results(results, "formation")
    
    def search_petrophysical(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for petrophysical information"""
        results = self.petrophysical_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_search_results(results, "petrophysical")
    
    def search_seismic(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for seismic information"""
        results = self.seismic_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_search_results(results, "seismic")
    
    def search_field(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for field overview information"""
        results = self.field_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_search_results(results, "field")
    
    def search_all(self, query: str, n_results: int = 10) -> Dict[str, List[Dict]]:
        """Search across all collections"""
        return {
            'wells': self.search_wells(query, n_results // 2),
            'formations': self.search_formations(query, n_results // 4),
            'petrophysical': self.search_petrophysical(query, n_results // 4),
            'seismic': self.search_seismic(query, n_results // 4),
            'field': self.search_field(query, n_results // 4)
        }
    
    def _format_search_results(self, results: Dict, collection_type: str) -> List[Dict]:
        """Format search results for consistent output"""
        formatted_results = []
        
        if 'documents' in results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_result = {
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'collection': collection_type
                }
                formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_well_by_name(self, well_name: str) -> Optional[Dict]:
        """Get specific well information by name"""
        results = self.well_collection.query(
            query_texts=[well_name],
            where={"well_name": well_name},
            n_results=1
        )
        
        if results['documents'] and results['documents'][0]:
            return {
                'content': results['documents'][0][0],
                'metadata': results['metadatas'][0][0] if results['metadatas'] else {},
                'collection': 'well'
            }
        
        return None
    
    def get_formation_by_name(self, formation_name: str) -> Optional[Dict]:
        """Get specific formation information by name"""
        results = self.formation_collection.query(
            query_texts=[formation_name],
            where={"formation_name": formation_name},
            n_results=1
        )
        
        if results['documents'] and results['documents'][0]:
            return {
                'content': results['documents'][0][0],
                'metadata': results['metadatas'][0][0] if results['metadatas'] else {},
                'collection': 'formation'
            }
        
        return None
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about indexed documents"""
        stats = {}
        
        try:
            stats['well_documents'] = self.well_collection.count()
            stats['formation_documents'] = self.formation_collection.count()
            stats['petrophysical_documents'] = self.petrophysical_collection.count()
            stats['seismic_documents'] = self.seismic_collection.count()
            stats['field_documents'] = self.field_collection.count()
            stats['total_documents'] = sum(stats.values())
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            stats = {'error': str(e)}
        
        return stats
    
    def reset_collections(self):
        """Reset all collections (use with caution)"""
        print("Resetting all collections...")
        
        self.client.delete_collection("well_data")
        self.client.delete_collection("formation_data")
        self.client.delete_collection("petrophysical_data")
        self.client.delete_collection("seismic_data")
        self.client.delete_collection("field_data")
        
        # Recreate collections
        self.__init__(self.persist_directory)
        
        print("Collections reset and recreated")

class OilGasQueryEngine:
    """Query engine for oil & gas data with intelligent routing"""
    
    def __init__(self, vector_store: OilGasVectorStore):
        self.vector_store = vector_store
        
    def query(self, question: str, max_results: int = 10) -> Dict[str, Any]:
        """Intelligent query routing based on question type"""
        
        # Determine query type and route accordingly
        query_type = self._classify_query(question)
        
        if query_type == 'well_specific':
            return self._handle_well_query(question, max_results)
        elif query_type == 'formation_specific':
            return self._handle_formation_query(question, max_results)
        elif query_type == 'petrophysical':
            return self._handle_petrophysical_query(question, max_results)
        elif query_type == 'seismic':
            return self._handle_seismic_query(question, max_results)
        elif query_type == 'field_overview':
            return self._handle_field_query(question, max_results)
        else:
            # General query - search all collections
            return self._handle_general_query(question, max_results)
    
    def _classify_query(self, question: str) -> str:
        """Classify the type of query"""
        question_lower = question.lower()
        
        # Well-specific keywords
        well_keywords = ['well', 'borehole', 'trajectory', 'survey', 'drilling']
        if any(keyword in question_lower for keyword in well_keywords):
            return 'well_specific'
        
        # Formation-specific keywords
        formation_keywords = ['formation', 'layer', 'strata', 'geological', 'hugin', 'draupne', 'heather']
        if any(keyword in question_lower for keyword in formation_keywords):
            return 'formation_specific'
        
        # Petrophysical keywords
        petro_keywords = ['porosity', 'permeability', 'saturation', 'resistivity', 'gamma ray', 'density', 'sonic']
        if any(keyword in question_lower for keyword in petro_keywords):
            return 'petrophysical'
        
        # Seismic keywords
        seismic_keywords = ['seismic', 'migration', 'reflection', 'attribute', 'interpretation']
        if any(keyword in question_lower for keyword in seismic_keywords):
            return 'seismic'
        
        # Field overview keywords
        field_keywords = ['field', 'overview', 'summary', 'statistics', 'volve']
        if any(keyword in question_lower for keyword in field_keywords):
            return 'field_overview'
        
        return 'general'
    
    def _handle_well_query(self, question: str, max_results: int) -> Dict[str, Any]:
        """Handle well-specific queries"""
        results = self.vector_store.search_wells(question, max_results)
        
        return {
            'query_type': 'well_specific',
            'question': question,
            'results': results,
            'summary': f"Found {len(results)} well-related documents"
        }
    
    def _handle_formation_query(self, question: str, max_results: int) -> Dict[str, Any]:
        """Handle formation-specific queries"""
        results = self.vector_store.search_formations(question, max_results)
        
        return {
            'query_type': 'formation_specific',
            'question': question,
            'results': results,
            'summary': f"Found {len(results)} formation-related documents"
        }
    
    def _handle_petrophysical_query(self, question: str, max_results: int) -> Dict[str, Any]:
        """Handle petrophysical queries"""
        results = self.vector_store.search_petrophysical(question, max_results)
        
        return {
            'query_type': 'petrophysical',
            'question': question,
            'results': results,
            'summary': f"Found {len(results)} petrophysical documents"
        }
    
    def _handle_seismic_query(self, question: str, max_results: int) -> Dict[str, Any]:
        """Handle seismic queries"""
        results = self.vector_store.search_seismic(question, max_results)
        
        return {
            'query_type': 'seismic',
            'question': question,
            'results': results,
            'summary': f"Found {len(results)} seismic documents"
        }
    
    def _handle_field_query(self, question: str, max_results: int) -> Dict[str, Any]:
        """Handle field overview queries"""
        results = self.vector_store.search_field(question, max_results)
        
        return {
            'query_type': 'field_overview',
            'question': question,
            'results': results,
            'summary': f"Found {len(results)} field overview documents"
        }
    
    def _handle_general_query(self, question: str, max_results: int) -> Dict[str, Any]:
        """Handle general queries across all collections"""
        results = self.vector_store.search_all(question, max_results)
        
        total_results = sum(len(r) for r in results.values())
        
        return {
            'query_type': 'general',
            'question': question,
            'results': results,
            'summary': f"Found {total_results} documents across all collections"
        }

if __name__ == "__main__":
    # Initialize vector store
    vector_store = OilGasVectorStore()
    
    # Load documents if they exist
    if os.path.exists("contextual_documents.json"):
        vector_store.load_contextual_documents()
        
        # Print collection statistics
        stats = vector_store.get_collection_stats()
        print("\nCollection Statistics:")
        for collection, count in stats.items():
            print(f"- {collection}: {count} documents")
        
        # Initialize query engine
        query_engine = OilGasQueryEngine(vector_store)
        
        # Example queries
        print("\nExample Queries:")
        
        # Test well query
        well_result = query_engine.query("What is the maximum inclination of well F-12?")
        print(f"\nWell Query: {well_result['summary']}")
        
        # Test formation query
        formation_result = query_engine.query("What is the average depth of the Hugin Formation?")
        print(f"\nFormation Query: {formation_result['summary']}")
        
        # Test petrophysical query
        petro_result = query_engine.query("What petrophysical curves are available?")
        print(f"\nPetrophysical Query: {petro_result['summary']}")
        
        # Test seismic query
        seismic_result = query_engine.query("What type of seismic data is available?")
        print(f"\nSeismic Query: {seismic_result['summary']}")
        
        # Test field query
        field_result = query_engine.query("What is the total number of wells in the field?")
        print(f"\nField Query: {field_result['summary']}")
        
    else:
        print("No contextual documents found. Please run context_processor.py first.") 