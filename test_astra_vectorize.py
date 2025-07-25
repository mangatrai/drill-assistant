#!/usr/bin/env python3
"""
Test script for Astra DB Vectorize Service
"""

import os
from astrapy import DataAPIClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_astra_vectorize():
    """Test Astra DB vectorize service"""
    
    # Get credentials
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    token = os.getenv('ASTRA_DB_TOKEN')
    keyspace = os.getenv('ASTRA_DB_KEYSPACE')
    
    if not api_endpoint or not token:
        print("❌ Missing Astra DB credentials in .env file")
        return
    
    print("🔧 Testing Astra DB Vectorize Service...")
    print(f"   API Endpoint: {api_endpoint}")
    
    try:
        # Initialize client
        client = DataAPIClient()
        database = client.get_database(api_endpoint, token=token, keyspace=keyspace )
        
        # Test collection name
        test_collection_name = "test_vectorize"
        
        print(f"   Testing collection: {test_collection_name}")
        
        # Try to get the collection (it should exist if you created it manually)
        try:
            collection = database.get_collection(test_collection_name)
            print(f"   ✅ Collection '{test_collection_name}' found")
            
            # Test inserting a document with vectorize
            test_doc = {
                "name": "Test Document",
                "description": "This is a test document for vectorize service",
                "$vectorize": "This is a test document for vectorize service"
            }
            
            print(f"   📝 Inserting test document...")
            result = collection.insert_one(test_doc)
            print(f"   ✅ Document inserted with ID: {result.inserted_id}")
            
            # Test vector search
            print(f"   🔍 Testing vector search...")
            search_results = collection.find(
                {},  # Empty filter
                sort={"$vectorize": "test document"},  # Sort by vector similarity
                limit=3,
                include_similarity=True  # Include similarity scores
            )
            results_list = list(search_results)
            print(f"   ✅ Vector search found {len(results_list)} results")
            
            if results_list:
                print(f"   📊 First result similarity: {results_list[0].get('$similarity', 'N/A')}")
            
            # Clean up - delete the test document
            print(f"   🧹 Cleaning up test document...")
            collection.delete_one({"_id": result.inserted_id})
            print(f"   ✅ Test document deleted")
            
        except Exception as e:
            print(f"   ❌ Error with collection '{test_collection_name}': {e}")
            print(f"   💡 Make sure you created the collection manually with vectorize service enabled")
        
    except Exception as e:
        print(f"❌ Error connecting to Astra DB: {e}")

if __name__ == "__main__":
    test_astra_vectorize() 