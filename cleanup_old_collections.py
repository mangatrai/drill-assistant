#!/usr/bin/env python3
"""
Cleanup script to remove old multi-collection approach
Since we've moved to single collection 'oil_gas_documents'
"""

import os
from astrapy import DataAPIClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def cleanup_old_collections():
    """Remove old collections from multi-collection approach"""
    
    # Get credentials
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    token = os.getenv('ASTRA_DB_TOKEN')
    keyspace = os.getenv('ASTRA_DB_KEYSPACE')
    
    if not api_endpoint or not token:
        print("❌ Missing Astra DB credentials in .env file")
        return
    
    print("🧹 Cleaning up old multi-collection approach...")
    print(f"   API Endpoint: {api_endpoint}")
    
    # Old collections to remove
    old_collections = [
        'well_summaries',
        'formation_analysis', 
        'petrophysical_insights',
        'seismic_metadata',
        'well_trajectories',
        'field_overview',
        'file_metadata'
    ]
    
    try:
        # Initialize client
        client = DataAPIClient()
        
        # Connect to database
        if keyspace:
            print(f"   Keyspace: {keyspace}")
            database = client.get_database(api_endpoint, token=token, keyspace=keyspace)
        else:
            print(f"   Keyspace: (using default)")
            database = client.get_database(api_endpoint, token=token)
        
        # Check which collections exist and remove them
        collections_to_remove = []
        
        for collection_name in old_collections:
            try:
                collection = database.get_collection(collection_name)
                print(f"   📋 Found old collection: {collection_name}")
                collections_to_remove.append(collection_name)
            except Exception as e:
                print(f"   ⚠️ Collection {collection_name} not found (already removed or doesn't exist)")
        
        if not collections_to_remove:
            print("   ✅ No old collections found to remove")
            return
        
        # Confirm deletion
        print(f"\n🗑️ Found {len(collections_to_remove)} old collections to remove:")
        for coll in collections_to_remove:
            print(f"   - {coll}")
        
        response = input("\n❓ Do you want to delete these collections? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            print("\n🗑️ Deleting old collections...")
            
            for collection_name in collections_to_remove:
                try:
                    collection = database.get_collection(collection_name)
                    
                    # Get document count before deletion
                    count = collection.count_documents({}, upper_bound=1000)
                    print(f"   📊 Collection '{collection_name}' has {count} documents")
                    
                    # Drop the collection
                    database.drop_collection(collection_name)
                    print(f"   ✅ Dropped collection: {collection_name}")
                    
                except Exception as e:
                    print(f"   ❌ Error deleting collection '{collection_name}': {e}")
            
            print("\n🎉 Cleanup completed!")
            print("   ✅ Old multi-collection approach removed")
            print("   ✅ Single collection 'oil_gas_documents' is now the only collection")
            print("   ✅ All old collections dropped successfully")
            
        else:
            print("   ❌ Cleanup cancelled")
    
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def verify_single_collection():
    """Verify that only the single collection exists"""
    
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    token = os.getenv('ASTRA_DB_TOKEN')
    keyspace = os.getenv('ASTRA_DB_KEYSPACE')
    
    try:
        client = DataAPIClient()
        
        if keyspace:
            database = client.get_database(api_endpoint, token=token, keyspace=keyspace)
        else:
            database = client.get_database(api_endpoint, token=token)
        
        # Check for the single collection
        try:
            collection = database.get_collection("oil_gas_documents")
            count = collection.count_documents({}, upper_bound=1000)
            print(f"✅ Single collection 'oil_gas_documents' exists with {count} documents")
        except Exception as e:
            print(f"❌ Single collection 'oil_gas_documents' not found: {e}")
        
        # Check for any remaining old collections
        old_collections = [
            'well_summaries', 'formation_analysis', 'petrophysical_insights',
            'seismic_metadata', 'well_trajectories', 'field_overview', 'file_metadata'
        ]
        
        remaining_old = []
        for coll_name in old_collections:
            try:
                database.get_collection(coll_name)
                remaining_old.append(coll_name)
            except:
                pass
        
        if remaining_old:
            print(f"⚠️ Found {len(remaining_old)} old collections still exist:")
            for coll in remaining_old:
                print(f"   - {coll}")
        else:
            print("✅ No old collections remain")
            
    except Exception as e:
        print(f"❌ Error verifying collections: {e}")

def main():
    """Main cleanup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cleanup old multi-collection approach")
    parser.add_argument("--verify", action="store_true", help="Verify current collection state")
    parser.add_argument("--cleanup", action="store_true", help="Remove old collections")
    
    args = parser.parse_args()
    
    if args.verify:
        print("🔍 Verifying collection state...")
        verify_single_collection()
    elif args.cleanup:
        cleanup_old_collections()
    else:
        print("Usage:")
        print("  python cleanup_old_collections.py --verify    # Check current state")
        print("  python cleanup_old_collections.py --cleanup   # Remove old collections")

if __name__ == "__main__":
    main() 