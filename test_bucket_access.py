from google.cloud import storage
import uuid

def test_gcs_access(bucket_name):
    # Create a unique object name
    blob_name = f"test_object_{uuid.uuid4()}.txt"
    
    # Instantiate the client
    client = storage.Client()
    
    try:
        # Get the bucket
        bucket = client.get_bucket(bucket_name)
        
        # Test writing
        blob = bucket.blob(blob_name)
        blob.upload_from_string("This is a test object")
        print(f"Successfully wrote object {blob_name} to bucket {bucket_name}")
        
        # Test reading
        content = blob.download_as_text()
        print(f"Successfully read object {blob_name} from bucket {bucket_name}")
        print(f"Content: {content}")
        
        # Test deleting
        blob.delete()
        print(f"Successfully deleted object {blob_name} from bucket {bucket_name}")
        
        print("All operations successful. You have full read/write access to the bucket.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("You may not have the necessary permissions for this bucket.")

if __name__ == "__main__":
    bucket_name = "contrailcast-trial-run-1-hierarchical"  # Replace with your actual bucket name
    test_gcs_access(bucket_name)