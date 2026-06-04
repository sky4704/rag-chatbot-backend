import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "rag-files")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_file(file_path: str, destination_path: str):
    """Uploads a file to Supabase Storage if configured, else returns filename."""
    if supabase:
        with open(file_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(destination_path, f.read())
        # Return the public URL
        res = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(destination_path)
        return res
    # Return just the filename if stored locally
    return os.path.basename(destination_path)

def delete_file(filename: str, base_dir: str = ""):
    """Deletes a file from Supabase Storage or local disk."""
    if supabase:
        try:
            supabase.storage.from_(SUPABASE_BUCKET).remove([filename])
        except Exception as e:
            print(f"Error deleting from Supabase: {e}")
    else:
        # For local, we need the full path
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            os.remove(full_path)
