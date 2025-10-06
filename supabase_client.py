from supabase import create_client

# Your Supabase URL and API key
SUPABASE_URL = "https://qvkwfvcievnokpiwzruo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF2a3dmdmNpZXZub2twaXd6cnVvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTU1Njk3MCwiZXhwIjoyMDc1MTMyOTcwfQ.qGPizxCYrt-X8pW_aJN1Yp_c0BGLo6S6ZaRzDMaCu5w"

def fetch_supabase_db(client_id):
  # Initialize client
  supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

  primary_key_value = client_id

  response = supabase.table('clients').select('*').eq('id', primary_key_value).execute()
  if response.data:
    client_info = response.data[0]
    return client_info
  else:
    return -1;


def fetch_supabase_cat_db():
  # Initialize client
  supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

  response = supabase.table('categories').select('*').execute()
  if response.data:
    client_info = response.data
    return client_info
  else:
    return -1;
