import deepl

def get_deepl_usage(auth_key):
    """
    Fetch the current DeepL character usage and quota.
    
    Args:
        auth_key (str): DeepL API key
    
    Returns:
        tuple: (characters_used, character_limit, characters_remaining)
    """
    try:
        # print(auth_key)
        # For Free API keys, specify the server_url parameter
        translator = deepl.Translator(
            auth_key,
            server_url="https://api-free.deepl.com"  # Required for Free tier
        )
        
        usage = translator.get_usage()
        
        if usage.character.valid:
            return (
                usage.character.count,
                usage.character.limit,
                usage.character.limit - usage.character.count
            )
        else:
            raise ValueError("Character usage tracking not available for this account type")
            
    except deepl.DeepLException as e:
        # print(f"DeepL API Error: {str(e)}")
        return None, None, None
    except Exception as e:
        # print(f"Unexpected error: {str(e)}")
        return None, None, None


# used, limit, remaining = get_deepl_usage("66c2b5ff-766e-424a-adbd-4ad803cb5ea5:fx")

# if used is not None:
#     print(f"Characters used: {used}")
#     print(f"Monthly limit: {limit}")
#     print(f"Remaining: {remaining}")
# else:
#     print("Failed to retrieve usage data")