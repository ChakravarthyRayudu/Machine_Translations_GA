import deepl

def get_deepl_usage(auth_key):
    """
    Fetch the current DeepL character usage and quota.
    
    Args:
        auth_key (str): DeepL API key
    
    Returns:
        tuple: (characters_used, character_limit, characters_remaining, estimated_cost_usd)
    """
    try:
        translator = deepl.Translator(auth_key)
        usage = translator.get_usage()
        
        if usage.character.valid:
            used = usage.character.count
            limit = usage.character.limit
            remaining = limit - used
            estimated_cost = used / 40000  # $1 per 40,000 characters

            return used, limit, remaining, estimated_cost
        else:
            raise ValueError("Character usage tracking not available for this account type")
    except deepl.DeepLException as e:
        return None, None, None, None
    except Exception as e:
        return None, None, None, None


# used, limit, remaining = get_deepl_usage("66c2b5ff-766e-424a-adbd-4ad803cb5ea5:fx")

# if used is not None:
#     print(f"Characters used: {used}")
#     print(f"Monthly limit: {limit}")
#     print(f"Remaining: {remaining}")
# else:
#     print("Failed to retrieve usage data")