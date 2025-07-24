#!/usr/bin/env python3

import re

def test_sanctuary_detection():
    """Test the new sanctuary detection logic"""
    
    # Test cases
    test_questions = [
        "Where are all the tiger reserve forests in india",
        "List all tiger reserves in India", 
        "Tell me about all national parks",
        "What are all the wildlife sanctuaries in India",
        "How many tiger reserves are there in India",
        "Ranthambore tiger reserve",
        "Tell me about Corbett national park",
        "Bandhavgarh wildlife sanctuary information",
        "Kaziranga national park details"
    ]
    
    # General query patterns
    general_query_patterns = [
        r'\b(?:where are all|list all|all the|how many|what are all)\b.*?(?:tiger reserve|national park|wildlife sanctuary|sanctuary|reserve)',
        r'\b(?:tell me about all|show me all|give me a list)\b.*?(?:tiger reserve|national park|wildlife sanctuary|sanctuary|reserve)',
        r'\b(?:tiger reserves? in india|national parks in india|sanctuaries in india)\b'
    ]
    
    # Specific sanctuary pattern (improved)
    sanctuary_pattern = r'\b(?!(?:the|all|any|where|what|which|how|many)\b)([A-Za-z][A-Za-z\s]{2,}?)\s+(?:wls|wildlife sanctuary|national park|tiger reserve|biosphere reserve|sanctuary|park|reserve)\b'
    
    print("Testing sanctuary detection logic:")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\nQuestion: '{question}'")
        
        # Check for general query
        is_general_query = any(re.search(pattern, question.lower()) for pattern in general_query_patterns)
        
        if is_general_query:
            print("  ✅ Detected as: GENERAL QUERY")
            sanctuary_matches = []
        else:
            # Check for specific sanctuary
            sanctuary_matches = re.findall(sanctuary_pattern, question.lower())
            sanctuary_matches = [match.strip() for match in sanctuary_matches if match.strip().lower() not in ['the', 'all', 'any', 'where', 'what', 'which', 'how', 'many', 'are', 'is']]
            
            if sanctuary_matches:
                print(f"  🏞️  Detected as: SPECIFIC SANCTUARY - {sanctuary_matches}")
            else:
                print("  ❓ Detected as: UNCLEAR (would use general context)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_sanctuary_detection()
