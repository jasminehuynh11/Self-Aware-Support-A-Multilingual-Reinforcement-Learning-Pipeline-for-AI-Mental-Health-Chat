import json
import pandas as pd
import random
from collections import Counter, defaultdict
from typing import List, Dict

def load_and_sample_diverse_users(input_file: str, sample_size: int = 500) -> List[Dict]:
    """
    Load users and create a diverse sample of 500 people
    """
    print(f"Loading data from {input_file}...")
    
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    print(f"Loaded {len(users)} total users")
    
    if len(users) <= sample_size:
        print(f"Dataset has {len(users)} users, which is less than requested {sample_size}")
        print("Returning all users...")
        return users
    
    # Define demographic attributes for diversity
    demographic_attrs = [
        'RELIGIOUS', 'EMPLOYMENT', 'MARITAL', 'RACE', 'IDEOLOGY', 
        'INCOME', 'AREA', 'PARTY', 'AGE', 'GENDER'
    ]
    
    print("\nAnalyzing demographic distributions...")
    
    # Analyze current distributions
    distributions = {}
    for attr in demographic_attrs:
        values = [user.get(attr, 'Unknown') for user in users]
        distributions[attr] = Counter(values)
        print(f"{attr}: {len(distributions[attr])} unique values")
    
    # Create diverse sample using stratified approach
    print(f"\nCreating diverse sample of {sample_size} users...")
    
    # Group users by race and gender (primary diversity dimensions)
    primary_groups = defaultdict(list)
    for user in users:
        race = user.get('RACE', 'Unknown')
        gender = user.get('GENDER', 'Unknown')
        age = user.get('AGE', 'Unknown')
        key = (race, gender, age)
        primary_groups[key].append(user)
    
    print(f"Found {len(primary_groups)} unique race-gender-age combinations")
    
    # Calculate target sample size for each group (proportional representation)
    total_users = len(users)
    sampled_users = []
    
    # Sort groups by size to handle larger groups first
    sorted_groups = sorted(primary_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    remaining_sample = sample_size
    
    for group_key, group_users in sorted_groups:
        if remaining_sample <= 0:
            break
            
        # Calculate proportional sample size for this group
        group_proportion = len(group_users) / total_users
        target_from_group = max(1, int(group_proportion * sample_size))
        
        # Don't exceed remaining sample size or group size
        actual_sample = min(target_from_group, remaining_sample, len(group_users))
        
        # Randomly sample from this group
        group_sample = random.sample(group_users, actual_sample)
        sampled_users.extend(group_sample)
        
        remaining_sample -= actual_sample
        
        race, gender, age = group_key
        print(f"  {race} + {gender} + {age}: {actual_sample} users (from {len(group_users)} available)")
    
    # If we still need more users, randomly sample from remaining
    if len(sampled_users) < sample_size:
        sampled_ids = set(user['user_id'] for user in sampled_users)
        remaining_users = [user for user in users if user['user_id'] not in sampled_ids]
        
        additional_needed = sample_size - len(sampled_users)
        if remaining_users:
            additional_sample = random.sample(remaining_users, min(additional_needed, len(remaining_users)))
            sampled_users.extend(additional_sample)
            print(f"  Added {len(additional_sample)} additional random users")
    
    print(f"\nFinal sample size: {len(sampled_users)} users")
    
    return sampled_users

def analyze_sample_diversity(users: List[Dict], title: str = "Sample"):
    """Analyze and print diversity statistics"""
    print(f"\n{'='*50}")
    print(f"{title.upper()} DIVERSITY ANALYSIS")
    print(f"{'='*50}")
    
    demographic_attrs = [
        'RELIGIOUS', 'EMPLOYMENT', 'MARITAL', 'RACE', 'IDEOLOGY', 
        'INCOME', 'AREA', 'PARTY', 'AGE', 'GENDER'
    ]
    
    for attr in demographic_attrs:
        values = [user.get(attr, 'Unknown') for user in users]
        counter = Counter(values)
        
        print(f"\n{attr}:")
        total = len(values)
        for value, count in counter.most_common():
            percentage = (count / total) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")

def save_sample(users: List[Dict], output_file: str):
    """Save the sample to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(users)} users to: {output_file}")

def main():
    # Set random seed for reproducible results
    random.seed(42)
    
    # File paths
    input_file = '/home/taitran/mq/mq-nlp-group/synthetic_data/user_pool_X.json'
    output_file = '/home/taitran/mq/mq-nlp-group/utils/diverse_sample_500.json'
    
    try:
        # Create diverse sample
        diverse_sample = load_and_sample_diverse_users(input_file, sample_size=500)
        
        # Analyze diversity of the sample
        analyze_sample_diversity(diverse_sample, "Diverse Sample (500 users)")
        
        # Save the sample
        save_sample(diverse_sample, output_file)
        
        print(f"\n🎉 Successfully created diverse sample!")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Sample size: {len(diverse_sample)} users")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found - {input_file}")
        print("Please make sure the file path is correct.")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {input_file}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
