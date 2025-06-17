import json
import csv
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import openai
from typing import List, Dict, Tuple
import time

class SyntheticDataGenerator:
    def __init__(self, api_key: str):
        """Initialize the synthetic data generator with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        self.lock = Lock()
        
        # Language groups mapping
        self.language_groups = {
            'English': 'English',
            'Vietnamese': 'Vietnamese', 
            'Arabic': 'Arabic',
            'Cantonese': 'Cantonese',
            'Mandarin': 'Mandarin'
        }
        
        # Load data files
        self.diverse_profiles = self._load_diverse_profiles()
        self.social_categories = self._load_social_categories()
        self.base_prompt = self._load_base_prompt()
        
        # Tracking for distribution
        self.used_profile_indices = set()
        self.language_counts = {lang: 0 for lang in self.language_groups.keys()}
        self.topic_counts = {topic: 0 for topic in self.social_categories}
        
    def _load_diverse_profiles(self) -> List[Dict]:
        """Load diverse user profiles from JSON file."""
        with open('/home/taitran/mq/mq-nlp-group/2_1_Synthetic_data/diverse_sample_500.json', 'r') as f:
            return json.load(f)
    
    def _load_social_categories(self) -> List[str]:
        """Load social categories from CSV file."""
        categories = []
        with open('/home/taitran/mq/mq-nlp-group/2_1_Synthetic_data/social_category.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                categories.append(row['category'])
        return categories
    
    def _load_base_prompt(self) -> str:
        """Load the base prompt template from total_prompt.md."""
        with open('/home/taitran/mq/mq-nlp-group/2_1_Synthetic_data/total_prompt.md', 'r') as f:
            return f.read()
    
    def _get_next_profile(self) -> Dict:
        """Get the next profile ensuring even distribution."""
        with self.lock:
            # If we've used all profiles, reset and shuffle
            if len(self.used_profile_indices) >= len(self.diverse_profiles):
                self.used_profile_indices.clear()
            
            # Find unused profiles
            available_indices = [i for i in range(len(self.diverse_profiles)) 
                               if i not in self.used_profile_indices]
            
            if not available_indices:
                # This shouldn't happen with the reset above, but just in case
                available_indices = list(range(len(self.diverse_profiles)))
                self.used_profile_indices.clear()
            
            # Select random unused profile
            selected_index = random.choice(available_indices)
            self.used_profile_indices.add(selected_index)
            
            return self.diverse_profiles[selected_index]
    
    def _get_next_language(self) -> str:
        """Get the next language ensuring even distribution (1000 samples each)."""
        with self.lock:
            # Find language with minimum count
            min_count = min(self.language_counts.values())
            available_languages = [lang for lang, count in self.language_counts.items() 
                                 if count == min_count]
            
            selected_language = random.choice(available_languages)
            self.language_counts[selected_language] += 1
            
            return selected_language
    
    def _get_next_topic(self) -> str:
        """Get the next topic ensuring even distribution."""
        with self.lock:
            # Find topic with minimum count
            min_count = min(self.topic_counts.values())
            available_topics = [topic for topic, count in self.topic_counts.items() 
                              if count == min_count]
            
            selected_topic = random.choice(available_topics)
            self.topic_counts[selected_topic] += 1
            
            return selected_topic
    
    def _create_prompt(self, profile: Dict, nationality: str, topic: str) -> str:
        """Create a customized prompt for GPT-4.1-mini."""
        # Get the base prompt and customize it
        prompt = self.base_prompt
        
        # Replace the customer profile section
        profile_json = json.dumps(profile, indent=2)
        
        # Find and replace the customer profile section
        start_marker = "## Start of Customer profile"
        end_marker = "## End of customer profile"
        
        start_idx = prompt.find(start_marker)
        end_idx = prompt.find(end_marker) + len(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            new_profile_section = f"""## Start of Customer profile

  {profile_json},

Patient's nationality: {nationality}

## End of customer profile"""
            
            prompt = prompt[:start_idx] + new_profile_section + prompt[end_idx:]
        
        # Replace the topic
        prompt = prompt.replace("Topics:\nFamily", f"Topics:\n{topic}")
        
        # Replace the output language instruction
        if nationality != 'English':
            prompt = prompt.replace("All output messages should be in Vietnamese.", 
                                  f"All output messages should be in {nationality}.")
        else:
            prompt = prompt.replace("All output messages should be in Vietnamese.", 
                                  "All output messages should be in English.")
        
        return prompt
    
    def _generate_conversation(self, profile: Dict, nationality: str, topic: str) -> Dict:
        """Generate a conversation using GPT-4.1-mini."""
        try:
            prompt = self._create_prompt(profile, nationality, topic)
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",  # Using the available model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates synthetic mental health counseling conversations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.8
            )
            
            content = response.choices[0].message.content
            
            # Parse patient and doctor responses
            patient_start = content.find("<patient>")
            patient_end = content.find("</patient>")
            doctor_start = content.find("<doctor>")
            doctor_end = content.find("</doctor>")
            
            patient_message = ""
            doctor_message = ""
            
            if patient_start != -1 and patient_end != -1:
                patient_message = content[patient_start + 9:patient_end].strip()
            
            if doctor_start != -1 and doctor_end != -1:
                doctor_message = content[doctor_start + 8:doctor_end].strip()
            
            return {
                "profile": profile,
                "nationality": nationality,
                "topic": topic,
                "patient_message": patient_message,
                "doctor_message": doctor_message,
                "full_response": content,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"Error generating conversation: {str(e)}")
            return None
    
    def generate_single_sample(self, sample_id: int) -> Dict:
        """Generate a single synthetic sample."""
        try:
            # Get distributed selections
            profile = self._get_next_profile()
            nationality = self._get_next_language()
            topic = self._get_next_topic()
            
            # Generate conversation
            conversation = self._generate_conversation(profile, nationality, topic)
            
            if conversation:
                conversation["sample_id"] = sample_id
                print(f"Generated sample {sample_id}: {nationality} - {topic}")
                return conversation
            else:
                print(f"Failed to generate sample {sample_id}")
                return None
                
        except Exception as e:
            print(f"Error in generate_single_sample {sample_id}: {str(e)}")
            return None
    
    def generate_batch(self, total_samples: int = 5000, num_threads: int = 4, 
                      output_file: str = "synthetic_counseling_data.json"):
        """Generate a batch of synthetic data using parallel processing."""
        print(f"Starting generation of {total_samples} samples with {num_threads} threads...")
        
        results = []
        failed_samples = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self.generate_single_sample, i): i 
                for i in range(total_samples)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_id):
                sample_id = future_to_id[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        failed_samples.append(sample_id)
                        
                    # Progress update
                    if len(results) % 50 == 0:
                        print(f"Completed: {len(results)}/{total_samples}")
                        
                except Exception as e:
                    print(f"Sample {sample_id} generated exception: {str(e)}")
                    failed_samples.append(sample_id)
        
        # Save results
        self._save_results(results, output_file)
        
        # Print statistics
        self._print_statistics(results)
        
        if failed_samples:
            print(f"Failed samples: {len(failed_samples)} - IDs: {failed_samples[:10]}...")
        
        return results
    
    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    def _print_statistics(self, results: List[Dict]):
        """Print generation statistics."""
        print("\n=== Generation Statistics ===")
        print(f"Total samples generated: {len(results)}")
        
        # Language distribution
        lang_dist = {}
        for result in results:
            lang = result['nationality']
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        
        print("\nLanguage Distribution:")
        for lang, count in sorted(lang_dist.items()):
            print(f"  {lang}: {count}")
        
        # Topic distribution
        topic_dist = {}
        for result in results:
            topic = result['topic']
            topic_dist[topic] = topic_dist.get(topic, 0) + 1
        
        print("\nTopic Distribution (top 10):")
        sorted_topics = sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)
        for topic, count in sorted_topics[:10]:
            print(f"  {topic}: {count}")


def main():
    """Main function to run the synthetic data generation."""
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Initialize generator
    generator = SyntheticDataGenerator(api_key)
    
    # Generate synthetic data
    # For testing, start with a smaller number
    total_samples = 100  # Change to 5000 for full generation
    num_threads = 4
    
    results = generator.generate_batch(
        total_samples=total_samples,
        num_threads=num_threads,
        output_file="synthetic_counseling_data.json"
    )
    
    print(f"\nGeneration completed! Generated {len(results)} samples.")


if __name__ == "__main__":
    main()
