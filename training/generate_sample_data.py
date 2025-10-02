#!/usr/bin/env python3
"""
Sample data generator for LLaDA training.
Creates example datasets for both pre-training and SFT.
"""

import json
import random
import os
from pathlib import Path


def generate_pretraining_data(output_path, num_samples=1000):
    """Generate sample pre-training data"""
    
    # Sample texts for pre-training
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used to test typewriters and fonts.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "The solar system consists of the Sun and eight planets, along with numerous moons, asteroids, and comets.",
        "Programming is the process of creating instructions for computers to execute specific tasks.",
        "Natural language processing enables computers to understand and generate human language.",
        "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level.",
        "The Internet has revolutionized how people communicate, work, and access information worldwide.",
        "Biodiversity refers to the variety of life forms found in different ecosystems on Earth."
    ]
    
    # Additional educational content
    educational_texts = [
        "Mathematics is the foundation of many scientific disciplines. It provides tools for understanding patterns, relationships, and structures in the natural world.",
        "History helps us understand the past and learn from previous experiences. It provides context for current events and future decisions.",
        "Literature offers insights into human nature and society through storytelling, poetry, and dramatic works.",
        "Science is a systematic method for understanding the natural world through observation, experimentation, and analysis.",
        "Philosophy examines fundamental questions about existence, knowledge, values, and the nature of reality."
    ]
    
    # Technical content
    technical_texts = [
        "Deep learning models use multiple layers of neural networks to automatically learn features from data.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Blockchain technology creates distributed ledgers that record transactions across multiple computers.",
        "Data visualization transforms complex datasets into understandable graphs, charts, and interactive displays.",
        "Cybersecurity protects digital systems, networks, and data from unauthorized access and attacks."
    ]
    
    all_texts = sample_texts + educational_texts + technical_texts
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Select random text and possibly extend it
            base_text = random.choice(all_texts)
            
            # Sometimes combine multiple texts
            if random.random() < 0.3:
                additional_text = random.choice(all_texts)
                text = f"{base_text} {additional_text}"
            else:
                text = base_text
            
            # Add some variation
            if random.random() < 0.1:
                text = text.upper()
            elif random.random() < 0.1:
                text = text.lower()
            
            data = {"text": text}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Generated {num_samples} pre-training samples in {output_path}")


def generate_sft_data(output_path, num_samples=500):
    """Generate sample SFT data with conversations"""
    
    # Sample conversation templates
    conversations = [
        {
            "conversations": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris. It is located in the north-central part of the country and is known for its art, fashion, and culture."}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "Explain what machine learning is."},
                {"role": "assistant", "content": "Machine learning is a branch of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns."}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "How do you make a sandwich?"},
                {"role": "assistant", "content": "To make a basic sandwich: 1) Take two slices of bread, 2) Add your desired filling (like meat, cheese, vegetables), 3) Add condiments if desired, 4) Place the second slice of bread on top, 5) Cut diagonally if preferred. Enjoy your sandwich!"}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "What are the benefits of exercise?"},
                {"role": "assistant", "content": "Regular exercise provides numerous benefits including: improved cardiovascular health, stronger muscles and bones, better mental health and mood, increased energy levels, better sleep quality, and reduced risk of chronic diseases like diabetes and heart disease."}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "Tell me about the solar system."},
                {"role": "assistant", "content": "The solar system consists of the Sun at its center, eight planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune), their moons, asteroids, comets, and other celestial bodies. It formed about 4.6 billion years ago from a collapsing cloud of gas and dust."}
            ]
        }
    ]
    
    # Question types for variation
    question_templates = [
        ("What is {topic}?", "explanation"),
        ("How does {topic} work?", "explanation"),
        ("Can you explain {topic}?", "explanation"),
        ("Tell me about {topic}.", "description"),
        ("What are the benefits of {topic}?", "benefits"),
        ("How to {action}?", "instruction")
    ]
    
    topics = [
        "photosynthesis", "gravity", "democracy", "programming", "cooking",
        "reading", "mathematics", "history", "chemistry", "biology",
        "physics", "literature", "art", "music", "sports"
    ]
    
    actions = [
        "learn a new language", "start exercising", "improve memory",
        "save money", "cook pasta", "write a letter", "study effectively",
        "manage time", "reduce stress", "stay organized"
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write base conversations
        for conversation in conversations:
            for _ in range(num_samples // (len(conversations) + 50)):  # Repeat base conversations
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        # Generate additional varied conversations
        remaining_samples = num_samples - (num_samples // (len(conversations) + 50)) * len(conversations)
        
        for i in range(remaining_samples):
            if random.random() < 0.7:  # 70% questions about topics
                topic = random.choice(topics)
                question_template, response_type = random.choice(question_templates)
                
                # Check if template needs action or topic
                if "{action}" in question_template:
                    action = random.choice(actions)
                    question = question_template.format(action=action)
                    response = f"To {action}, you should start with understanding the basics, create a plan, practice regularly, and be patient with yourself. Consistency and dedication are key to success."
                else:
                    question = question_template.format(topic=topic)
                    if response_type == "explanation":
                        response = f"{topic.capitalize()} is an important concept that involves various processes and mechanisms. It plays a significant role in many areas and has practical applications in daily life."
                    elif response_type == "benefits":
                        response = f"The benefits of {topic} include improved understanding, practical skills, and personal development. It can enhance your knowledge and provide valuable insights."
                    else:
                        response = f"{topic.capitalize()} is a fascinating subject with many interesting aspects. It encompasses various elements and has influenced many fields of study."
            
            else:  # 30% how-to questions
                action = random.choice(actions)
                question = f"How to {action}?"
                response = f"To {action}, you should start with understanding the basics, create a plan, practice regularly, and be patient with yourself. Consistency and dedication are key to success."
            
            conversation = {
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
            }
            
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    
    print(f"Generated {num_samples} SFT samples in {output_path}")


def main():
    """Generate sample datasets"""
    
    print("Generating sample datasets for LLaDA training...")
    
    # Create data directories
    os.makedirs("data/pretrain", exist_ok=True)
    os.makedirs("data/sft", exist_ok=True)
    
    # Generate pre-training data
    generate_pretraining_data("data/pretrain/train.jsonl", 2000)
    generate_pretraining_data("data/pretrain/eval.jsonl", 200)
    
    # Generate SFT data
    generate_sft_data("data/sft/train.jsonl", 1000)
    generate_sft_data("data/sft/eval.jsonl", 100)
    
    print("\nSample data generation completed!")
    print("Files created:")
    print("- data/pretrain/train.jsonl (2000 samples)")
    print("- data/pretrain/eval.jsonl (200 samples)")
    print("- data/sft/train.jsonl (1000 samples)")
    print("- data/sft/eval.jsonl (100 samples)")
    
    print("\nTo use this data for training, update the paths in config.yaml:")
    print("  pretraining.train_data_path: './data/pretrain/train.jsonl'")
    print("  pretraining.eval_data_path: './data/pretrain/eval.jsonl'")
    print("  sft.train_data_path: './data/sft/train.jsonl'")
    print("  sft.eval_data_path: './data/sft/eval.jsonl'")


if __name__ == "__main__":
    main()