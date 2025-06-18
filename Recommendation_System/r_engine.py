import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import sys
import yaml 
import google.generativeai as genai 
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
import anthropic
import re

# Define paths
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "Database_tables", "")
output_path = os.path.join(script_dir, "Output", "")
future_suggestions_path = os.path.join(script_dir, "Future_Suggestions", "")

# Creating directories if they don't exist
os.makedirs(output_path, exist_ok=True)
os.makedirs(future_suggestions_path, exist_ok=True)

# Load API keys from secrets.yaml
GEMINI_API_KEY = None # Initialize
try:
    with open("secrets.yaml", 'r') as stream:
        secrets = yaml.safe_load(stream)
    OPENAI_API_KEY = secrets.get("OpenAI Chatgpt Key")
    ANTHROPIC_API_KEY = secrets.get("Anthropic Claude API Key") # Added Anthropic key
    PERPLEXITY_API_KEY = secrets.get("Perplexity API Key") # Added Perplexity key
    GEMINI_API_KEY = secrets.get("Google Gemini API Key")
    RAPIDAPI_KEY = secrets.get("RAPIDAPI_KEY") # Added RapidAPI Key
    
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY) # Configure Gemini
    else:
        print("Warning: Google Gemini API Key not found in secrets.yaml. Gemini calls will be skipped.")

except FileNotFoundError:
    print("ℹ️  Running without API keys (secrets.yaml not found). Basic functionality available.")
    OPENAI_API_KEY = None
    ANTHROPIC_API_KEY = None
    PERPLEXITY_API_KEY = None
    # GEMINI_API_KEY is already None
except yaml.YAMLError as exc:
    print(f"Error parsing secrets.yaml: {exc}")
    OPENAI_API_KEY = None
    ANTHROPIC_API_KEY = None
    PERPLEXITY_API_KEY = None
    # GEMINI_API_KEY is already None
except Exception as e: # Catching potential errors from genai.configure
    print(f"Error configuring Google Gemini API: {e}")
    GEMINI_API_KEY = None # Ensure key is considered unusable

# OpenAI API Configuration
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-4o-mini"

# Anthropic API Configuration
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL_NAME = "claude-3-haiku-20240307"

# Perplexity API Configuration
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
# Find a suitable Perplexity model, e.g., "llama-3-sonar-small-32k-chat" or "llama-3-sonar-large-32k-chat"
PERPLEXITY_MODEL_NAME = "llama-3-sonar-small-32k-chat" 

# Google Gemini API Configuration
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Updated to a more available model

# Define a structure for an empty plan
EMPTY_PLAN_STRUCTURE = {
    "medication_recommendations": [],
    "surgical_recommendations": [],
    "other_treatments": [],
    "next_steps": [],
    "follow_up_timeline": [],
    "health_goals": [],
    "lifestyle_adjustments": [],
    "warning_signs": []
}

# Load datasets
print("Loading datasets...")
datasets = {
    "patients": pd.read_csv(data_path + "patients_transformed.csv"),
    "medications": pd.read_csv(data_path + "medications_transformed.csv"),
    "conditions": pd.read_csv(data_path + "conditions_transformed.csv"),
    "allergies": pd.read_csv(data_path + "allergies_transformed.csv"),
    "careplans": pd.read_csv(data_path + "careplans_transformed.csv"),
    "encounters": pd.read_csv(data_path + "encounters_transformed.csv"),
    "observations": pd.read_csv(data_path + "observations_transformed.csv")
}

# Load recommendation features if available
try:
    recommendation_features = pd.read_csv(data_path + "recommendation_features.csv", index_col=0)
    print("Recommendation features loaded successfully.")
except FileNotFoundError:
    print("Warning: Recommendation features not found. Will use basic recommendations only.")
    recommendation_features = None

# Load data (before the function definitions)
patient_df = pd.read_csv(data_path + "patients.csv", index_col='patient')
patient_features = pd.read_csv(data_path + "patient_features.csv", index_col=0)

def call_openai_api(prompt):
    """
    Call OpenAI API with the given prompt
    
    Args:
        prompt: The text prompt to send to the API
        
    Returns:
        The response text from the API
    """
    if not OPENAI_API_KEY:
        print("OpenAI API key not found. Skipping OpenAI call.")
        return None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 3000 # Increased max_tokens slightly for potentially longer prompt
    }
    
    try:
        print("Sending request to OpenAI API (timeout: 180 seconds)...")
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=180)
        response.raise_for_status()
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print("Request to OpenAI API timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request to OpenAI API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error calling OpenAI API: {e}")
        return None

def call_anthropic_api(prompt):
    """
    Call Anthropic API with the given prompt
    
    Args:
        prompt: The text prompt to send to the API
        
    Returns:
        The response text from the API
    """
    if not ANTHROPIC_API_KEY:
        print("Anthropic API key not found. Skipping Anthropic call.")
        return None
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": ANTHROPIC_MODEL_NAME,
        "max_tokens": 3500,
        "system": "You are a helpful medical AI assistant. Follow the user's instructions carefully and provide responses in the requested format.",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        print("Sending request to Anthropic API (timeout: 180 seconds)...")
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=180)
        response.raise_for_status()
        
        response_data = response.json()
        if response_data.get("content") and isinstance(response_data["content"], list) and len(response_data["content"]) > 0:
            return response_data["content"][0].get("text")
        else:
            print(f"Unexpected Anthropic API response format: {response_data}")
            return None
    except requests.exceptions.Timeout:
        print("Request to Anthropic API timed out.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred with Anthropic API: {http_err}") 
        try:
            error_details = http_err.response.json()
            print(f"Anthropic API error details: {json.dumps(error_details, indent=2)}")
        except json.JSONDecodeError:
            print(f"Anthropic API error response (not JSON): {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Anthropic API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error calling Anthropic API: {e}")
        return None

def call_perplexity_api(prompt):
    """
    Call Perplexity API with the given prompt
    """
    if not PERPLEXITY_API_KEY:
        print("Perplexity API key not found. Skipping Perplexity call.")
        return None
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": PERPLEXITY_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 3000
    }
    
    try:
        print("Sending request to Perplexity API (timeout: 180 seconds)...")
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=data, timeout=180)
        response.raise_for_status()
        
        response_data = response.json()
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            print(f"Unexpected Perplexity API response format: {response_data}")
            return None
    except requests.exceptions.Timeout:
        print("Request to Perplexity API timed out.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred with Perplexity API: {http_err}")
        try:
            error_details = http_err.response.json()
            print(f"Perplexity API error details: {json.dumps(error_details, indent=2)}")
        except (json.JSONDecodeError, ValueError):
            print(f"Perplexity API error response (not JSON): {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Perplexity API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error calling Perplexity API: {e}")
        return None

def call_gemini_api(prompt):
    """
    Call Google Gemini API with the given prompt using the SDK.
    """
    if not GEMINI_API_KEY:
        print("GEMINI_CALL_DEBUG: Gemini API key not available or configuration failed. Skipping Gemini call.")
        return None
    
    # Add a small marker to differentiate suggestion prompts from formatting prompts if needed for debugging
    prompt_type = "Suggestion Prompt" if len(prompt) < 2000 else "Formatting Prompt" # Heuristic for differentiation
    print(f"GEMINI_CALL_DEBUG: Calling Google Gemini API (model: {GEMINI_MODEL_NAME}) for: {prompt_type}")
    # print(f"GEMINI_CALL_DEBUG: Prompt (first 300 chars): {prompt[:300]}...") # Optional: print part of prompt

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=3500, 
            temperature=0.7,
        )
        print(f"GEMINI_CALL_DEBUG: Sending content to Gemini for {prompt_type}...")
        response = model.generate_content(prompt, generation_config=generation_config)
        print(f"GEMINI_CALL_DEBUG: Received response object from Gemini for {prompt_type}.")

        if hasattr(response, 'text') and response.text:
            print(f"GEMINI_CALL_DEBUG: Gemini returned text for {prompt_type}.")
            return response.text
        else:
            print(f"GEMINI_CALL_DEBUG: Gemini API call for {prompt_type} succeeded but returned no text content or 'text' attribute missing.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 print(f"GEMINI_CALL_DEBUG: Gemini Prompt Feedback for {prompt_type}: {response.prompt_feedback}")
            else:
                 print(f"GEMINI_CALL_DEBUG: No additional feedback available in response object for {prompt_type}. Response was: {response}")
            return None
    except Exception as e:
        print(f"GEMINI_CALL_DEBUG: Error calling Google Gemini API for {prompt_type}: {e}")
        import traceback
        print(f"GEMINI_CALL_DEBUG: Traceback:\n{traceback.format_exc()}")
    return None

def process_llm_response(response_text):
    """
    Process the LLM response text into a structured format.
    Extracts lines of text under known section headers.
    Handles various header formats including numbered sections.
    """
    if not response_text:
        return EMPTY_PLAN_STRUCTURE.copy()
    
    import re
    
    processed_plan = EMPTY_PLAN_STRUCTURE.copy()
    lines = response_text.split('\n')
    current_section_key = None
    
    # Keywords to identify sections - improved matching
    section_header_keywords = {
        "medication_recommendations": ["medication recommendation", "medication", "drug", "prescription", "pharmaceutical", "medicine"],
        "surgical_recommendations": ["surgical recommendation", "surgical", "surgery", "procedure", "operation"],
        "other_treatments": ["other treatment", "non-pharmacological", "therapy", "supplement", "nutrition"],
        "next_steps": ["next step", "action", "immediate", "plan"],
        "follow_up_timeline": ["follow-up", "follow up", "appointment", "visit", "timeline"],
        "health_goals": ["health goal", "goal", "target", "objective", "milestone"],
        "lifestyle_adjustments": ["lifestyle", "exercise", "diet", "sleep", "activity"],
        "warning_signs": ["warning sign", "alert", "emergency", "danger", "monitor", "watch"]
    }

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
                continue
            
        # Clean line for header detection - remove numbering and bold markers
        header_check_line = stripped_line
        header_check_line = re.sub(r'^\*\*', '', header_check_line)  # Remove leading **
        header_check_line = re.sub(r'\*\*$', '', header_check_line)  # Remove trailing **
        header_check_line = re.sub(r'^\d+[\.\)]\s*', '', header_check_line)  # Remove "1. " or "1) "
        header_check_line = header_check_line.strip()
        
        found_new_section = False
        for key, keywords in section_header_keywords.items():
            for keyword in keywords:
                # More flexible header matching
                if keyword.lower() in header_check_line.lower():
                    # Check if this looks like a header (not too long, ends with colon, or is a standalone phrase)
                    if (len(header_check_line) < 80 and 
                        (header_check_line.endswith(':') or 
                         header_check_line.endswith('s:') or
                         len(header_check_line.split()) < 6)):
                        
                        current_section_key = key
                        found_new_section = True
                        break
            if found_new_section:
                    break
            
        if found_new_section:
            continue  # Skip the header line
        
        # Process content under current section
        if current_section_key and stripped_line:
            # Clean up the content line
            content = stripped_line
            
            # Remove common bullet points and numbering
            content = re.sub(r'^\s*[\•\-\*\+]\s*', '', content)  # Bullets
            content = re.sub(r'^\s*\d+[\.\)]\s*', '', content)  # Numbered lists
            content = re.sub(r'^\*\*', '', content)  # Bold start
            content = re.sub(r'\*\*$', '', content)  # Bold end
            content = content.strip()
            
            # Skip empty content or dividers
            if content and not content.startswith(('---', '...', '___')):
                # For medications, handle complex formats
                if current_section_key == "medication_recommendations" and ':' in content:
                    # Extract medication name and full recommendation
                    processed_plan[current_section_key].append(content)
                else:
                    processed_plan[current_section_key].append(content)
    
    # Ensure all sections exist
    for key in EMPTY_PLAN_STRUCTURE.keys():
        if key not in processed_plan:
            processed_plan[key] = []
    
    return processed_plan

def consolidate_and_deduplicate_suggestions(list_of_processed_plans):
    """
    Consolidates raw string suggestions from multiple LLM processed plans.
    Ensures uniqueness of items in each section.
    Input: list of dictionaries, where each dict is a processed plan {section_key: [suggestion_string, ...]}.
    Output: a single dictionary {section_key: [unique_suggestion_string, ...]}.
    """
    if not list_of_processed_plans:
        return EMPTY_PLAN_STRUCTURE.copy()

    consolidated_suggestions = {key: set() for key in EMPTY_PLAN_STRUCTURE.keys()}

    for plan in list_of_processed_plans:
        if not plan: continue
        for section_key, suggestions_list in plan.items():
            if section_key in consolidated_suggestions and isinstance(suggestions_list, list):
                for suggestion_string in suggestions_list:
                    if isinstance(suggestion_string, str) and suggestion_string.strip():
                        consolidated_suggestions[section_key].add(suggestion_string.strip())
    
    # Convert sets back to sorted lists
    final_deduplicated_plan = {key: sorted(list(item_set)) for key, item_set in consolidated_suggestions.items()}
    return final_deduplicated_plan

def format_final_plan_with_gemini(consolidated_suggestions_dict, example_structure_text, patient_record=None, history=None):
    """
    Takes consolidated suggestions and formats them using Gemini to create a complete health plan document.
    """
    if not GEMINI_API_KEY:
        print("Gemini API key not available. Using basic fallback for final plan formatting.")
        # Fallback formatting logic
        fallback_formatted_plan = "FUTURE HEALTH PLAN (Fallback - Gemini API Key Missing):\n\n"
        for section, items in consolidated_suggestions_dict.items():
            header = section.replace("_", " ").title()
            fallback_formatted_plan += f"{header}:\n"
            if items:
                for item in items:
                    fallback_formatted_plan += f"- {item}\n"
            else:
                fallback_formatted_plan += "- None\n"
            fallback_formatted_plan += "\n"
        return fallback_formatted_plan

    # Prepare complete patient information for the prompt
    patient_info_text = ""
    if patient_record and history is not None:
        from datetime import datetime
        import pandas as pd
        
        birthdate = pd.to_datetime(patient_record["birthdate"])
        age = (datetime.now() - birthdate).days // 365
        gender = patient_record["gender"]
        address = patient_record["address"]
        full_name = f"{patient_record['first']} {patient_record['last']}"
        patient_id = patient_record['patient']
        
        patient_info_text = f"""
COMPLETE PATIENT DATA TO USE:
- Patient Name: {full_name}
- Patient ID: {patient_id}
- Age: {age} years
- Gender: {gender}
- Address: {address}

Current Conditions:
"""
        if not history["conditions"].empty:
            for _, condition in history["conditions"].iterrows():
                patient_info_text += f"- {condition['DESCRIPTION']} (since {condition['START']})\n"
        else:
            patient_info_text += "- None\n"
    
        patient_info_text += "\nCurrent Medications:\n"
        if not history["medications"].empty:
            for _, medication in history["medications"].iterrows():
                status = "ACTIVE" if pd.isna(medication['STOP']) else f"Until {medication['STOP']}"
                patient_info_text += f"- {medication['DESCRIPTION']} - {status}\n"
        else:
            patient_info_text += "- None\n"
    
        patient_info_text += "\nAllergies:\n"
        if not history["allergies"].empty:
            for _, allergy in history["allergies"].iterrows():
                patient_info_text += f"- {allergy['DESCRIPTION']}\n"
        else:
            patient_info_text += "- None\n"
    
        patient_info_text += "\nRecent Measurements:\n"
        if not history["observations"].empty:
            recent_observations = history["observations"].sort_values("DATE", ascending=False).head(5)
            for _, observation in recent_observations.iterrows():
                patient_info_text += f"- {observation['DESCRIPTION']}: {observation['VALUE']} {observation['UNITS']} ({observation['DATE']})\n"
        else:
            patient_info_text += "- None\n"

    # Construct the prompt for Gemini to create a COMPLETE document
    gemini_prompt_parts = [f"""You are a medical writing assistant. Create a complete, professional health plan document using the patient data and medical recommendations below.

INSTRUCTIONS:
1. Create a COMPLETE health plan document starting with the header
2. Use the PATIENT DATA to fill in all patient information sections
3. Include current conditions, medications, allergies, and recent measurements
4. Use the CONSOLIDATED SUGGESTIONS for the future health plan recommendations
5. Create one cohesive document without duplication
6. Use professional medical formatting
7. Include appropriate disclaimers at the end

{patient_info_text}

CONSOLIDATED FUTURE HEALTH PLAN SUGGESTIONS:"""]
    
    for section_key, items_list in consolidated_suggestions_dict.items():
        header = section_key.replace("_", " ").replace("recommendations", "").replace("timeline", "").replace("adjustments", "").strip().title()
        if "Medication" in header: header = "Latest Medication Alternatives to Consider"
        elif "Surgical" in header: header = "Recommended Surgical Procedures"
        gemini_prompt_parts.append(f"\n## {header}:\n")
        if items_list:
            for item_str in items_list:
                gemini_prompt_parts.append(f"- {item_str}\n")
        else:
            gemini_prompt_parts.append("- None provided.\n")
    
    gemini_prompt_parts.append(f"""

INSTRUCTIONS FOR COMPLETE DOCUMENT CREATION:
1. Start with a header: "PERSONALIZED HEALTH PLAN FOR: [Patient Name]"
2. Include Patient ID and current date
3. Add patient information section with age, gender, address
4. List current conditions with dates
5. List current medications with status
6. Include allergies if any
7. Add recent measurements
8. Create a comprehensive FUTURE HEALTH PLAN section using the consolidated suggestions above
9. End with appropriate medical disclaimers
10. Use consistent formatting throughout (use "•" or "-" for bullets, not "?")
11. Do NOT create duplicate sections
12. Make it one complete, professional document

Create the complete health plan document now:""")
    
    final_gemini_prompt = "".join(gemini_prompt_parts)

    print("\nCalling Gemini to create complete health plan document...")
    gemini_formatted_text = call_gemini_api(final_gemini_prompt)

    if gemini_formatted_text:
        print("Successfully created complete health plan document with Gemini.")
        return gemini_formatted_text
    else:
        print("Gemini formatting call failed. Using basic fallback.")
        # Create a basic complete document as fallback
        if patient_record and history is not None:
            full_name = f"{patient_record['first']} {patient_record['last']}"
            fallback_plan = f"PERSONALIZED HEALTH PLAN FOR: {full_name}\n"
            fallback_plan += f"Patient ID: {patient_record['patient']}\n"
            fallback_plan += f"Date Generated: {datetime.now().strftime('%Y-%m-%d')}\n"
            fallback_plan += "=" * 60 + "\n\n"
            fallback_plan += "FUTURE HEALTH PLAN (Fallback - Gemini Call Failed):\n\n"
            for section, items in consolidated_suggestions_dict.items():
                header = section.replace("_", " ").title()
                fallback_plan += f"{header}:\n"
                if items:
                    for item in items:
                        fallback_plan += f"- {item}\n"
                else:
                    fallback_plan += "- None\n"
                fallback_plan += "\n"
            return fallback_plan
        else:
            return "Error: Unable to create health plan due to missing patient data."

def generate_future_health_plan(patient_record, history, recommendations):
    """
    Generates a future health plan by querying multiple LLMs, consolidating suggestions,
    and then formatting them using a final LLM (Gemini).
    """
    # ... (patient data prep for LLM prompt - remains similar)
    birthdate = pd.to_datetime(patient_record["birthdate"])
    age = (datetime.now() - birthdate).days // 365
    gender = patient_record["gender"]
    current_conditions_list = [cond['DESCRIPTION'] for _, cond in history["conditions"].iterrows()] if not history["conditions"].empty else ['None']
    current_medications_list = [med['DESCRIPTION'] for _, med in history["medications"].iterrows()] if not history["medications"].empty else ['None']
    allergies_list_str = ', '.join([alg['DESCRIPTION'] for _, alg in history["allergies"].iterrows()]) if not history["allergies"].empty else 'None'
    recent_observations_list = []
    if not history["observations"].empty:
        recent_obs = history["observations"].sort_values("DATE", ascending=False).head(10)
        for _, observation in recent_obs.iterrows():
            recent_observations_list.append(f"{observation['DESCRIPTION']}: {observation['VALUE']} {observation['UNITS']}")
    recent_observations_str = ', '.join(recent_observations_list) if recent_observations_list else 'None'

    # Initial prompt for information gathering (improved to get specific drug names)
    initial_llm_prompt = f"""
    You are a medical AI assistant with 20 years of experience. Based on the following patient profile, please provide SPECIFIC medication recommendations with exact drug names and dosages for their future health plan.

    IMPORTANT: 
    - Provide SPECIFIC drug names (e.g., "Morphine 10mg every 4 hours" NOT "Pain medication")
    - Include exact dosages and frequencies (e.g., "Ondansetron 8mg every 8 hours" NOT "Antiemetics")
    - For each medication, specify the indication (why it's prescribed)
    - Only recommend medications that are appropriate for this patient's conditions
    - Do NOT use generic categories like "antibiotics" or "pain medications"

    PATIENT PROFILE:
    Age: {age}
    Gender: {gender}
    Current Conditions: {', '.join(current_conditions_list)}
    Current Medications: {', '.join(current_medications_list)}
    Allergies: {allergies_list_str}
    Recent Measurements: {recent_observations_str}

    Please provide recommendations in these categories:

    MEDICATION RECOMMENDATIONS:
    - List specific drug names with dosages (e.g., "Morphine sulfate 15mg every 4 hours for pain management")
    - Include route of administration (oral, IV, etc.)

    SURGICAL PROCEDURES:
    - List specific procedures that may be needed

    NON-PHARMACOLOGICAL TREATMENTS:
    - Physical therapy, dietary changes, counseling, etc.

    NEXT STEPS:
    - Immediate actions needed (appointments, tests, etc.)

    FOLLOW-UP TIMELINE:
    - Specific scheduling for follow-ups

    HEALTH GOALS:
    - Specific, measurable goals

    LIFESTYLE ADJUSTMENTS:
    - Specific recommendations for daily life

    WARNING SIGNS:
    - Specific symptoms to watch for

    Focus on SPECIFIC drug names, not categories!
    """

    all_llm_parsed_outputs = []

    # Call OpenAI
    print("\nQuerying OpenAI...")
    openai_text = call_openai_api(initial_llm_prompt)
    if openai_text:
        print("--- RAW OPENAI RESPONSE (first 1000 chars) ---")
        print(f"{openai_text[:1000]}...")
        print("--- END RAW OPENAI RESPONSE ---")
        parsed_openai = process_llm_response(openai_text)
        all_llm_parsed_outputs.append(parsed_openai)
        print("OpenAI response processed.")
    else:
        print("OpenAI call failed or returned no text.")

    # Call Anthropic
    print("\nQuerying Anthropic...")
    anthropic_text = call_anthropic_api(initial_llm_prompt)
    if anthropic_text:
        print("--- RAW ANTHROPIC RESPONSE (first 1000 chars) ---")
        print(f"{anthropic_text[:1000]}...")
        print("--- END RAW ANTHROPIC RESPONSE ---")
        all_llm_parsed_outputs.append(process_llm_response(anthropic_text))
        print("Anthropic response processed.")
    else:
        print("Anthropic call failed or returned no text (credit issue likely).")

    # Call Perplexity
    print("\nQuerying Perplexity...")
    perplexity_text = call_perplexity_api(initial_llm_prompt)
    if perplexity_text:
        print("--- RAW PERPLEXITY RESPONSE (first 1000 chars) ---")
        print(f"{perplexity_text[:1000]}...")
        print("--- END RAW PERPLEXITY RESPONSE ---")
        all_llm_parsed_outputs.append(process_llm_response(perplexity_text))
        print("Perplexity response processed.")
    else:
        print("Perplexity call skipped or failed (placeholder).")

    # Call Gemini for initial suggestions
    print("\nQuerying Gemini for initial suggestions...")
    gemini_suggestion_text = call_gemini_api(initial_llm_prompt) 
    if gemini_suggestion_text: 
        print("--- RAW GEMINI SUGGESTION RESPONSE (first 1000 chars) ---")
        print(f"{gemini_suggestion_text[:1000]}...")
        print("--- END RAW GEMINI SUGGESTION RESPONSE ---")
        all_llm_parsed_outputs.append(process_llm_response(gemini_suggestion_text))
        print("Gemini (suggestions) response processed.")
    else:
        print("Gemini (suggestions) call skipped or failed.")

    if not all_llm_parsed_outputs:
        print("No LLM provided any suggestions. Returning empty plan.")
        return "FUTURE HEALTH PLAN:\n\nNo recommendations available due to LLM processing errors or empty responses.\n"

    # Consolidate all gathered suggestions
    print(f"\nConsolidating suggestions from {len(all_llm_parsed_outputs)} LLM(s)...")
    consolidated_suggestions = consolidate_and_deduplicate_suggestions(all_llm_parsed_outputs)
    print("Consolidation complete.")
    
    # Phase 2: Verify and enrich medications using external APIs
    enriched_suggestions = verify_and_enrich_medications(consolidated_suggestions)
    
    # Load the example structure text for final formatting
    example_plan_structure_text = ""
    try:
        example_file_path = os.path.join(data_path, "output", "f0438525-1d1c-4e55-81c5-ab5b8d15e7e8_health_plan.txt")
        if os.path.exists(example_file_path):
            with open(example_file_path, 'r', encoding='utf-8', errors='replace') as f:
                example_plan_structure_text = f.read()
        else:
            print(f"Warning: Example plan file for final formatting not found at {example_file_path}. Gemini formatting might be poor.")
            # Basic structure if file not found (Gemini might struggle with just this)
            example_plan_structure_text = "Provide a detailed future health plan with sections for medications, surgery, other treatments, next steps, follow-up, goals, lifestyle, and warnings."
    except Exception as e:
        print(f"Error reading example plan structure for final formatting: {e}. Using basic instructions for Gemini.")
        example_plan_structure_text = "Provide a detailed future health plan with sections for medications, surgery, other treatments, next steps, follow-up, goals, lifestyle, and warnings."

    # Format the consolidated plan using Gemini
    print("\nAttempting to format the final plan using Google Gemini...")
    final_formatted_plan_string = format_final_plan_with_gemini(enriched_suggestions, example_plan_structure_text, patient_record, history)
    
    return final_formatted_plan_string

def create_personalized_plan(patient_record, recommendations):
    """Create a detailed health plan based on patient data and recommendations"""
    
    # Get patient history
    patient_id = patient_record['patient']
    history = get_patient_history(patient_id)
    
    # Calculate patient age
    birthdate = pd.to_datetime(patient_record["birthdate"])
    age = (datetime.now() - birthdate).days // 365
    full_name = f"{patient_record['first']} {patient_record['last']}"
    
    # ====================================================================================
    # FILE 1: CURRENT PATIENT STATUS (Clean, well-formatted current data from CSV)
    # ====================================================================================
    
    current_status = f"""
===============================================================================
                           CURRENT PATIENT STATUS                                  
===============================================================================

PATIENT INFORMATION:
    • Full Name: {full_name}
    • Patient ID: {patient_id}
    • Age: {age} years
    • Gender: {patient_record['gender']}
    • Date of Birth: {patient_record['birthdate']}
    • Address: {patient_record['address']}
    • Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT MEDICAL CONDITIONS:
"""
    
    if not history["conditions"].empty:
        for _, condition in history["conditions"].iterrows():
            current_status += f"    • {condition['DESCRIPTION']}\n"
            current_status += f"      └─ Started: {condition['START']}\n"
            if pd.notna(condition.get('STOP')):
                current_status += f"      └─ Ended: {condition['STOP']}\n"
    else:
        current_status += "    • No current medical conditions on record\n"
    
    current_status += f"""
CURRENT MEDICATIONS:
"""
    
    if not history["medications"].empty:
        active_meds = 0
        for _, medication in history["medications"].iterrows():
            if pd.isna(medication.get('STOP')):
                status = "[ACTIVE]"
                active_meds += 1
            else:
                status = f"[STOPPED - {medication['STOP']}]"
            current_status += f"    • {medication['DESCRIPTION']}\n"
            current_status += f"      └─ Status: {status}\n"
            current_status += f"      └─ Started: {medication['START']}\n"
        current_status += f"\n    Summary: {active_meds} active medications\n"
    else:
        current_status += "    • No medications on record\n"
    
    current_status += f"""
KNOWN ALLERGIES:
"""
    
    if not history["allergies"].empty:
        for _, allergy in history["allergies"].iterrows():
            current_status += f"    • {allergy['DESCRIPTION']}\n"
            current_status += f"      └─ Since: {allergy['START']}\n"
    else:
        current_status += "    • No known allergies\n"
    
    current_status += f"""
RECENT MEASUREMENTS & OBSERVATIONS:
"""
    
    if not history["observations"].empty:
        recent_obs = history["observations"].sort_values("DATE", ascending=False).head(10)
        for _, observation in recent_obs.iterrows():
            current_status += f"    • {observation['DESCRIPTION']}: {observation['VALUE']} {observation['UNITS']}\n"
            current_status += f"      └─ Date: {observation['DATE']}\n"
    else:
        current_status += "    • No recent measurements available\n"
    
    current_status += f"""
RECENT HEALTHCARE ENCOUNTERS:
"""
    
    if not history["encounters"].empty:
        # Check which date column exists in encounters
        date_columns = ['START', 'Start', 'start', 'DATE', 'Date', 'date']
        encounter_date_col = None
        for col in date_columns:
            if col in history["encounters"].columns:
                encounter_date_col = col
                break
        
        if encounter_date_col:
            recent_enc = history["encounters"].sort_values(encounter_date_col, ascending=False).head(5)
            for _, enc in recent_enc.iterrows():
                desc = enc.get('DESCRIPTION', 'Unknown encounter')
                reason = enc.get('REASONDESCRIPTION', 'No reason specified')
                date = enc.get(encounter_date_col, 'Unknown date')
                status_text += f"• {desc} - {reason} ({date})\n"
        else:
            # If no date column found, just show first 5 encounters
            for idx, (_, enc) in enumerate(history["encounters"].head(5).iterrows()):
                desc = enc.get('DESCRIPTION', 'Unknown encounter')
                reason = enc.get('REASONDESCRIPTION', 'No reason specified')
                status_text += f"• {desc} - {reason}\n"
    else:
        status_text += "• None\n"
    
    current_status += f"""

===============================================================================
This file contains CURRENT patient data extracted from medical records (CSV)      
For FUTURE health recommendations, see: ../Future_Suggestions/{patient_id}_future_plan.txt              
===============================================================================
"""
    
    # Save current status file in main Database_tables directory
    current_status_filename = f"{patient_id}_current_status.txt"
    current_status_path = os.path.join(output_path, current_status_filename)
    
    with open(current_status_path, 'w', encoding='utf-8') as f:
        f.write(current_status)
    
    # ====================================================================================
    # FILE 2: FUTURE HEALTH PLAN (LLM recommendations + API verifications)
    # ====================================================================================
    
    # Generate the LLM-based future health plan
    future_plan_content = generate_future_health_plan(patient_record, history, recommendations)
    
    # Create a clean header for the future plan
    future_plan = f"""
===============================================================================
                     AI-POWERED FUTURE HEALTH PLAN                                 
===============================================================================

GENERATED FOR: {full_name}
PATIENT ID: {patient_id}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AI MODELS USED: OpenAI GPT-4, Google Gemini, Medical APIs (openFDA, RxNorm)

"""
    
    # Add the LLM-generated content
    future_plan += future_plan_content
    
    # Add footer with data sources
    future_plan += f"""

===============================================================================
                              DATA SOURCES & VERIFICATION                          
===============================================================================

AI MODELS:
    • OpenAI GPT-4: Medical recommendations and analysis
    • Google Gemini: Document formatting and synthesis
    • Anthropic Claude: Additional medical insights (when available)

MEDICAL DATABASES:
    • openFDA: Drug safety information and warnings
    • RxNorm (NLM): Standardized drug names and alternatives  
    • DrugBank: Comprehensive drug information (when available)

IMPORTANT DISCLAIMERS:
    • This plan is AI-generated for informational purposes only
    • All recommendations require validation by licensed healthcare providers
    • Medication dosages and alternatives need professional review
    • This is NOT a substitute for professional medical advice

===============================================================================
For CURRENT patient status, see: ../Output/{patient_id}_current_status.txt                  
===============================================================================
"""
    
    # Save future plan file in Future_Suggestions directory
    future_plan_filename = f"{patient_id}_future_plan.txt"
    future_plan_path = os.path.join(future_suggestions_path, future_plan_filename)
    
    with open(future_plan_path, 'w', encoding='utf-8') as f:
        f.write(future_plan)
    
    # Return summary of what was created
    

def identify_user(identifier):
    """
    Find a patient based on their identifier (patient ID, SSN, etc.)
    """
    # Search by patient ID first
    patient_match = datasets["patients"][datasets["patients"]["patient"] == identifier]
    if not patient_match.empty:
        return patient_match.iloc[0].to_dict()
    
    # Search by SSN if available
    if "ssn" in datasets["patients"].columns:
        ssn_match = datasets["patients"][datasets["patients"]["ssn"] == identifier]
        if not ssn_match.empty:
            return ssn_match.iloc[0].to_dict()
    
    # Search by drivers license if available  
    if "drivers" in datasets["patients"].columns:
        drivers_match = datasets["patients"][datasets["patients"]["drivers"] == identifier]
        if not drivers_match.empty:
            return drivers_match.iloc[0].to_dict()
    
    # Search by passport if available
    if "passport" in datasets["patients"].columns:
        passport_match = datasets["patients"][datasets["patients"]["passport"] == identifier]
        if not passport_match.empty:
            return passport_match.iloc[0].to_dict()
    
    return None

def check_patient_death_status(patient_id):
    """
    Check if patient has a death certification in their encounters
    
    Args:
        patient_id: The patient's unique identifier
        
    Returns:
        tuple: (is_deceased, death_date, death_encounter_type)
    """
    # First check if there's a deathdate in the patients table
    if "deathdate" in datasets["patients"].columns:
        patient_row = datasets["patients"][datasets["patients"]["patient"] == patient_id]
        if not patient_row.empty:
            death_date = patient_row.iloc[0].get("deathdate")
            if pd.notna(death_date) and str(death_date).strip() != "":
                return True, death_date, "Patient record shows death date"
    
    # Then check encounters for death certification
    encounters = datasets["encounters"][datasets["encounters"]["PATIENT"] == patient_id]
    
    if encounters.empty:
        return False, None, None
    
    # Only look for very specific death certification encounters
    # Don't just look for "death" keyword as it could be in many contexts
    specific_death_phrases = [
        "death certification",
        "patient death",
        "patient expired",
        "patient deceased",
        "certificate of death",
        "declaration of death"
    ]
    
    for _, encounter in encounters.iterrows():
        encounter_description = str(encounter.get("DESCRIPTION", "")).lower()
        encounter_reason = str(encounter.get("REASONDESCRIPTION", "")).lower()
        
        # Check for specific death certification encounters
        for phrase in specific_death_phrases:
            if phrase in encounter_description:
                # Even if we find a death-related encounter, verify the date
                death_date = encounter.get("START", "Unknown")
                # Only mark as deceased if we have a real date, not "Unknown"
                if death_date != "Unknown" and pd.notna(death_date) and str(death_date).strip() != "":
                    return True, death_date, encounter_description
                else:
                    # Found death encounter but no valid date - patient may not actually be deceased
                    print(f"Warning: Found death-related encounter but no valid death date for patient {patient_id}")
    
    return False, None, None

def get_patient_history(patient_id):
    """
    Get medical history for a patient
    """
    history = {}
    
    # Get conditions
    history["conditions"] = datasets["conditions"][datasets["conditions"]["PATIENT"] == patient_id]
    
    # Get medications
    history["medications"] = datasets["medications"][datasets["medications"]["PATIENT"] == patient_id]
    
    # Get allergies
    history["allergies"] = datasets["allergies"][datasets["allergies"]["PATIENT"] == patient_id]
    
    # Get encounters
    history["encounters"] = datasets["encounters"][datasets["encounters"]["PATIENT"] == patient_id]
    
    # Get observations
    history["observations"] = datasets["observations"][datasets["observations"]["PATIENT"] == patient_id]
    
    return history

def generate_recommendations(patient_id):
    """
    Generate basic recommendations based on patient data (non-LLM)
    """
    recommendations = {
        "medications": [],
        "condition_plans": {},
        "preventive_care": [],
        "general_advice": []
    }
    
    # Get patient history
    history = get_patient_history(patient_id)
    
    # Basic preventive care recommendations
    recommendations["preventive_care"] = [
        "Annual physical examination",
        "Regular blood pressure monitoring",
        "Cholesterol screening every 5 years",
        "Age-appropriate cancer screenings"
    ]
    
    # General health advice
    recommendations["general_advice"] = [
        "Maintain a balanced diet rich in fruits and vegetables",
        "Exercise regularly - at least 150 minutes moderate activity per week",
        "Get adequate sleep (7-9 hours per night)",
        "Avoid smoking and limit alcohol consumption",
        "Stay hydrated by drinking plenty of water"
    ]
    
    # Add condition-specific recommendations if conditions exist
    if not history["conditions"].empty:
        for _, condition in history["conditions"].iterrows():
            condition_code = str(condition["CODE"])
            # Find related care plans
            related_plans = datasets["careplans"][datasets["careplans"]["PATIENT"] == patient_id]
            if not related_plans.empty:
                plan_codes = related_plans["CODE"].tolist()
                recommendations["condition_plans"][condition_code] = plan_codes
    
    return recommendations

# Phase 2: Medicine Database API Integration
def query_openfda_api(drug_name):
    """
    Query openFDA API for drug information
    """
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        base_url = "https://api.fda.gov/drug/label.json"
        params = {
            "search": f"openfda.brand_name:{drug_name} OR openfda.generic_name:{drug_name}",
            "limit": 5
        }
        
        print(f"Querying openFDA for: {drug_name}")
        response = requests.get(base_url, params=params, timeout=30, verify=False)  # Added verify=False for development
        response.raise_for_status()
        
        data = response.json()
        if data.get("results"):
            drug_info = {
                "source": "openFDA",
                "drug_name": drug_name,
                "brand_names": [],
                "generic_names": [],
                "indications": [],
                "warnings": [],
                "dosage_forms": []
            }
            
            for result in data["results"][:3]:  # Take first 3 results
                openfda = result.get("openfda", {})
                if openfda.get("brand_name"):
                    drug_info["brand_names"].extend(openfda["brand_name"])
                if openfda.get("generic_name"):
                    drug_info["generic_names"].extend(openfda["generic_name"])
                if result.get("indications_and_usage"):
                    drug_info["indications"].extend(result["indications_and_usage"])
                if result.get("warnings"):
                    drug_info["warnings"].extend(result["warnings"])
                if openfda.get("dosage_form"):
                    drug_info["dosage_forms"].extend(openfda["dosage_form"])
            
            # Remove duplicates
            for key in ["brand_names", "generic_names", "indications", "warnings", "dosage_forms"]:
                drug_info[key] = list(set(drug_info[key]))
            
            return drug_info
        else:
            print(f"No openFDA results found for: {drug_name}")
            return None
            
    except Exception as e:
        print(f"Error querying openFDA for {drug_name}: {e}")
        return None

def query_rxnorm_api(drug_name):
    """
    Query RxNorm API for standardized drug information
    """
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        base_url = "https://rxnav.nlm.nih.gov/REST/drugs.json"
        params = {"name": drug_name}
        
        print(f"Querying RxNorm for: {drug_name}")
        response = requests.get(base_url, params=params, timeout=30, verify=False)  # Added verify=False for development
        response.raise_for_status()
        
        data = response.json()
        if data.get("drugGroup", {}).get("conceptGroup"):
            drug_info = {
                "source": "RxNorm",
                "drug_name": drug_name,
                "rxcui": [],
                "synonyms": [],
                "related_drugs": []
            }
            
            for group in data["drugGroup"]["conceptGroup"]:
                if group.get("conceptProperties"):
                    for concept in group["conceptProperties"]:
                        if concept.get("rxcui"):
                            drug_info["rxcui"].append(concept.get("rxcui"))
                        if concept.get("name"):
                            drug_info["synonyms"].append(concept.get("name"))
            
            # Remove duplicates
            drug_info["rxcui"] = list(set(drug_info["rxcui"]))
            drug_info["synonyms"] = list(set(drug_info["synonyms"]))
            
            return drug_info
        else:
            print(f"No RxNorm results found for: {drug_name}")
            return None
            
    except Exception as e:
        print(f"Error querying RxNorm for {drug_name}: {e}")
        return None

def query_rapidapi_openfda(drug_name):
    """
    Query OpenFDA via RapidAPI for drug information (ndc_directory focus).
    Serves as a backup or alternative source.
    """
    if not RAPIDAPI_KEY:
        print("RapidAPI key not found. Skipping RapidAPI OpenFDA call.")
        return None

    url = "https://open-fda.p.rapidapi.com/drug/ndc_directory/search"
    # The ndc_directory endpoint typically searches on fields like generic_name or brand_name directly.
    # We might need to try one then the other if a combined search isn't directly supported.
    # For simplicity, let's try searching by the provided drug_name, assuming it could be generic or brand.
    # A more robust implementation might try specific fields or make two calls.
    querystring = {"search": f'generic_name:"{drug_name}" OR brand_name:"{drug_name}"', "limit": "5"} 
    # Alternative simple search: querystring = {"search": drug_name, "limit": "5"}
    # Or specific field: querystring = {"generic_name": drug_name, "limit": "5"} - but drug_name could be a brand name.

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "open-fda.p.rapidapi.com"
    }

    try:
        print(f"Querying RapidAPI OpenFDA (ndc_directory) for: {drug_name}")
        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            drug_info = {
                "source": "RapidAPI-OpenFDA-NDC",
                "drug_name": drug_name,
                "brand_names": set(),
                "generic_names": set(),
                "indications": [], # Likely not available or limited in ndc_directory
                "warnings": [],    # Likely not available or limited in ndc_directory
                "dosage_forms": set() # May or may not be present
            }
            for result in data["results"]:
                if result.get("brand_name"):
                    drug_info["brand_names"].add(result["brand_name"])
                if result.get("generic_name"):
                    drug_info["generic_names"].add(result["generic_name"])
                if result.get("dosage_form"):
                    drug_info["dosage_forms"].add(result["dosage_form"])
                # Add other fields if available and relevant, e.g., from 'packaging'

            drug_info["brand_names"] = sorted(list(drug_info["brand_names"]))
            drug_info["generic_names"] = sorted(list(drug_info["generic_names"]))
            drug_info["dosage_forms"] = sorted(list(drug_info["dosage_forms"]))
            
            # If after processing, we have no useful names, consider it not found via this method
            if not drug_info["brand_names"] and not drug_info["generic_names"]:
                print(f"No distinct drug names found via RapidAPI for {drug_name}")
                return None
            return drug_info
        else:
            print(f"No results from RapidAPI OpenFDA (ndc_directory) for: {drug_name}")
            return None
    except requests.exceptions.Timeout:
        print(f"Request to RapidAPI OpenFDA for {drug_name} timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error querying RapidAPI OpenFDA for {drug_name}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error in query_rapidapi_openfda for {drug_name}: {e}")
        return None

def query_drugbank_api(drug_name):
    """
    Placeholder for DrugBank API (requires API key and account)
    In a real implementation, this would query DrugBank's API
    """
    print(f"DrugBank API placeholder for: {drug_name}")
    # TODO: Implement actual DrugBank API call when credentials are available
    return {
        "source": "DrugBank_Placeholder",
        "drug_name": drug_name,
        "note": "DrugBank integration requires API credentials"
    }

def verify_and_enrich_medications(consolidated_suggestions):
    """
    Verify and enrich medication recommendations using external APIs
    """
    print("\n=== Phase 2: Medicine Database Verification ===")
    
    enriched_suggestions = consolidated_suggestions.copy()
    medication_recommendations = consolidated_suggestions.get("medication_recommendations", [])
    
    if not medication_recommendations:
        print("No medication recommendations to verify.")
        return enriched_suggestions
    
    verified_medications = []
    enhanced_recommendations = []
    
    # Filter out non-medications and improve drug name extraction
    for medication in medication_recommendations:
        print(f"\nAnalyzing: {medication}")
        
        # Skip obvious non-medications
        skip_keywords = {
            'diagnostic', 'imaging', 'biopsy', 'surgery', 'surgical', 'consultation', 
            'follow-up', 'monitoring', 'evaluation', 'assessment', 'procedure',
            'intervention', 'therapy:', 'care:', 'support:', 'management:', 'goals:',
            'timelines:', 'steps:', 'signs:', 'adjustments:', 'treatments:'
        }
        
        if any(skip_word in medication.lower() for skip_word in skip_keywords):
            print(f"  Skipping non-medication: {medication[:50]}...")
            continue
        
        # Improved drug name extraction
        potential_drug_name = None
        words = medication.split()
        
        # Look for patterns like "Drug name dose frequency"
        for i, word in enumerate(words):
            clean_word = word.strip('.,()').strip('*').strip(':')
            
            # Skip common medical words
            skip_words = {'consider', 'continue', 'start', 'stop', 'increase', 'decrease', 
                         'daily', 'twice', 'for', 'with', 'without', 'as', 'needed', 'take', 
                         'avoid', 'recommend', 'suggested', 'prescribed', 'administer'}
            
            if (len(clean_word) > 3 and 
                clean_word.lower() not in skip_words and
                not clean_word.endswith('mg') and
                not clean_word.endswith(':**') and
                not clean_word.isdigit()):
                
                # Look for drug-like patterns
                if (clean_word[0].isupper() and 
                    len(clean_word) > 4 and
                    not clean_word.startswith(('**', 'NEXT', 'FOLLOW', 'HEALTH', 'WARNING', 'LIFESTYLE'))):
                    potential_drug_name = clean_word
                    break
        
        if not potential_drug_name:
            print(f"  No valid drug name found in: {medication[:50]}...")
            continue
            
        print(f"  Extracted drug name: {potential_drug_name}")
        
        medication_data = {
            "original_recommendation": medication,
            "extracted_drug_name": potential_drug_name,
            "verification_data": {},
            "verified_alternatives": []
        }
        
        # Query multiple APIs for verification and alternatives
        openfda_data = query_openfda_api(potential_drug_name)
        if openfda_data:
            medication_data["verification_data"]["openFDA"] = openfda_data
            print(f"  ✅ openFDA: Found {len(openfda_data.get('brand_names', []))} brand names")
        else:
            print(f"  ⚠️ openFDA (direct) call failed or no info for {potential_drug_name}. Trying RapidAPI OpenFDA backup...")
            rapidapi_openfda_data = query_rapidapi_openfda(potential_drug_name)
            if rapidapi_openfda_data:
                medication_data["verification_data"]["RapidAPI_OpenFDA"] = rapidapi_openfda_data # Store under a different key
                print(f"  ✅ RapidAPI OpenFDA: Found {len(rapidapi_openfda_data.get('brand_names', []))} brand names, {len(rapidapi_openfda_data.get('generic_names', []))} generic names")
                # Merge or prioritize data if needed, for now, just adding it
                # For the enhanced_rec, we might want to use this data if original openfda_data was None
                if not openfda_data and rapidapi_openfda_data.get('warnings'): # If this new source has warnings (unlikely for ndc_dir)
                     enhanced_rec = f" [FDA Warning (RapidAPI): {rapidapi_openfda_data['warnings'][0][:50]}...]" if rapidapi_openfda_data['warnings'] else ""
            else:
                print(f"  ❌ RapidAPI OpenFDA backup also failed or no info for {potential_drug_name}.")
        
        rxnorm_data = query_rxnorm_api(potential_drug_name)
        if rxnorm_data:
            medication_data["verification_data"]["RxNorm"] = rxnorm_data
            print(f"  ✅ RxNorm: Found {len(rxnorm_data.get('synonyms', []))} alternatives")
            
            # Use RxNorm data to suggest specific alternatives
            synonyms = rxnorm_data.get('synonyms', [])[:3]  # Take first 3 alternatives
            for synonym in synonyms:
                if synonym and synonym.lower() != potential_drug_name.lower():
                    medication_data["verified_alternatives"].append(synonym)
        
        drugbank_data = query_drugbank_api(potential_drug_name)
        if drugbank_data:
            medication_data["verification_data"]["DrugBank"] = drugbank_data
        
        verified_medications.append(medication_data)
        
        # Create enhanced recommendation with verified data
        enhanced_rec = medication
        if medication_data["verified_alternatives"]:
            alternatives = ", ".join(medication_data["verified_alternatives"][:2])
            enhanced_rec += f" (Verified alternatives: {alternatives})"
        
        # Prioritize warnings from direct openFDA if available, then RapidAPI (though ndc_directory may not have them)
        fda_warnings_to_display = None
        if openfda_data and openfda_data.get('warnings'):
            fda_warnings_to_display = openfda_data['warnings']
        elif not openfda_data and medication_data.get("verification_data", {}).get("RapidAPI_OpenFDA", {}).get('warnings'):
            # This path is less likely as ndc_directory usually doesn't have extensive warning sections like label data.
            fda_warnings_to_display = medication_data["verification_data"]["RapidAPI_OpenFDA"]['warnings']

        if fda_warnings_to_display:
            enhanced_rec += f" [FDA Warning: {fda_warnings_to_display[0][:50]}...]" if fda_warnings_to_display else ""
        
        enhanced_recommendations.append(enhanced_rec)
    
    # Update the enriched suggestions with verified and enhanced data
    enriched_suggestions["medication_recommendations"] = enhanced_recommendations
    enriched_suggestions["verified_medications"] = verified_medications
    enriched_suggestions["verification_summary"] = {
        "total_processed": len(medication_recommendations),
        "actual_medications": len(verified_medications),
        "successfully_verified": len([m for m in verified_medications if m["verification_data"]]),
        "alternatives_found": sum(len(m["verified_alternatives"]) for m in verified_medications),
        "sources_used": ["openFDA", "RxNorm", "DrugBank_Placeholder"]
    }
    
    print(f"\nVerification Summary:")
    print(f"  Original items: {enriched_suggestions['verification_summary']['total_processed']}")
    print(f"  Actual medications: {enriched_suggestions['verification_summary']['actual_medications']}")
    print(f"  Successfully verified: {enriched_suggestions['verification_summary']['successfully_verified']}")
    print(f"  Alternatives found: {enriched_suggestions['verification_summary']['alternatives_found']}")
    
    return enriched_suggestions

def find_similar_patients(target_patient_id, patient_features_df, top_k=5):
    """
    Find the top k most similar patients to the target patient using cosine similarity.
    Excludes deceased patients from results.
    
    Args:
        target_patient_id: ID of the target patient
        patient_features_df: DataFrame with patient features
        top_k: Number of similar patients to return
    
    Returns:
        List of tuples (patient_id, similarity_score)
    """
    # Ensure target patient exists
    if target_patient_id not in patient_features_df.index:
        print(f"Patient {target_patient_id} not found in patient features")
        return []
    
    # Get feature columns (excluding patient ID)
    feature_cols = ['age', 'gender_encoded', 'condition_count', 'medication_count', 
                   'avg_med_duration', 'allergy_count', 'careplan_count', 'adherent']
    
    # Prepare features matrix
    features_matrix = patient_features_df[feature_cols].fillna(0).values
    
    # Standardize features for better similarity calculation
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_matrix)
    
    # Find target patient index
    patient_ids = patient_features_df.index.tolist()
    target_idx = patient_ids.index(target_patient_id)
    
    # Calculate cosine similarity
    target_features = features_scaled[target_idx].reshape(1, -1)
    similarities = cosine_similarity(target_features, features_scaled)[0]
    
    # Get top k similar patients (excluding the target patient itself and deceased patients)
    similar_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    
    similar_patients = []
    checked_count = 0
    max_checks = top_k * 3  # Check up to 3x the requested number to account for deceased patients
    
    for idx in similar_indices:
        if idx != target_idx and len(similar_patients) < top_k and checked_count < max_checks:
            patient_id = patient_ids[idx]
            checked_count += 1
            
            # Check if patient is deceased
            is_deceased, _, _ = check_patient_death_status(patient_id)
            if not is_deceased:
                similarity_score = similarities[idx]
                similar_patients.append((patient_id, similarity_score))
                print(f"  ✓ Added similar patient: {patient_id} (similarity: {similarity_score:.3f})")
            else:
                print(f"  ✗ Skipped deceased patient: {patient_id}")
    
    return similar_patients

def generate_current_status_for_patient(patient_id):
    """
    Generate current status text file for a given patient.
    Returns the content as a string.
    """
    # Get patient basic info
    patient_info = patient_df.loc[patient_id]
    
    # Get patient history
    history = get_patient_history(patient_id)
    
    # Calculate age
    birthdate = pd.to_datetime(patient_info["birthdate"])
    age = (datetime.now() - birthdate).days // 365
    
    # Build current status text
    status_text = f"CURRENT STATUS - Patient ID: {patient_id}\n"
    status_text += "="*60 + "\n\n"
    
    status_text += f"Name: {patient_info['first']} {patient_info['last']}\n"
    status_text += f"Age: {age} years\n"
    status_text += f"Gender: {patient_info['gender']}\n"
    status_text += f"Address: {patient_info['address']}\n\n"
    
    # Add conditions
    status_text += "CURRENT CONDITIONS:\n"
    if not history["conditions"].empty:
        for _, condition in history["conditions"].iterrows():
            status_text += f"• {condition['DESCRIPTION']} (since {condition['START']})\n"
    else:
        status_text += "• None\n"
    status_text += "\n"
    
    # Add medications
    status_text += "CURRENT MEDICATIONS:\n"
    if not history["medications"].empty:
        active_meds = history["medications"][history["medications"]["STOP"].isna()]
        if not active_meds.empty:
            for _, med in active_meds.iterrows():
                status_text += f"• {med['DESCRIPTION']} (started {med['START']})\n"
        else:
            status_text += "• None currently active\n"
    else:
        status_text += "• None\n"
    status_text += "\n"
    
    # Add allergies
    status_text += "ALLERGIES:\n"
    if not history["allergies"].empty:
        for _, allergy in history["allergies"].iterrows():
            status_text += f"• {allergy['DESCRIPTION']}\n"
    else:
        status_text += "• None\n"
    status_text += "\n"
    
    # Add recent observations
    status_text += "RECENT OBSERVATIONS:\n"
    if not history["observations"].empty:
        recent_obs = history["observations"].sort_values("DATE", ascending=False).head(10)
        for _, obs in recent_obs.iterrows():
            status_text += f"• {obs['DESCRIPTION']}: {obs['VALUE']} {obs['UNITS']} ({obs['DATE']})\n"
    else:
        status_text += "• None\n"
    status_text += "\n"
    
    # Add encounters summary
    status_text += "RECENT ENCOUNTERS:\n"
    if not history["encounters"].empty:
        # Check which date column exists in encounters
        date_columns = ['START', 'Start', 'start', 'DATE', 'Date', 'date']
        encounter_date_col = None
        for col in date_columns:
            if col in history["encounters"].columns:
                encounter_date_col = col
                break
        
        if encounter_date_col:
            recent_enc = history["encounters"].sort_values(encounter_date_col, ascending=False).head(5)
            for _, enc in recent_enc.iterrows():
                desc = enc.get('DESCRIPTION', 'Unknown encounter')
                reason = enc.get('REASONDESCRIPTION', 'No reason specified')
                date = enc.get(encounter_date_col, 'Unknown date')
                status_text += f"• {desc} - {reason} ({date})\n"
        else:
            # If no date column found, just show first 5 encounters
            for idx, (_, enc) in enumerate(history["encounters"].head(5).iterrows()):
                desc = enc.get('DESCRIPTION', 'Unknown encounter')
                reason = enc.get('REASONDESCRIPTION', 'No reason specified')
                status_text += f"• {desc} - {reason}\n"
    else:
        status_text += "• None\n"
    
    return status_text

def main():
    """Main function for the recommendation system"""
    print("Healthcare Recommendation System Starting...\n")
    
    # Check if patient ID was provided as command line argument
    if len(sys.argv) > 1:
        patient_identifier = sys.argv[1].strip()
        print(f"Using command line patient ID: {patient_identifier}")
    else:
        # Get patient identifier from user input
        patient_identifier = input("Enter patient name or ID: ").strip()
    
    if not patient_identifier:
        print("No patient identifier provided. Exiting.")
        return
    
    # Identify patient
    patient_data = identify_user(patient_identifier)
    
    if patient_data is None:
        print("Patient not found. Please check the identifier.")
        return
    
    patient_id = patient_data['patient']
    patient_name = f"{patient_data['first']} {patient_data['last']}"
    
    print(f"\nPatient found: {patient_name} (ID: {patient_id})")
    
    # Check if patient is alive
    is_deceased, death_date, death_encounter = check_patient_death_status(patient_id)
    
    if is_deceased:
        print(f"Patient {patient_name} is marked as deceased (Death date: {death_date}). Cannot generate future health plan.")
        return
    
    # Get patient history
    print("\nFetching patient history...")
    history = get_patient_history(patient_id)
    
    # Generate current status for target patient
    print("\nGenerating current status for target patient...")
    target_patient_status = generate_current_status_for_patient(patient_id)
    
    # Save target patient current status
    target_status_filename = f"{output_path}Current_Status_{patient_name.replace(' ', '_')}_{patient_id}.txt"
    with open(target_status_filename, 'w', encoding='utf-8') as f:
        f.write(target_patient_status)
    print(f"Target patient status saved to: {target_status_filename}")
    
    # Find similar patients using cosine similarity (now excludes deceased patients)
    print("\nFinding similar patients (excluding deceased)...")
    similar_patients = find_similar_patients(patient_id, patient_features, top_k=5)
    
    if similar_patients:
        print(f"\nFound {len(similar_patients)} similar living patients:")
        for similar_id, similarity_score in similar_patients:
            similar_patient_info = patient_df.loc[similar_id]
            print(f"  - {similar_patient_info['first']} {similar_patient_info['last']} (ID: {similar_id}, Similarity: {similarity_score:.3f})")
        
        # Generate current status for all similar patients
        print("\nGenerating current status for similar patients...")
        similar_patient_statuses = []
        
        for similar_id, _ in similar_patients:
            print(f"  Generating status for patient {similar_id}...")
            similar_status = generate_current_status_for_patient(similar_id)
            similar_patient_statuses.append((similar_id, similar_status))
            
            # Save similar patient status
            similar_patient_info = patient_df.loc[similar_id]
            similar_name = f"{similar_patient_info['first']}_{similar_patient_info['last']}"
            similar_status_filename = f"{output_path}Similar_Patient_Status_{similar_name}_{similar_id}.txt"
            with open(similar_status_filename, 'w', encoding='utf-8') as f:
                f.write(similar_status)
            print(f"    Status saved to: {similar_status_filename}")
    else:
        print("\nNo similar living patients found in the database.")
        similar_patient_statuses = []
    
    # Generate recommendations using multi-LLM approach with similar patient data
    print("\nGenerating multi-LLM recommendations with similar patient context...")
    recommendations = generate_recommendations_with_similar_patients(
        patient_id, 
        target_patient_status, 
        similar_patient_statuses
    )
    
    # Save recommendations
    if recommendations:
        # Save as JSON
        recommendations_file = f"{output_path}Recommendations_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(recommendations_file, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=4)
        print(f"Recommendations saved to: {recommendations_file}")
        
        # Generate future health plan
        print("\nGenerating future health plan...")
        future_plan = generate_future_health_plan(patient_data, history, recommendations)
        
        # Save future plan
        plan_file = f"{future_suggestions_path}Future_Health_Plan_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(plan_file, 'w', encoding='utf-8') as f:
            f.write(future_plan)
        print(f"Future health plan saved to: {plan_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Patient: {patient_name}")
        print(f"Similar patients analyzed: {len(similar_patient_statuses)}")
        print(f"Current status saved: {target_status_filename}")
        print(f"Recommendations saved: {recommendations_file}")
        print(f"Future health plan saved: {plan_file}")
    else:
        print("Failed to generate recommendations.")

def generate_recommendations_with_similar_patients(patient_id, target_patient_status, similar_patient_statuses):
    """
    Generate recommendations using multi-LLM approach with similar patient context.
    
    Args:
        patient_id: Target patient ID
        target_patient_status: Current status text of target patient
        similar_patient_statuses: List of tuples (patient_id, status_text) for similar patients
    
    Returns:
        Consolidated recommendations dictionary
    """
    # Create enhanced prompt with similar patient context
    context_prompt = f"""You are an expert healthcare AI assistant. You need to analyze a target patient and similar patients to provide personalized health recommendations.

TARGET PATIENT (Patient to create recommendations for):
{target_patient_status}

SIMILAR PATIENTS (Case studies for reference):
"""
    
    for idx, (similar_id, similar_status) in enumerate(similar_patient_statuses, 1):
        context_prompt += f"\n\nSIMILAR PATIENT #{idx}:\n{similar_status}\n"
    
    context_prompt += """

Based on the target patient's current status and the treatment patterns observed in similar patients, provide comprehensive health recommendations. Consider:
1. What treatments worked well for similar patients with similar conditions
2. Medication patterns that show good outcomes
3. Common complications to watch for based on similar cases
4. Preventive measures that similar patients benefited from

Provide recommendations in the following categories:
- Medication Recommendations (including alternatives)
- Surgical Recommendations (if applicable)
- Other Treatments (therapies, lifestyle changes)
- Next Steps
- Follow-up Timeline
- Health Goals
- Lifestyle Adjustments
- Warning Signs to Monitor

Be specific and evidence-based, drawing insights from the similar patient cases provided."""

    # Call multiple LLMs with the enhanced prompt
    print("\nCalling multiple LLMs for recommendations...")
    all_suggestions = []
    
    # Debug: Save prompt to file for inspection
    debug_prompt_file = f"debug_prompt_{patient_id}.txt"
    with open(debug_prompt_file, 'w', encoding='utf-8') as f:
        f.write(context_prompt)
    print(f"DEBUG: Prompt saved to {debug_prompt_file}")
    
    # OpenAI
    openai_response = call_openai_api(context_prompt)
    if openai_response:
        print("✓ OpenAI response received")
        # Debug: Save raw response
        with open(f"debug_openai_response_{patient_id}.txt", 'w', encoding='utf-8') as f:
            f.write(openai_response)
        print(f"DEBUG: OpenAI raw response saved to debug_openai_response_{patient_id}.txt")
        
        processed_openai = process_llm_response(openai_response)
        print(f"DEBUG: OpenAI processed response: {json.dumps(processed_openai, indent=2)[:500]}...")
        all_suggestions.append(processed_openai)
    else:
        print("✗ OpenAI response failed or empty")
    
    # Anthropic
    anthropic_response = call_anthropic_api(context_prompt)
    if anthropic_response:
        print("✓ Anthropic response received")
        # Debug: Save raw response
        with open(f"debug_anthropic_response_{patient_id}.txt", 'w', encoding='utf-8') as f:
            f.write(anthropic_response)
        print(f"DEBUG: Anthropic raw response saved to debug_anthropic_response_{patient_id}.txt")
        
        processed_anthropic = process_llm_response(anthropic_response)
        all_suggestions.append(processed_anthropic)
    else:
        print("✗ Anthropic response failed or empty")
    
    # Perplexity
    perplexity_response = call_perplexity_api(context_prompt)
    if perplexity_response:
        print("✓ Perplexity response received")
        # Debug: Save raw response
        with open(f"debug_perplexity_response_{patient_id}.txt", 'w', encoding='utf-8') as f:
            f.write(perplexity_response)
        print(f"DEBUG: Perplexity raw response saved to debug_perplexity_response_{patient_id}.txt")
        
        processed_perplexity = process_llm_response(perplexity_response)
        all_suggestions.append(processed_perplexity)
    else:
        print("✗ Perplexity response failed or empty")
    
    # Gemini
    gemini_response = call_gemini_api(context_prompt)
    if gemini_response:
        print("✓ Gemini response received")
        # Debug: Save raw response
        with open(f"debug_gemini_response_{patient_id}.txt", 'w', encoding='utf-8') as f:
            f.write(gemini_response)
        print(f"DEBUG: Gemini raw response saved to debug_gemini_response_{patient_id}.txt")
        
        processed_gemini = process_llm_response(gemini_response)
        print(f"DEBUG: Gemini processed response: {json.dumps(processed_gemini, indent=2)[:500]}...")
        all_suggestions.append(processed_gemini)
    else:
        print("✗ Gemini response failed or empty")
    
    print(f"\nDEBUG: Total valid LLM responses: {len(all_suggestions)}")
    
    # Consolidate and deduplicate
    print("\nConsolidating recommendations from all LLMs...")
    consolidated = consolidate_and_deduplicate_suggestions(all_suggestions)
    
    print(f"DEBUG: Consolidated recommendations: {json.dumps(consolidated, indent=2)[:500]}...")
    
    # Verify and enrich medications (if applicable)
    print("\nVerifying and enriching medication recommendations...")
    enriched_suggestions = verify_and_enrich_medications(consolidated)
    
    print(f"DEBUG: Final enriched suggestions: {json.dumps(enriched_suggestions, indent=2)[:500]}...")
    
    return enriched_suggestions

if __name__ == "__main__":
    main()
