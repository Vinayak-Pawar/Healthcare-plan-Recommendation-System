import os
import sys
import json
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for, flash

# Add the parent directory to sys.path to allow importing r_engine
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
    import r_engine
    print("✅ Successfully imported r_engine")
except ImportError as e:
    print(f"❌ Error importing r_engine: {e}")
    print(f"Ensure r_engine.py is in the directory: {parent_dir}")
    r_engine = None

app = Flask(__name__)
app.secret_key = 'healthcare_recommendation_system_2024'  # For flash messages

# Configuration for output directories
OUTPUT_PATH = os.path.join(parent_dir, "Output")
FUTURE_SUGGESTIONS_PATH = os.path.join(parent_dir, "Future_Suggestions")

# Ensure directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FUTURE_SUGGESTIONS_PATH, exist_ok=True)

@app.route('/')
def home():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/patient/<patient_id>')
def patient_details(patient_id):
    """Patient details page"""
    if not r_engine:
        flash("❌ Recommendation engine not available", "error")
        return redirect(url_for('home'))
    
    try:
        # Get patient record
        patient_record = r_engine.identify_user(patient_id)
        if not patient_record:
            flash(f"❌ Patient not found: {patient_id}", "error")
            return redirect(url_for('home'))
        
        # Check if patient is deceased
        is_deceased, death_date, death_encounter = r_engine.check_patient_death_status(patient_id)
        
        # Get patient history
        history = r_engine.get_patient_history(patient_id)
        
        return render_template('patient_details.html', 
                             patient=patient_record, 
                             history=history,
                             is_deceased=is_deceased,
                             death_date=death_date)
    
    except Exception as e:
        flash(f"❌ Error loading patient details: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    """Generate health plan for a patient"""
    if not r_engine:
        return jsonify({"error": "Recommendation engine not available"}), 500
    
    data = request.get_json()
    patient_id = data.get('patient_id')
    
    if not patient_id:
        return jsonify({"error": "Patient ID is required"}), 400
    
    try:
        # Identify patient
        patient_record = r_engine.identify_user(patient_id)
        if not patient_record:
            return jsonify({"error": f"Patient not found: {patient_id}"}), 404
        
        # Check death status
        is_deceased, death_date, death_encounter = r_engine.check_patient_death_status(patient_id)
        if is_deceased:
            return jsonify({
                "error": "Cannot generate treatment plan for deceased patient",
                "patient_status": "deceased",
                "death_date": death_date
            }), 400
        
        # Generate recommendations
        recommendations = r_engine.generate_recommendations(patient_id)
        
        # Create personalized plan
        summary_message = r_engine.create_personalized_plan(patient_record, recommendations)
        
        # Prepare response with file paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        current_status_file = f"Current_Status_{patient_record['first']}_{patient_record['last']}_{patient_id}.txt"
        future_plan_file = f"Future_Health_Plan_{patient_record['first']}_{patient_record['last']}_{timestamp}.txt"
        recommendations_file = f"Recommendations_{patient_record['first']}_{patient_record['last']}_{timestamp}.json"
        
        # Check if files exist and get content
        current_status_content = ""
        future_plan_content = ""
        recommendations_content = {}
        
        current_status_path = os.path.join(OUTPUT_PATH, current_status_file)
        if os.path.exists(current_status_path):
            with open(current_status_path, 'r', encoding='utf-8') as f:
                current_status_content = f.read()
        
        future_plan_path = os.path.join(FUTURE_SUGGESTIONS_PATH, future_plan_file)
        if os.path.exists(future_plan_path):
            with open(future_plan_path, 'r', encoding='utf-8') as f:
                future_plan_content = f.read()
        
        # Find the most recent recommendations file
        recommendations_files = [f for f in os.listdir(OUTPUT_PATH) 
                               if f.startswith(f"Recommendations_{patient_record['first']}_{patient_record['last']}") 
                               and f.endswith('.json')]
        
        if recommendations_files:
            latest_rec_file = sorted(recommendations_files)[-1]
            rec_path = os.path.join(OUTPUT_PATH, latest_rec_file)
            try:
                with open(rec_path, 'r', encoding='utf-8') as f:
                    recommendations_content = json.load(f)
            except:
                recommendations_content = {"error": "Could not parse recommendations file"}
        
        return jsonify({
            "success": True,
            "patient": patient_record,
            "current_status": current_status_content,
            "future_plan": future_plan_content,
            "recommendations": recommendations_content,
            "files": {
                "current_status": current_status_file,
                "future_plan": future_plan_file,
                "recommendations": latest_rec_file if recommendations_files else None
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/run_engine', methods=['POST'])
def run_engine():
    """Execute the r_engine.py program workflow"""
    data = request.get_json()
    patient_id = data.get('patient_id')
    
    if not patient_id:
        return jsonify({"error": "Patient ID is required"}), 400
    
    try:
        # Check if r_engine module is available
        if not r_engine:
            return jsonify({"error": "Recommendation engine module not available"}), 500
        
        # Step 1: Identify patient
        patient_data = r_engine.identify_user(patient_id)
        if patient_data is None:
            return jsonify({"error": f"Patient not found: {patient_id}"}), 404
        
        patient_name = f"{patient_data['first']} {patient_data['last']}"
        
        # Step 2: Check if patient is alive
        is_deceased, death_date, death_encounter = r_engine.check_patient_death_status(patient_id)
        if is_deceased:
            return jsonify({
                "error": f"Patient {patient_name} is marked as deceased (Death date: {death_date}). Cannot generate future health plan.",
                "patient_status": "deceased",
                "death_date": death_date
            }), 400
        
        # Step 3: Get patient history
        history = r_engine.get_patient_history(patient_id)
        
        # Step 4: Generate current status
        target_patient_status = r_engine.generate_current_status_for_patient(patient_id)
        
        # Step 5: Save target patient current status
        target_status_filename = f"Current_Status_{patient_name.replace(' ', '_')}_{patient_id}.txt"
        target_status_path = os.path.join(OUTPUT_PATH, target_status_filename)
        with open(target_status_path, 'w', encoding='utf-8') as f:
            f.write(target_patient_status)
        
        # Step 6: Find similar patients
        similar_patients = r_engine.find_similar_patients(patient_id, r_engine.patient_features, top_k=5)
        
        similar_patient_statuses = []
        similar_patient_files = []
        if similar_patients:
            for similar_id, similarity_score in similar_patients:
                similar_status = r_engine.generate_current_status_for_patient(similar_id)
                similar_patient_statuses.append((similar_id, similar_status))
                
                # Save similar patient status
                similar_patient_info = r_engine.patient_df.loc[similar_id]
                similar_name = f"{similar_patient_info['first']}_{similar_patient_info['last']}"
                similar_status_filename = f"Similar_Patient_Status_{similar_name}_{similar_id}.txt"
                similar_status_path = os.path.join(OUTPUT_PATH, similar_status_filename)
                with open(similar_status_path, 'w', encoding='utf-8') as f:
                    f.write(similar_status)
                
                # Track the similar patient file info
                similar_patient_files.append({
                    "filename": similar_status_filename,
                    "patient_name": f"{similar_patient_info['first']} {similar_patient_info['last']}",
                    "patient_id": similar_id,
                    "similarity_score": similarity_score
                })
        
        # Step 7: Generate recommendations
        recommendations = r_engine.generate_recommendations_with_similar_patients(
            patient_id, 
            target_patient_status, 
            similar_patient_statuses
        )
        
        # Step 8: Save recommendations and generate future plan
        if recommendations:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save recommendations as JSON
            recommendations_file = f"Recommendations_{patient_name.replace(' ', '_')}_{timestamp}.json"
            recommendations_path = os.path.join(OUTPUT_PATH, recommendations_file)
            with open(recommendations_path, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, indent=4)
            
            # Generate future health plan
            future_plan = r_engine.generate_future_health_plan(patient_data, history, recommendations)
            
            # Save future plan
            plan_file = f"Future_Health_Plan_{patient_name.replace(' ', '_')}_{timestamp}.txt"
            plan_path = os.path.join(FUTURE_SUGGESTIONS_PATH, plan_file)
            with open(plan_path, 'w', encoding='utf-8') as f:
                f.write(future_plan)
            
            return jsonify({
                "success": True,
                "message": "Health plan generation completed successfully",
                "patient": {
                    "id": patient_id,
                    "name": patient_name
                },
                "similar_patients_count": len(similar_patient_statuses),
                "files": {
                    "current_status": target_status_filename,
                    "recommendations": recommendations_file,
                    "future_plan": plan_file
                },
                "current_status": target_patient_status[:1000] + "..." if len(target_patient_status) > 1000 else target_patient_status,
                "future_plan": future_plan,
                "recommendations": recommendations,
                "similar_patient_files": similar_patient_files
            })
        else:
            return jsonify({"error": "Failed to generate recommendations"}), 500
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in run_engine: {error_details}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/search_patients')
def search_patients():
    """Search for patients by name or ID"""
    if not r_engine:
        return jsonify({"error": "Recommendation engine not available"}), 500
    
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({"patients": []})
    
    try:
        # Load patient data
        import pandas as pd
        patients_df = pd.read_csv(os.path.join(parent_dir, "Database_tables", "patients.csv"))
        
        # Search by ID, first name, or last name
        results = patients_df[
            (patients_df['patient'].str.contains(query, case=False, na=False)) |
            (patients_df['first'].str.contains(query, case=False, na=False)) |
            (patients_df['last'].str.contains(query, case=False, na=False))
        ].head(10)  # Limit to 10 results
        
        patients = []
        for _, row in results.iterrows():
            # Check if patient is deceased
            is_deceased, _, _ = r_engine.check_patient_death_status(row['patient'])
            
            patients.append({
                "id": row['patient'],
                "name": f"{row['first']} {row['last']}",
                "birthdate": row['birthdate'],
                "gender": row['gender'],
                "city": row.get('city', ''),
                "state": row.get('state', ''),
                "is_deceased": is_deceased
            })
        
        return jsonify({"patients": patients})
    
    except Exception as e:
        return jsonify({"error": f"Search error: {str(e)}"}), 500

@app.route('/files/<path:directory>/<path:filename>')
def serve_file(directory, filename):
    """Serve generated files for download"""
    if directory == "Output":
        return send_from_directory(OUTPUT_PATH, filename, as_attachment=True)
    elif directory == "Future_Suggestions":
        return send_from_directory(FUTURE_SUGGESTIONS_PATH, filename, as_attachment=True)
    else:
        return "Invalid directory", 404

if __name__ == '__main__':
    print("🏥 Healthcare Recommendation System - Flask UI")
    print("=" * 50)
    
    if r_engine:
        print("✅ Recommendation engine loaded successfully")
        print(f"📁 Output directory: {OUTPUT_PATH}")
        print(f"📁 Future suggestions directory: {FUTURE_SUGGESTIONS_PATH}")
        print("🌐 Starting Flask application...")
        print("🔗 Access at: http://localhost:5001")
    else:
        print("❌ Warning: Recommendation engine not available")
    
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5001) 