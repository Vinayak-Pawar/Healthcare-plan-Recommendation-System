# 🏥 Healthcare Recommendation System - Flask UI

A modern, responsive web interface for the AI-powered Healthcare Recommendation System that provides personalized medical insights and treatment plans.

## ✨ Features

### 🎯 Core Functionality
- **Patient Search**: Search patients by name or ID with real-time results
- **AI-Powered Recommendations**: Multi-AI integration (OpenAI, Gemini, Anthropic)
- **Interactive Dashboard**: Real-time statistics and system overview
- **Report Generation**: Downloadable medical reports and health plans
- **Safety Features**: Automatic deceased patient detection

### 🎨 Modern UI Design
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Medical Theme**: Professional healthcare color scheme
- **Interactive Elements**: Smooth animations and loading states
- **Tabbed Interface**: Organized results display
- **Real-time Search**: Live patient search with debounced input

### 🔒 Professional Features
- **Error Handling**: Comprehensive error messages and validation
- **File Management**: Secure file serving and downloads
- **API Integration**: RESTful API endpoints for all functionality
- **Loading States**: Visual feedback for all operations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- All dependencies from `requirements.txt`
- Healthcare recommendation system backend (`r_engine.py`)

### Installation

1. **Navigate to UI directory**:
   ```bash
   cd UI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your browser and go to: `http://localhost:5000`

## 📱 Usage Guide

### 1. Dashboard Overview
- View system statistics (patients, conditions, medications)
- Monitor generated reports count
- Get real-time system status

### 2. Patient Search
- **By Name**: Type patient's first or last name
- **By ID**: Enter complete patient UUID
- **Live Results**: See matching patients instantly
- **Patient Status**: Deceased patients are clearly marked

### 3. Health Plan Generation
- Select a patient from search results OR enter patient ID directly
- Click "Generate Health Plan" button
- Wait for AI processing (typically 30-60 seconds)
- View results in organized tabs

### 4. Results Analysis
- **Current Status Tab**: Patient's current health overview
- **Future Health Plan Tab**: AI-generated treatment recommendations
- **AI Recommendations Tab**: Detailed medication and treatment suggestions
- **Download Options**: Save all reports as files

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Main dashboard page
- `POST /generate_plan` - Generate health plan for patient
- `GET /search_patients` - Search patients by query
- `GET /api/stats` - Get system statistics
- `GET /files/<directory>/<filename>` - Serve generated files

### Example API Usage

**Search Patients**:
```bash
curl "http://localhost:5000/search_patients?q=John"
```

**Generate Health Plan**:
```bash
curl -X POST "http://localhost:5000/generate_plan" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "8a79e149-26a7-4200-b4fe-e8165e7b67e0"}'
```

## 🧪 Sample Data

The system includes sample patient IDs for testing:
- **Ivory Kertzmann**: `8a79e149-26a7-4200-b4fe-e8165e7b67e0`
- **Refugio Barton**: `6ace7b81-a04c-4424-9715-4168ab3e007c`
- **Felix Jaskolski**: `f3f498e2-eadf-4fcf-9ecb-6462c9f742d7`
- **Hobert Reynolds**: `802e4305-78e0-423b-9604-69b5d2f9a759`

## 📁 Project Structure

```
UI/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Main dashboard template
└── static/
    ├── css/
    │   └── style.css     # Modern CSS styling
    └── js/
        └── main.js       # Interactive JavaScript
```

## 🎨 Technology Stack

### Backend
- **Flask**: Web framework
- **Python**: Core backend logic
- **Pandas**: Data processing
- **Integration**: Full r_engine.py backend integration

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with CSS Grid and Flexbox
- **JavaScript ES6+**: Interactive functionality
- **Responsive Design**: Mobile-first approach

### AI Integration
- **OpenAI GPT**: Medical analysis
- **Google Gemini**: Comprehensive insights
- **Anthropic Claude**: Safety validation
- **Multi-AI Consensus**: Improved recommendation accuracy

## 🔒 Security Features

- **Input Validation**: All user inputs are validated
- **Error Handling**: Comprehensive error catching and reporting
- **File Security**: Secure file serving with proper headers
- **Patient Privacy**: Deceased patient detection and handling

## 📊 Performance

- **Real-time Search**: Debounced search with 300ms delay
- **Lazy Loading**: Statistics loaded asynchronously
- **Responsive Design**: Optimized for all screen sizes
- **Modern Web Standards**: Uses latest HTML5, CSS3, and ES6+ features

## 🐛 Troubleshooting

### Common Issues

1. **"Recommendation engine not available"**
   - Ensure `r_engine.py` is in the parent directory
   - Check that all required dependencies are installed

2. **Search not working**
   - Verify patient database files exist in `Database_tables/`
   - Check file permissions

3. **AI generation fails**
   - Verify API keys in `secrets.yaml`
   - Check internet connection for AI API calls

### Debug Mode
The application runs in debug mode by default for development. To disable:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes only. Not intended as a substitute for professional medical advice.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console logs for error messages
3. Ensure all dependencies are properly installed
4. Verify the backend `r_engine.py` is functioning correctly

---

**🏥 Healthcare Recommendation System - Making AI-powered medical insights accessible through modern web technology.** 