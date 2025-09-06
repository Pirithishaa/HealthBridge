# HealthBridge Authentication Setup

This guide will help you set up authentication for both NGO and Medical Provider users with MySQL database storage.

## Prerequisites

1. **MySQL Server** - Make sure MySQL is installed and running on your system
2. **Python 3.7+** - Required for running the Flask application
3. **pip** - Python package installer

## Installation Steps

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

### 2. Set Up MySQL Database

#### Option A: Using the Setup Script (Recommended)

Run the interactive database setup script:

```bash
python setup_database.py
```

This script will:
- Connect to your MySQL server
- Create the `healthbridge_db` database
- Create the required tables (`ngo_users` and `medical_providers`)
- Generate a configuration file with your database settings

#### Option B: Manual Setup

1. **Create Database:**
   ```sql
   CREATE DATABASE healthbridge_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

2. **Create Tables:**
   ```sql
   USE healthbridge_db;
   
   CREATE TABLE ngo_users (
       id INT AUTO_INCREMENT PRIMARY KEY,
       full_name VARCHAR(255) NOT NULL,
       email VARCHAR(255) UNIQUE NOT NULL,
       password_hash VARCHAR(255) NOT NULL,
       designation ENUM('program-manager', 'director', 'volunteer') NOT NULL,
       organization VARCHAR(255) NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
       is_active BOOLEAN DEFAULT TRUE
   );
   
   CREATE TABLE medical_providers (
       id INT AUTO_INCREMENT PRIMARY KEY,
       full_name VARCHAR(255) NOT NULL,
       email VARCHAR(255) UNIQUE NOT NULL,
       medical_id VARCHAR(50) UNIQUE,
       password_hash VARCHAR(255) NOT NULL,
       specialization VARCHAR(255),
       hospital_name VARCHAR(255),
       license_number VARCHAR(100),
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
       is_active BOOLEAN DEFAULT TRUE
   );
   ```

### 3. Configure Database Connection

Update the database configuration in `database.py`:

```python
self.config = {
    'host': 'localhost',        # Your MySQL host
    'user': 'your_username',    # Your MySQL username
    'password': 'your_password', # Your MySQL password
    'database': 'healthbridge_db',
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}
```

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Authentication Features

### NGO Authentication
- **Signup**: `/ngo/signup` - Register new NGO users
- **Login**: `/ngo/login` - Authenticate NGO users
- **Dashboard**: `/ngo/dashboard` - Protected NGO dashboard

### Medical Provider Authentication
- **Signup**: `/provider/signup` - Register new medical providers
- **Login**: `/provider/login` - Authenticate medical providers
- **Dashboard**: `/provider/dashboard` - Protected provider dashboard

### Security Features
- Password hashing using Werkzeug
- Session management
- Protected routes with login requirements
- Input validation and sanitization

## User Registration Fields

### NGO Users
- Full Name
- Email Address
- Password (minimum 8 characters)
- Designation (Program Manager, Director, Volunteer)
- Organization Name

### Medical Providers
- Full Name
- Email Address
- Medical ID (unique identifier)
- Password (minimum 8 characters)
- Specialization (optional)
- Hospital/Clinic Name (optional)
- License Number (optional)

## Testing the Authentication

1. **Start the application**: `python app.py`
2. **Access the landing page**: `http://localhost:5000`
3. **Test NGO signup**: Click "SignUp" → "NGO" → Fill out the form
4. **Test NGO login**: Click "Login" → "NGO" → Use registered credentials
5. **Test Provider signup**: Click "SignUp" → "Medical Providers" → Fill out the form
6. **Test Provider login**: Click "Login" → "Medical Providers" → Use registered credentials

## Troubleshooting

### Common Issues

1. **MySQL Connection Error**
   - Ensure MySQL server is running
   - Check username and password
   - Verify database exists

2. **Import Errors**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

3. **Permission Errors**
   - Ensure MySQL user has CREATE, INSERT, SELECT, UPDATE permissions
   - Check file permissions for the application directory

### Database Connection Test

You can test your database connection by running:

```python
from database import db_manager
if db_manager.connect():
    print("Database connection successful!")
    db_manager.disconnect()
else:
    print("Database connection failed!")
```

## File Structure

```
HealthBridge/FRONTEND/
├── app.py                 # Main Flask application with authentication routes
├── database.py           # Database connection and authentication logic
├── setup_database.py     # Database setup script
├── requirements.txt      # Python dependencies
├── templates/
│   ├── landingPage.html  # Landing page with login/signup links
│   ├── ngosignup.html    # NGO registration form
│   ├── ngologin.html     # NGO login form
│   ├── providersignup.html # Medical provider registration form
│   ├── Provider Login.html # Medical provider login form
│   ├── ngo_dashboard.html # NGO dashboard (protected)
│   └── provider_dashboard.html # Provider dashboard (protected)
└── README_AUTHENTICATION.md # This file
```

## Security Notes

- Change the `secret_key` in `app.py` for production use
- Use environment variables for database credentials in production
- Enable HTTPS in production
- Regularly update dependencies
- Implement rate limiting for login attempts
- Add password complexity requirements if needed

## Support

If you encounter any issues:
1. Check the console output for error messages
2. Verify MySQL server is running and accessible
3. Ensure all required packages are installed
4. Check database permissions and configuration
