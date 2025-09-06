# HealthBridge Authentication Setup Guide (app1.py)

This guide will help you set up authentication for your HealthBridge application using `app1.py` as the main file.

## ✅ **What's Been Added to app1.py**

Your `app1.py` now includes:
- **Complete authentication system** for both NGO and Medical Provider users
- **MySQL database integration** with secure password hashing
- **Session management** with Flask sessions
- **Protected routes** - your existing features now require login
- **User dashboards** with links to your existing functionality

## 🚀 **Quick Setup Steps**

### 1. Install Required Packages
```bash
pip install -r requirements.txt
```

### 2. Set Up MySQL Database
```bash
python setup_database.py
```
This will:
- Create the `healthbridge_db` database
- Create user tables for NGO and medical providers
- Generate configuration file

### 3. Update Database Configuration
Edit `database.py` and update the config dictionary:
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

### 4. Run Your Application
```bash
python app1.py
```

## 🔐 **Authentication Flow**

### **Landing Page** (`/` or `/landingPage.html`)
- Shows your existing landing page
- Login/Signup dropdowns for both user types

### **NGO Users**
- **Signup**: `/ngo/signup` → Register with name, email, password, designation, organization
- **Login**: `/ngo/login` → Login with email/password
- **Dashboard**: `/ngo/dashboard` → Protected NGO dashboard

### **Medical Providers**
- **Signup**: `/provider/signup` → Register with name, email, medical ID, password, etc.
- **Login**: `/provider/login` → Login with email/medical ID and password
- **Dashboard**: `/provider/dashboard` → Protected provider dashboard with links to your features

## 🛡️ **Protected Routes**

Your existing routes are now protected and require login:
- `/provider.html` - Your provider page
- `/medicalreportuplload.html` - Medical report upload
- `/dashboard_page` - Your analytics dashboard
- `/generate_report` - Report generation
- `/analyze_report` - Report analysis

## 📱 **User Experience Flow**

1. **User visits** `http://localhost:5000` → Landing page
2. **Clicks SignUp** → Chooses NGO or Medical Provider → Fills form → Account created
3. **Clicks Login** → Enters credentials → Authenticated → Redirected to dashboard
4. **From dashboard** → Can access all your existing features:
   - Medical Report Analysis
   - Patient Management  
   - Analytics Dashboard
   - NGO Collaboration

## 🔧 **Key Features Added**

### **Security**
- Password hashing with Werkzeug
- Session management
- Protected routes with `@login_required` decorator
- Input validation and sanitization

### **User Management**
- Separate user types (NGO vs Medical Provider)
- User-specific dashboards
- Session-based authentication
- Logout functionality

### **Database Integration**
- MySQL database with proper tables
- User registration and authentication
- Secure data storage

## 🎯 **Your Existing Features Integration**

All your existing functionality is preserved and now protected:

- **Medical Report Analysis** (`/medicalreportuplload.html`)
- **Patient Management** (`/provider.html`) 
- **Analytics Dashboard** (`/dashboard_page`)
- **Report Generation** (`/generate_report`)
- **Report Analysis** (`/analyze_report`)

## 🧪 **Testing the System**

1. **Start the app**: `python app1.py`
2. **Visit**: `http://localhost:5000`
3. **Test NGO signup**: SignUp → NGO → Fill form → Register
4. **Test NGO login**: Login → NGO → Use credentials → Access dashboard
5. **Test Provider signup**: SignUp → Medical Providers → Fill form → Register  
6. **Test Provider login**: Login → Medical Providers → Use credentials → Access dashboard
7. **Test protected features**: From dashboard, click on any feature card

## 🔍 **Troubleshooting**

### **Database Connection Issues**
- Ensure MySQL server is running
- Check username/password in `database.py`
- Verify database exists: `healthbridge_db`

### **Import Errors**
- Run: `pip install -r requirements.txt`
- Check Python version (3.7+ required)

### **Session Issues**
- Clear browser cookies/cache
- Check Flask secret key in `app1.py`

## 📁 **File Structure**

```
HealthBridge/FRONTEND/
├── app1.py                    # ✅ Your main application (updated with auth)
├── database.py               # ✅ Database connection and auth logic
├── setup_database.py         # ✅ Database setup script
├── requirements.txt          # ✅ Updated with auth dependencies
├── templates/
│   ├── landingPage.html      # ✅ Your existing landing page
│   ├── ngosignup.html        # ✅ NGO registration form
│   ├── ngologin.html         # ✅ NGO login form
│   ├── providersignup.html   # ✅ Medical provider registration form
│   ├── Provider Login.html   # ✅ Medical provider login form
│   ├── ngo_dashboard.html    # ✅ NGO dashboard
│   ├── provider_dashboard.html # ✅ Provider dashboard
│   ├── provider.html         # ✅ Your existing provider page
│   ├── medicalreportuplload.html # ✅ Your existing upload page
│   └── blankheader.html      # ✅ Your existing dashboard page
└── SETUP_GUIDE_APP1.md       # ✅ This guide
```

## 🎉 **You're All Set!**

Your HealthBridge application now has:
- ✅ Complete authentication system
- ✅ MySQL database integration  
- ✅ Protected access to all your existing features
- ✅ User-specific dashboards
- ✅ Secure session management

**Run `python app1.py` and start using your authenticated HealthBridge application!**
