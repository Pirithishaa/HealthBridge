#!/usr/bin/env python3
"""
Database Setup Script for HealthBridge
This script helps you set up the MySQL database for the HealthBridge application.
"""

import mysql.connector
from mysql.connector import Error
import getpass
import sys

def create_database_and_tables():
    """Interactive database setup"""
    print("=== HealthBridge Database Setup ===")
    print("This script will help you set up the MySQL database for HealthBridge.")
    print()
    
    # Get database connection details
    print("Please provide your MySQL connection details:")
    host = input("MySQL Host (default: localhost): ").strip() or "localhost"
    user = input("MySQL Username (default: root): ").strip() or "root"
    password = getpass.getpass("MySQL Password: ")
    database_name = input("Database Name (default: healthbridge_db): ").strip() or "healthbridge_db"
    
    print(f"\nConnecting to MySQL server at {host}...")
    
    try:
        # Connect to MySQL server (without specifying database)
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            charset='utf8mb4',
            collation='utf8mb4_unicode_ci'
        )
        
        if connection.is_connected():
            print("✅ Successfully connected to MySQL server")
            
            cursor = connection.cursor()
            
            # Create database
            print(f"Creating database '{database_name}'...")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Database '{database_name}' created or already exists")
            
            # Use the database
            cursor.execute(f"USE {database_name}")
            
            # Create NGO users table
            print("Creating NGO users table...")
            ngo_table = """
            CREATE TABLE IF NOT EXISTS ngo_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                designation ENUM('program-manager', 'director', 'volunteer') NOT NULL,
                organization VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
            """
            cursor.execute(ngo_table)
            print("✅ NGO users table created")
            
            # Create medical provider users table
            print("Creating medical providers table...")
            provider_table = """
            CREATE TABLE IF NOT EXISTS medical_providers (
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
            )
            """
            cursor.execute(provider_table)
            print("✅ Medical providers table created")
            
            # Commit changes
            connection.commit()
            
            print("\n🎉 Database setup completed successfully!")
            print(f"Database: {database_name}")
            print("Tables created:")
            print("  - ngo_users")
            print("  - medical_providers")
            
            # Show next steps
            print("\n📋 Next Steps:")
            print("1. Update the database configuration in database.py with your MySQL credentials")
            print("2. Install required Python packages: pip install -r requirements.txt")
            print("3. Run the Flask application: python app.py")
            
            # Create sample configuration
            config_content = f'''# Database Configuration for HealthBridge
# Update these values in database.py

DB_CONFIG = {{
    'host': '{host}',
    'user': '{user}',
    'password': '{password}',
    'database': '{database_name}',
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}}
'''
            
            with open('database_config.txt', 'w') as f:
                f.write(config_content)
            
            print(f"\n💾 Database configuration saved to 'database_config.txt'")
            print("   Copy these values to the 'config' dictionary in database.py")
            
    except Error as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure MySQL server is running")
        print("2. Check your username and password")
        print("3. Ensure you have permission to create databases")
        return False
        
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("\n🔌 Database connection closed")
    
    return True

def update_database_config():
    """Update database.py with the configuration"""
    try:
        with open('database_config.txt', 'r') as f:
            config_content = f.read()
        
        print("\n📝 To complete the setup, update database.py:")
        print("Replace the 'config' dictionary in database.py with:")
        print(config_content)
        
    except FileNotFoundError:
        print("Configuration file not found. Please run the setup first.")

if __name__ == "__main__":
    print("HealthBridge Database Setup")
    print("=" * 30)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--config":
        update_database_config()
    else:
        success = create_database_and_tables()
        if success:
            update_database_config()
