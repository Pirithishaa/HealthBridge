import mysql.connector
from mysql.connector import Error
import os
from werkzeug.security import generate_password_hash, check_password_hash
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.config = {
            'host': 'localhost',
            'user': 'root',  
            'password': 'Piri@2005',  
            'database': 'healthbridge',
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
    
    def connect(self):
        """Establish connection to MySQL database"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
    
    def create_tables(self):
        """Create necessary tables for the application"""
        if not self.connection:
            if not self.connect():
                return False
        
        cursor = self.connection.cursor(buffered=True)
        
        try:
          
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
            
            cursor.execute(ngo_table)
            cursor.execute(provider_table)
            self.connection.commit()
            
            logger.info("Database tables created successfully")
            return True
            
        except Error as e:
            logger.error(f"Error creating tables: {e}")
            return False
        finally:
            cursor.close()
    
    def create_database(self):
        """Create the database if it doesn't exist"""
        try:
            # Connect without specifying database
            temp_config = self.config.copy()
            del temp_config['database']
            
            temp_connection = mysql.connector.connect(**temp_config)
            cursor = temp_connection.cursor(buffered=True)
            
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {self.config['database']} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            logger.info(f"Database '{self.config['database']}' created or already exists")
            
            cursor.close()
            temp_connection.close()
            return True
            
        except Error as e:
            logger.error(f"Error creating database: {e}")
            return False


class AuthManager:
    def __init__(self, db_manager):
        self.db = db_manager

    def authenticate_provider(self, email_or_id, password):
        """Authenticate medical provider by email or medical ID"""
        if not self.db.connection:
            if not self.db.connect():
                return False, None, "Database connection failed"
        
        cursor = self.db.connection.cursor(buffered=True, dictionary=True)
        try:
            cursor.execute("""
                SELECT id, full_name, email, medical_id, password_hash, specialization,
                       hospital_name, license_number, is_active
                FROM medical_providers
                WHERE email = %s OR medical_id = %s
            """, (email_or_id, email_or_id))
            
            user = cursor.fetchone()
            if user and user['is_active'] and check_password_hash(user['password_hash'], password):
                del user['password_hash']  # don’t return sensitive info
                return True, user, "Login successful"
            else:
                return False, None, "Invalid credentials"
        except Error as e:
            logger.error(f"Error authenticating medical provider: {e}")
            return False, None, "Authentication failed"
        finally:
            cursor.close()

    def authenticate_ngo(self, email, password):
        """Authenticate NGO user"""
        if not self.db.connection:
            if not self.db.connect():
                return False, None, "Database connection failed"
        
        cursor = self.db.connection.cursor(buffered=True, dictionary=True)
        try:
            cursor.execute("""
                SELECT id, full_name, email, password_hash, designation, organization, is_active
                FROM ngo_users
                WHERE email = %s
            """, (email,))
            
            user = cursor.fetchone()
            if user and user["is_active"] and check_password_hash(user["password_hash"], password):
                del user["password_hash"]  # remove sensitive field
                return True, user, "Login successful"
            else:
                return False, None, "Invalid email or password"
        except Error as e:
            logger.error(f"Error authenticating NGO user: {e}")
            return False, None, "Authentication failed"
        finally:
            cursor.close()

    def register_provider(self, full_name, email, medical_id, password, specialization=None, hospital_name=None, license_number=None):
        """Register a new medical provider"""
        if not self.db.connection:
            if not self.db.connect():
                return False, None, "Database connection failed"
        
        cursor = self.db.connection.cursor(buffered=True)
        try:
            # Check if email or medical_id already exists
            cursor.execute("SELECT id FROM medical_providers WHERE email = %s OR medical_id = %s", (email, medical_id))
            _ = cursor.fetchall()  # ✅ consume results

            if _:
                return False, None, "Email or Medical ID already registered"
            
            # Hash password
            password_hash = generate_password_hash(password)
            
            # Insert new user
            cursor.execute("""
                INSERT INTO medical_providers (full_name, email, medical_id, password_hash, specialization, hospital_name, license_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (full_name, email, medical_id, password_hash, specialization, hospital_name, license_number))
            
            self.db.connection.commit()

            # Build user object
            user = {
                "id": cursor.lastrowid,
                "full_name": full_name,
                "email": email,
                "medical_id": medical_id,
                "specialization": specialization,
                "hospital_name": hospital_name,
                "license_number": license_number
            }

            logger.info(f"Medical provider registered: {email}")
            return True, user, "Registration successful"
            
        except Error as e:
            logger.error(f"Error registering medical provider: {e}")
            return False, None, "Registration failed"
        finally:
            cursor.close()

    def register_ngo(self, full_name, email, password, designation, organization):
        """Register a new NGO user"""
        if not self.db.connection:
            if not self.db.connect():
                return False, None, "Database connection failed"
        
        cursor = self.db.connection.cursor(buffered=True)
        try:
           
            cursor.execute("SELECT id FROM ngo_users WHERE email = %s", (email,))
            _ = cursor.fetchall()  

            if _:
                return False, None, "Email already registered"
            
            
            password_hash = generate_password_hash(password)
            
           
            cursor.execute("""
                INSERT INTO ngo_users (full_name, email, password_hash, designation, organization)
                VALUES (%s, %s, %s, %s, %s)
            """, (full_name, email, password_hash, designation, organization))
            
            self.db.connection.commit()

            user = {
                "id": cursor.lastrowid,
                "full_name": full_name,
                "email": email,
                "designation": designation,
                "organization": organization
            }

            logger.info(f"NGO user registered: {email}")
            return True, user, "Registration successful"
            
        except Error as e:
            logger.error(f"Error registering NGO user: {e}")
            return False, None, "Registration failed"
        finally:
            cursor.close()



db_manager = DatabaseManager()
auth_manager = AuthManager(db_manager)
