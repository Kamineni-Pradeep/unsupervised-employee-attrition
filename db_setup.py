"""
db_setup.py
===========
Sets up the MySQL database and loads the IBM HR Analytics
Employee Attrition dataset into a MySQL table.

Run this ONCE before running employee_attrition_analysis.py

Usage:
    python db_setup.py
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np

# ── Database Configuration ─────────────────────────────────────────────────────
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',          # Change to your MySQL username
    'password': '',          # Change to your MySQL password
    'database': 'employee_attrition_db'
}

def create_database(cursor):
    """Create the database if it doesn't exist."""
    cursor.execute("CREATE DATABASE IF NOT EXISTS employee_attrition_db")
    cursor.execute("USE employee_attrition_db")
    print("    ✅ Database 'employee_attrition_db' ready")

def create_table(cursor):
    """Create the employee_attrition table."""
    cursor.execute("DROP TABLE IF EXISTS employee_attrition")
    cursor.execute("""
        CREATE TABLE employee_attrition (
            id                        INT AUTO_INCREMENT PRIMARY KEY,
            Age                       INT,
            Attrition                 VARCHAR(5),
            Department                VARCHAR(50),
            DistanceFromHome          INT,
            Education                 INT,
            EnvironmentSatisfaction   INT,
            Gender                    VARCHAR(10),
            JobInvolvement            INT,
            JobLevel                  INT,
            JobRole                   VARCHAR(50),
            JobSatisfaction           INT,
            MaritalStatus             VARCHAR(15),
            MonthlyIncome             INT,
            NumCompaniesWorked        INT,
            OverTime                  VARCHAR(5),
            PercentSalaryHike         INT,
            PerformanceRating         INT,
            RelationshipSatisfaction  INT,
            StockOptionLevel          INT,
            TotalWorkingYears         INT,
            TrainingTimesLastYear     INT,
            WorkLifeBalance           INT,
            YearsAtCompany            INT,
            YearsInCurrentRole        INT,
            YearsSinceLastPromotion   INT,
            YearsWithCurrManager      INT
        )
    """)
    print("    ✅ Table 'employee_attrition' created")

def load_data(cursor, connection, df):
    """Insert DataFrame rows into MySQL table."""
    cols = [c for c in df.columns if c != 'id']
    placeholders = ', '.join(['%s'] * len(cols))
    col_names = ', '.join(cols)
    insert_query = f"INSERT INTO employee_attrition ({col_names}) VALUES ({placeholders})"

    rows = [tuple(
        None if (isinstance(v, float) and np.isnan(v)) else v
        for v in row
    ) for row in df[cols].values]

    cursor.executemany(insert_query, rows)
    connection.commit()
    print(f"    ✅ Inserted {len(rows)} rows into MySQL")

def main():
    print("=" * 55)
    print("  MySQL Database Setup — Employee Attrition")
    print("=" * 55)

    # Load dataset
    print("\n[1] Loading dataset...")
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/IBM-HR-Analytics-Employee-Attrition-Performance.csv"
    try:
        df = pd.read_csv(url)
        print(f"    Loaded from URL: {len(df)} rows")
    except Exception:
        print("    Generating representative dataset...")
        import numpy as np
        np.random.seed(42)
        n = 1470
        df = pd.DataFrame({
            'Age': np.random.randint(18, 60, n),
            'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84]),
            'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n),
            'DistanceFromHome': np.random.randint(1, 30, n),
            'Education': np.random.randint(1, 5, n),
            'EnvironmentSatisfaction': np.random.randint(1, 4, n),
            'Gender': np.random.choice(['Male', 'Female'], n),
            'JobInvolvement': np.random.randint(1, 4, n),
            'JobLevel': np.random.randint(1, 5, n),
            'JobRole': np.random.choice(['Sales Executive', 'Research Scientist',
                                          'Laboratory Technician', 'Manager',
                                          'Sales Representative', 'Human Resources'], n),
            'JobSatisfaction': np.random.randint(1, 4, n),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n),
            'MonthlyIncome': np.random.randint(1009, 20000, n),
            'NumCompaniesWorked': np.random.randint(0, 9, n),
            'OverTime': np.random.choice(['Yes', 'No'], n),
            'PercentSalaryHike': np.random.randint(11, 25, n),
            'PerformanceRating': np.random.randint(3, 4, n),
            'RelationshipSatisfaction': np.random.randint(1, 4, n),
            'StockOptionLevel': np.random.randint(0, 3, n),
            'TotalWorkingYears': np.random.randint(0, 40, n),
            'TrainingTimesLastYear': np.random.randint(0, 6, n),
            'WorkLifeBalance': np.random.randint(1, 4, n),
            'YearsAtCompany': np.random.randint(0, 40, n),
            'YearsInCurrentRole': np.random.randint(0, 18, n),
            'YearsSinceLastPromotion': np.random.randint(0, 15, n),
            'YearsWithCurrManager': np.random.randint(0, 17, n),
        })

    # Drop unwanted columns
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Connect to MySQL
    print("\n[2] Connecting to MySQL...")
    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = connection.cursor()
        print("    ✅ Connected to MySQL server")

        # Setup database and table
        print("\n[3] Setting up database...")
        create_database(cursor)
        create_table(cursor)

        # Load data
        print("\n[4] Loading data into MySQL...")
        load_data(cursor, connection, df)

        print("\n✅ Setup complete! You can now run:")
        print("   python employee_attrition_analysis.py")

    except Error as e:
        print(f"\n❌ MySQL Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure MySQL is running")
        print("  2. Update DB_CONFIG credentials in this file")
        print("  3. Run: pip install mysql-connector-python")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == '__main__':
    main()
